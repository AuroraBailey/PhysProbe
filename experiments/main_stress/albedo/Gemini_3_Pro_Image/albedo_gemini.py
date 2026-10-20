#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

# Support both direct script execution and editable package installation.
import sys as _env_sys
from pathlib import Path as _EnvPath
for _env_root in _EnvPath(__file__).resolve().parents:
    if (_env_root / "src/common/utils/env.py").is_file():
        _env_sys.path.insert(0, str(_env_root / "src"))
        _env_sys.path.insert(0, str(_env_root))
        break
from common.utils.env import load_project_env as _load_project_env
_load_project_env()

from experiments.main_stress.albedo.prompts import make_albedo_prompt


import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from common.utils.gemini_client import GEMINI_IMAGE_MODEL, GeminiImageClient, gemini_request_record

try:
    from PIL import Image
except Exception:
    Image = None

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ALBEDO_MODEL = GEMINI_IMAGE_MODEL
DEFAULT_GENERATION_MODE = "generate"
DEFAULT_GT_ROOT = os.environ.get("GT_ROOT") or str(PROJECT_ROOT / "data" / "raw")
KNOWN_DATASET_NAMES = {
    "interiorverse_mainaxis",
    "interiorverse_stresstest",
    "openroomff_mainaxis",
    "openroomff_stresstest",
}
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


DEFAULT_ALBEDO_QUALITY = "medium"

SUPPORTED_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多视角场景生成 albedo 贴图（独立版）")
    parser.add_argument("--input_dir", type=str, default="", help="Deprecated compatibility argument; --scene_list controls RGB inputs.")
    parser.add_argument("--scene_list", type=str, required=True, help="Scene list with paths relative to --dataset_root.")
    parser.add_argument("--dataset_root", type=str, default=str(Path(__file__).resolve().parents[4] / "data" / "raw"), help="Dataset root used to resolve --scene_list entries.")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "output" / "albedo"))
    parser.add_argument("--filename_suffix", type=str, default=None)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描 input_dir 下的所有子目录。",
    )
    parser.add_argument("--base_url", type=str, default=os.environ.get("AIHUBMIX_BASE_URL"), help="Aihubmix endpoint; defaults to AIHUBMIX_BASE_URL.")
    parser.add_argument("--api_key", type=str, default=None, help="Aihubmix API key")
    parser.add_argument("--albedo_api_key", type=str, default=None, help="Aihubmix API key override for albedo generation")
    parser.add_argument("--albedo_model", type=str, default=DEFAULT_ALBEDO_MODEL)
    parser.add_argument(
        "--generation_mode",
        type=str,
        default=DEFAULT_GENERATION_MODE,
        choices=["generate", "edit"],
        help="generate: 调用 Gemini 生成接口；edit: 调用编辑接口。",
    )
    parser.add_argument(
        "--generate_requires_image",
        action="store_true",
        help="generate 模式必须成功传入图片，否则直接报错（不允许降级为纯文本生成）。",
    )

    parser.add_argument(
        "--gt_root",
        type=str,
        default=DEFAULT_GT_ROOT,
        help="原始 GT 图像根目录。若 input_dir 是旧路径，会自动映射到该目录下对应数据集。",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance_scale", type=float, default=5.5, help="兼容旧命令保留参数，当前不使用。")
    parser.add_argument(
        "--albedo_size",
        type=str,
        default="1536x1024",
        help="Gemini 输出尺寸，支持 1024x1024 / 1024x1536 / 1536x1024 / auto。",
    )
    parser.add_argument("--albedo_quality", type=str, default=DEFAULT_ALBEDO_QUALITY, choices=["low", "medium", "high", "auto"])
    parser.add_argument("--watermark", action="store_true", help="兼容旧命令保留参数，当前不使用。")
    parser.add_argument("--max_generate", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="默认会跳过已生成的结果；开启后会覆盖已存在的输出。",
    )
    parser.add_argument("--num_parts", type=int, default=1)
    parser.add_argument("--part_index", type=int, default=0)

    parser.add_argument(
        "--preserve_relative_dirs",
        action="store_true",
        help="输出时保留相对于 input_dir 的子目录结构。",
    )

    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--request_retries", type=int, default=5, help="generation 阶段遇到网络抖动、超时、限流、5xx 时的重试次数。")
    parser.add_argument("--retry_backoff", type=float, default=5.0, help="generation 阶段首次重试前的等待秒数，后续按指数退避。")
    parser.add_argument("--retry_max_backoff", type=float, default=60.0, help="generation 阶段单次重试等待的最大秒数。")
    return parser.parse_args()


def validate_parts(num_parts: int, part_index: int) -> None:
    if num_parts < 1:
        raise ValueError("--num_parts 必须 >= 1")
    if part_index < 0 or part_index >= num_parts:
        raise ValueError("--part_index 必须满足 0 <= part_index < num_parts")


def shard_paths(paths: List[Path], num_parts: int, part_index: int) -> List[Path]:
    if num_parts == 1:
        return list(paths)
    return [path for idx, path in enumerate(paths) if idx % num_parts == part_index]


def list_images(input_dir: Path, filename_suffix: Optional[str] = None, recursive: bool = False) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = [p for p in sorted(iterator) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    if filename_suffix:
        images = [p for p in images if p.name.endswith(filename_suffix)]
    if not images:
        suffix_msg = f" 且文件名需以 {filename_suffix} 结尾" if filename_suffix else ""
        recursive_msg = "（已递归扫描）" if recursive else "（仅扫描第一层）"
        raise FileNotFoundError(
            f"目录中没有找到支持的图片文件{suffix_msg}{recursive_msg}: {input_dir}"
        )
    return images


def list_rgb_images_from_scene_list(scene_list_path: Path, dataset_root: Path) -> List[Path]:
    """List only RGB inputs (*_im.png or Image_*.png) from custom scenes."""
    if not scene_list_path.is_file():
        raise FileNotFoundError(f"Scene list was not found: {scene_list_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root was not found: {dataset_root}")

    scene_names = [
        line.strip()
        for line in scene_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not scene_names:
        raise RuntimeError(f"Scene list is empty: {scene_list_path}")

    images: List[Path] = []
    for scene_name in scene_names:
        scene_path = dataset_root / scene_name
        if not scene_path.is_dir():
            raise FileNotFoundError(f"Scene was not found: {scene_path}")
        preferred = sorted(path for path in scene_path.rglob("*_im.png") if path.is_file())
        if not preferred:
            preferred = sorted(path for path in scene_path.rglob("Image_*.png") if path.is_file())
        if not preferred:
            raise FileNotFoundError(f"No RGB images were found in scene: {scene_path}")
        images.extend(preferred)
    return sorted(images)


def ensure_api_key(cli_api_key: Optional[str] = None) -> str:
    api_key = cli_api_key or os.environ.get("AIHUBMIX_API_KEY")
    if api_key:
        api_key = api_key.strip()
        try:
            api_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError(
                "API key contains non-ASCII characters and cannot be written to the HTTP header. "
                "Please replace any placeholder text with the real key."
            ) from e
        return api_key
    raise RuntimeError("Missing Aihubmix API key: pass --api_key or set AIHUBMIX_API_KEY.")


def resolve_image_client_config(args: argparse.Namespace) -> Tuple[Any, str, str, Optional[str], Optional[str]]:
    api_key = ensure_api_key(getattr(args, "albedo_api_key", None) or getattr(args, "api_key", None))
    return GeminiImageClient(api_key, base_url=args.base_url, aspect_ratio="4:3", timeout=args.timeout), args.albedo_model, "aihubmix", args.base_url or "https://aihubmix.com/v1", None


def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


def file_to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    mime = guess_mime(path)
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def get_image_size(path: Path) -> Optional[Tuple[int, int]]:
    if Image is None:
        return None
    with Image.open(path) as img:
        return img.size




def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_completed_output(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = load_json_file(path)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def save_manifest(path: Path, manifest: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_manifest_entry(manifest: List[Dict[str, Any]], entry: Dict[str, Any], key_field: str = "image_name") -> None:
    entry = gemini_request_record(entry)
    entry_key = entry.get(key_field)
    if entry_key is None:
        manifest.append(entry)
        return
    for idx, item in enumerate(manifest):
        if item.get(key_field) == entry_key:
            manifest[idx] = entry
            return
    manifest.append(entry)


def extract_error_code(exc: Exception) -> str:
    text = str(exc)
    patterns = [
        r"'code':\s*'([^']+)'",
        r'"code":\s*"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return exc.__class__.__name__


def should_skip_image_error(exc: Exception) -> bool:
    text = str(exc)
    skip_patterns = [
        "InputTextSensitiveContentDetected",
        "InputImageSensitiveContentDetected",
        "SensitiveContentDetected",
        "sensitive information",
        "sensitive content",
        "moderation_blocked",
        "rejected by the safety system",
        "image_generation_user_error",
    ]
    return any(pattern in text for pattern in skip_patterns)


def should_retry_image_error(exc: Exception) -> bool:
    text = str(exc)
    retry_patterns = [
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "ConnectError",
        "ReadTimeout",
        "timed out",
        "Connection refused",
        "Connection reset",
        "connection aborted",
        "temporarily unavailable",
        "rate limit",
        "rate_limit",
        "Too Many Requests",
        '"status_code": 429',
        "'status_code': 429",
        '"status_code": 500',
        "'status_code': 500",
        '"status_code": 502',
        "'status_code': 502",
        '"status_code": 503',
        "'status_code': 503",
        '"status_code": 504',
        "'status_code': 504",
    ]
    return any(pattern in text for pattern in retry_patterns)


def build_skip_path(meta_dir: Path, num_parts: int, part_index: int) -> Path:
    filename = "skipped_images.json"
    if num_parts > 1:
        filename = f"skipped_images.part{part_index + 1}of{num_parts}.json"
    return meta_dir / filename


def record_skipped_image(
    skipped_images: List[Dict[str, Any]],
    skipped_path: Path,
    image_path: Path,
    input_dir: Path,
    stage: str,
    exc: Exception,
    args: argparse.Namespace,
) -> None:
    entry = {
        "skip_key": f"{image_path.relative_to(input_dir).as_posix()}::{stage}",
        "image_name": image_path.name,
        "relative_image_path": image_path.relative_to(input_dir).as_posix(),
        "stage": stage,
        "error_code": extract_error_code(exc),
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
        "part_index": args.part_index,
        "num_parts": args.num_parts,
        "skipped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    upsert_manifest_entry(skipped_images, entry, key_field="skip_key")
    skipped_path.write_text(json.dumps(skipped_images, ensure_ascii=False, indent=2), encoding="utf-8")


def was_image_skipped(skipped_images: List[Dict[str, Any]], image_path: Path, input_dir: Path) -> Optional[Dict[str, Any]]:
    relative_path = image_path.relative_to(input_dir).as_posix()
    for item in skipped_images:
        if item.get("relative_image_path") == relative_path:
            return item
    return None


def save_url_to_file(url: str, save_path: Path, timeout: int = 120) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def save_image_response(image_item: Any, save_path: Path, timeout: int = 120) -> None:
    url = getattr(image_item, "url", None)
    if url:
        save_url_to_file(url, save_path, timeout=timeout)
        return
    b64_json = getattr(image_item, "b64_json", None)
    if b64_json:
        save_path.write_bytes(base64.b64decode(b64_json))
        return
    raise RuntimeError("返回结果中既没有 url，也没有 b64_json，无法保存图像")


def detect_dataset_name(path_like: Path) -> Optional[str]:
    parts = [part for part in path_like.parts if part not in {path_like.anchor, ""}]
    for part in reversed(parts):
        if part in KNOWN_DATASET_NAMES:
            return part
    if path_like.name in KNOWN_DATASET_NAMES:
        return path_like.name
    return None


def resolve_listing_input_dir(input_dir: Path, gt_root: Path) -> Path:
    if input_dir.exists():
        return input_dir
    dataset_name = detect_dataset_name(input_dir)
    if dataset_name:
        mapped_dir = gt_root / dataset_name
        if mapped_dir.exists():
            return mapped_dir
    raise FileNotFoundError(
        f"input_dir 不存在，且未能在 GT 根目录下找到对应数据集: input_dir={input_dir.as_posix()} gt_root={gt_root.as_posix()}"
    )


def candidate_relative_paths(image_path: Path, input_dir: Path, dataset_name: Optional[str]) -> List[Path]:
    candidates: List[Path] = []
    seen = set()

    def add_candidate(path_like: Path) -> None:
        key = path_like.as_posix().lstrip("/")
        if key and key not in seen:
            seen.add(key)
            candidates.append(path_like)

    try:
        add_candidate(image_path.relative_to(input_dir))
    except Exception:
        pass

    parts = [part for part in image_path.parts if part not in {image_path.anchor, ""}]
    if dataset_name and dataset_name in parts:
        last_idx = max(idx for idx, part in enumerate(parts) if part == dataset_name)
        if last_idx + 1 < len(parts):
            add_candidate(Path(*parts[last_idx + 1 :]))

    for start in range(len(parts)):
        add_candidate(Path(*parts[start:]))
    return candidates


def resolve_gt_image_path(image_path: Path, input_dir: Path, gt_root: Path) -> Path:
    dataset_name = detect_dataset_name(image_path) or detect_dataset_name(input_dir)
    if dataset_name:
        dataset_root = gt_root / dataset_name
        for relative_path in candidate_relative_paths(image_path, input_dir, dataset_name):
            candidate = dataset_root / relative_path
            if candidate.exists():
                return candidate
    if image_path.exists():
        return image_path
    raise FileNotFoundError(
        f"未找到对应原始图像: image={image_path.as_posix()} gt_root={gt_root.as_posix()}"
    )


def run_gemini_image_generation(
    client: Any,
    model: str,
    prompt: str,
    source_image_path: Path,
    generation_mode: str,
    requested_size_arg: str,
    generate_requires_image: bool,
    quality: str,
    seed: int,
) -> Tuple[Any, str, Optional[Tuple[int, int]]]:
    mode = (generation_mode or DEFAULT_GENERATION_MODE).strip().lower()
    if mode not in {"generate", "edit"}:
        raise ValueError(f"不支持的 generation_mode: {generation_mode}")
    requested_size = normalize_image_size(requested_size_arg)
    common_kwargs: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "quality": quality,
        "output_format": "png",
        "n": 1,
    }
    image_size = get_image_size(source_image_path)
    if mode == "generate":
        generate_kwargs = {**common_kwargs, "size": requested_size}
        # Prefer image-conditioned generation when supported.
        with source_image_path.open("rb") as image_file:
            try:
                return client.images.generate(image=image_file, seed=seed, **generate_kwargs), requested_size, image_size
            except TypeError:
                pass
        try:
            return client.images.generate(images=[{"image_url": file_to_data_uri(source_image_path)}], seed=seed, **generate_kwargs), requested_size, image_size
        except TypeError:
            if generate_requires_image:
                raise RuntimeError("generate 模式未能成功传入图片参数，已中止以避免降级为纯文本生成。")
            try:
                return client.images.generate(**generate_kwargs), requested_size, image_size
            except TypeError:
                fallback_kwargs = dict(generate_kwargs)
                fallback_kwargs.pop("output_format", None)
                fallback_kwargs.pop("n", None)
                return client.images.generate(**fallback_kwargs), requested_size, image_size
    with source_image_path.open("rb") as image_file:
        try:
            return client.images.edit(image=image_file, size=requested_size, **common_kwargs), requested_size, image_size
        except TypeError:
            pass
    edit_kwargs = {
        **common_kwargs,
        "size": requested_size,
        "images": [{"image_url": file_to_data_uri(source_image_path)}],
    }
    try:
        return client.images.edit(**edit_kwargs), requested_size, image_size
    except TypeError:
        fallback_kwargs = dict(edit_kwargs)
        fallback_kwargs.pop("output_format", None)
        fallback_kwargs.pop("n", None)
        return client.images.edit(**fallback_kwargs), requested_size, image_size


def build_manifest_path(meta_dir: Path, num_parts: int, part_index: int) -> Path:
    manifest_name = "manifest.json"
    if num_parts > 1:
        suffix = f".part{part_index + 1}of{num_parts}"
        manifest_name = f"manifest{suffix}.json"
    return meta_dir / manifest_name


def build_image_output_path(image_path: Path, input_dir: Path, base_dir: Path, suffix: str, preserve_relative_dirs: bool) -> Path:
    if preserve_relative_dirs:
        relative_path = image_path.relative_to(input_dir)
        return base_dir / relative_path.parent / f"{image_path.stem}{suffix}"
    return base_dir / f"{image_path.stem}{suffix}"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_pending_images(image_paths: List[Path], input_dir: Path, albedo_dir: Path, overwrite: bool, preserve_relative_dirs: bool) -> List[Path]:
    if overwrite:
        return list(image_paths)
    pending = []
    for image_path in image_paths:
        out_albedo = build_image_output_path(image_path, input_dir, albedo_dir, "_albedo.png", preserve_relative_dirs)
        if is_completed_output(out_albedo):
            continue
        pending.append(image_path)
    return pending


def process_images_direct(
    args: argparse.Namespace,
    image_paths_for_generate: List[Path],
    pending_image_paths: List[Path],
    input_dir: Path,
    output_dir: Path,
    albedo_dir: Path,
    meta_dir: Path,
    image_client: Any,
    albedo_model_name: str,
    client_kind: str,
    endpoint: Optional[str],
    api_version: Optional[str],
    gt_root: Path,
    intro_text: str,
) -> None:
    print(intro_text)
    print(
        "      "
        f"input_dir={input_dir.as_posix()} | "
        f"gt_root={gt_root.as_posix()} | "
        f"filename_suffix={args.filename_suffix or 'N/A'} | "
        f"recursive={args.recursive} | "
        f"preserve_relative_dirs={args.preserve_relative_dirs} | "
        f"albedo_model={args.albedo_model} | "
        f"albedo_model_name={albedo_model_name} | "
        f"generation_mode={args.generation_mode} | "
        f"albedo_quality={args.albedo_quality} | "
        f"client_kind={client_kind} | "
        f"endpoint={endpoint or 'default'}"
    )
    manifest_path = build_manifest_path(meta_dir, args.num_parts, args.part_index)
    skipped_path = build_skip_path(meta_dir, args.num_parts, args.part_index)
    manifest = load_manifest(manifest_path)
    skipped_images = load_manifest(skipped_path)
    print(f"[2/2] 开始逐张导出 albedo（待处理 {len(pending_image_paths)} 张）")
    if not pending_image_paths:
        print("  - 检测到当前分片已全部完成，直接退出。")
    for idx, image_path in enumerate(image_paths_for_generate, start=1):
        out_albedo = build_image_output_path(image_path, input_dir, albedo_dir, "_albedo.png", args.preserve_relative_dirs)
        skipped_entry = None if args.overwrite else was_image_skipped(skipped_images, image_path, input_dir)
        if skipped_entry is not None:
            print(
                f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} "
                f"此前已跳过，原因={skipped_entry.get('error_code', 'N/A')}"
            )
            continue
        if not args.overwrite and is_completed_output(out_albedo):
            print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} 已存在，跳过")
            continue
        print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name}")
        try:
            source_image_path = resolve_gt_image_path(image_path, input_dir, gt_root)
        except Exception as exc:
            print(f"      跳过：未能定位 GT 原图 -> {exc}")
            record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "gt_lookup", exc, args)
            continue
        albedo_prompt = make_albedo_prompt("gemini")
        last_generation_exc: Optional[Exception] = None
        albedo_resp = None
        requested_size = None
        input_size = None
        for attempt in range(max(0, args.request_retries) + 1):
            try:
                albedo_resp, requested_size, input_size = run_gemini_image_generation(
                    image_client,
                    albedo_model_name,
                    albedo_prompt,
                    source_image_path,
                    args.generation_mode,
                    args.albedo_size,
                    args.generate_requires_image,
                    args.albedo_quality,
                    args.seed,
                )
                break
            except Exception as exc:
                last_generation_exc = exc
                if should_skip_image_error(exc):
                    print(f"      跳过：generation 阶段触发内容审核 -> {extract_error_code(exc)}")
                    record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation", exc, args)
                    break
                if should_retry_image_error(exc) and attempt < max(0, args.request_retries):
                    wait_seconds = min(args.retry_backoff * (2 ** attempt), args.retry_max_backoff)
                    print(
                        f"      generation 阶段异常，准备重试 "
                        f"({attempt + 1}/{max(0, args.request_retries)}) -> "
                        f"{exc.__class__.__name__}: {exc} | wait={wait_seconds:.1f}s"
                    )
                    time.sleep(max(0.0, wait_seconds))
                    continue
                if should_retry_image_error(exc):
                    print(f"      跳过：generation 阶段多次重试仍失败 -> {extract_error_code(exc)}")
                    record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation_retry_exhausted", exc, args)
                    break
                raise
        if albedo_resp is None:
            if last_generation_exc is None:
                continue
            if should_skip_image_error(last_generation_exc) or should_retry_image_error(last_generation_exc):
                continue
            raise last_generation_exc
        ensure_parent_dir(out_albedo)
        save_image_response(albedo_resp.data[0], out_albedo, timeout=args.timeout)
        upsert_manifest_entry(manifest, {
            "image_name": image_path.name,
            "relative_image_path": image_path.relative_to(input_dir).as_posix(),
            "num_parts": args.num_parts,
            "part_index": args.part_index,
            "albedo_model": args.albedo_model,
            "resolved_albedo_model": albedo_model_name,
            "client_kind": client_kind,
            "endpoint": endpoint,
            "api_version": api_version,
            "generation_mode": args.generation_mode,
            "requested_size": requested_size,
            "input_size": list(input_size) if input_size else None,
            "input_resize_policy": "model_input_native",
            "output_resolution_policy": "model_native",
            "saved_albedo_size": "model_native",
            "source_image_path": source_image_path.as_posix(),
            "albedo_output": out_albedo.relative_to(output_dir).as_posix(),
            "albedo_prompt": albedo_prompt,
            **getattr(albedo_resp, "request_parameters", {}),
        })
        save_manifest(manifest_path, manifest)
        time.sleep(max(0.0, args.sleep))
    save_manifest(manifest_path, manifest)

def main() -> None:
    args = parse_args()
    validate_parts(args.num_parts, args.part_index)
    image_client, albedo_model_name, client_kind, endpoint, api_version = resolve_image_client_config(args)
    gt_root = Path(args.gt_root)
    requested_input_dir = Path(args.dataset_root).expanduser()
    input_dir = requested_input_dir
    args.preserve_relative_dirs = True
    output_dir = Path(args.output_dir)
    albedo_dir = output_dir
    meta_dir = output_dir / "meta"
    albedo_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if args.scene_list:
        scene_list_path = Path(args.scene_list).expanduser()
        image_paths = list_rgb_images_from_scene_list(scene_list_path, input_dir)
        print(f"[info] scene_list={scene_list_path.as_posix()}")

    image_paths_for_generate_all = image_paths[: args.max_generate] if args.max_generate > 0 else image_paths
    image_paths_for_generate = shard_paths(image_paths_for_generate_all, args.num_parts, args.part_index)
    pending_image_paths = get_pending_images(
        image_paths_for_generate,
        input_dir,
        albedo_dir,
        args.overwrite,
        args.preserve_relative_dirs,
    )
    shard_label = f"part {args.part_index + 1}/{args.num_parts}"
    intro_text = (
        f"[1/2] 找到 {len(image_paths)} 张图，"
        f"当前分片 {shard_label} 负责 {len(image_paths_for_generate)} 张，待处理 {len(pending_image_paths)} 张，"
        "直接执行 Gemini 生成/编辑"
    )
    if requested_input_dir != input_dir:
        print(
            f"[info] 检测到 input_dir 已迁移，自动映射: "
            f"{requested_input_dir.as_posix()} -> {input_dir.as_posix()}"
        )
    process_images_direct(
        args=args,
        image_paths_for_generate=image_paths_for_generate,
        pending_image_paths=pending_image_paths,
        input_dir=input_dir,
        output_dir=output_dir,
        albedo_dir=albedo_dir,
        meta_dir=meta_dir,
        image_client=image_client,
        albedo_model_name=albedo_model_name,
        client_kind=client_kind,
        endpoint=endpoint,
        api_version=api_version,
        gt_root=gt_root,
        intro_text=intro_text,
    )
    print("完成。")
    print(f"albedo 输出目录: {albedo_dir.as_posix()}")
    print(f"meta 输出目录: {meta_dir.as_posix()}")





def normalize_image_size(size: Optional[str]) -> str:
    normalized = (size or "auto").strip().lower()
    if normalized not in SUPPORTED_IMAGE_SIZES:
        raise ValueError(
            f"不支持的 --albedo_size: {size}。可选值为: {', '.join(sorted(SUPPORTED_IMAGE_SIZES))}"
        )
    return normalized



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        sys.exit(130)
