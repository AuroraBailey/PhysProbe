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

try:
    from PIL import Image
except Exception:
    Image = None

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_ALBEDO_MODEL = "wan2.7-image"
DEFAULT_GENERATION_MODE = "edit"
DEFAULT_GT_ROOT = os.environ.get("GT_ROOT") or str(PROJECT_ROOT / "data" / "raw")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "main_stress" / "albedo" / "wan")
KNOWN_DATASET_NAMES = {
    "interiorverse_mainaxis",
    "interiorverse_stresstest",
    "openroomff_mainaxis",
    "openroomff_stresstest",
}
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
WAN_MIN_PIXELS = 768 * 768
WAN_MAX_PIXELS = 2048 * 2048
DEFAULT_WAN_OUTPUT_WIDTH = 1280
DEFAULT_WAN_OUTPUT_HEIGHT = 960
DEFAULT_WAN_OUTPUT_SIZE = f"{DEFAULT_WAN_OUTPUT_WIDTH}*{DEFAULT_WAN_OUTPUT_HEIGHT}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多视角场景生成 albedo 贴图（独立版）")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--filename_suffix", type=str, default="_im.png")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描 input_dir 下的所有子目录。",
    )
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))

    parser.add_argument("--albedo_model", type=str, default=DEFAULT_ALBEDO_MODEL)
    parser.add_argument(
        "--generation_mode",
        type=str,
        default=DEFAULT_GENERATION_MODE,
        choices=["generate", "edit"],
        help="generate: 调用 wan2.7-image 生成接口；edit: 调用编辑接口。",
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
        default=DEFAULT_WAN_OUTPUT_SIZE,
        help="兼容旧命令保留参数，当前固定输出 1280*960。",
    )
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--max_generate", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="默认会跳过已生成的结果；开启后会覆盖已存在的输出。",
    )
    parser.add_argument("--num_parts", type=int, default=1)
    parser.add_argument("--part_index", type=int, default=0)
    parser.add_argument(
        "--independent_images",
        action="store_true",
        help="兼容旧命令保留参数。当前无论是否开启，都会逐图生成。",
    )
    parser.add_argument(
        "--preserve_relative_dirs",
        action="store_true",
        help="输出时保留相对于 input_dir 的子目录结构。",
    )

    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def ensure_api_key(cli_api_key: Optional[str] = None) -> str:
    api_key = cli_api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 DashScope API Key，请通过 --api_key 或环境变量 DASHSCOPE_API_KEY 提供。")
    return api_key


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
        raise FileNotFoundError(f"目录中没有找到支持的图片文件{suffix_msg}{recursive_msg}: {input_dir}")
    return images


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


def get_image_size(path: Path) -> Tuple[int, int]:
    if Image is None:
        raise ImportError('未能导入 Pillow，请先安装: pip install Pillow')
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


def upsert_manifest_entry(manifest: List[Dict[str, Any]], entry: Dict[str, Any], key_field: str = "image_name") -> None:
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
    ]
    return any(pattern in text for pattern in skip_patterns)


def is_skippable_http_client_error(exc: Exception) -> bool:
    return bool(re.match(r"HTTP 4\d\d\b", str(exc)))


def summarize_http_error(response: requests.Response, max_len: int = 4000) -> str:
    try:
        body_text = json.dumps(response.json(), ensure_ascii=False)
    except Exception:
        body_text = response.text.strip()
    body_text = body_text.strip()
    if len(body_text) > max_len:
        body_text = body_text[:max_len] + "...(truncated)"
    reason = response.reason or ""
    return f"HTTP {response.status_code} {reason}: {body_text}" if body_text else f"HTTP {response.status_code} {reason}"


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


def validate_wan_output_size(width: int, height: int) -> None:
    pixels = width * height
    if pixels < WAN_MIN_PIXELS or pixels > WAN_MAX_PIXELS:
        raise ValueError(
            f"wan2.7-image 输出总像素必须在 [{WAN_MIN_PIXELS}, {WAN_MAX_PIXELS}] 内，"
            f"当前原图尺寸为 {width}x{height}，总像素={pixels}"
        )
    ratio = width / float(height)
    if ratio < 1.0 / 8.0 or ratio > 8.0:
        raise ValueError(f"wan2.7-image 输出宽高比必须在 [1:8, 8:1] 内，当前原图尺寸为 {width}x{height}")


def build_wan_size() -> str:
    validate_wan_output_size(DEFAULT_WAN_OUTPUT_WIDTH, DEFAULT_WAN_OUTPUT_HEIGHT)
    return DEFAULT_WAN_OUTPUT_SIZE


def extract_wan_image_url(response_json: Dict[str, Any]) -> str:
    for choice in response_json.get("output", {}).get("choices", []):
        message = choice.get("message", {})
        for item in message.get("content", []):
            if item.get("type") == "image" and item.get("image"):
                return item["image"]
            if item.get("image"):
                return item["image"]
    raise RuntimeError(f"模型返回中未找到图像 URL: {json.dumps(response_json, ensure_ascii=False)}")


def save_wan_image_response(response_json: Dict[str, Any], save_path: Path, timeout: int = 120) -> str:
    image_url = extract_wan_image_url(response_json)
    save_url_to_file(image_url, save_path, timeout=timeout)
    return image_url


def run_wan_image_generation(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    source_image_path: Path,
    generation_mode: str,
    watermark: bool,
    seed: int,
    timeout: int,
) -> Tuple[Dict[str, Any], str, Tuple[int, int]]:
    if model != "wan2.7-image":
        raise ValueError("This runner requires wan2.7-image")
    mode = (generation_mode or DEFAULT_GENERATION_MODE).strip().lower()
    if mode not in {"generate", "edit"}:
        raise ValueError(f"不支持的 generation_mode: {generation_mode}")
    requested_size = build_wan_size()
    input_size = get_image_size(source_image_path)
    content: List[Dict[str, Any]] = []
    if mode == "edit":
        content.append({"image": file_to_data_uri(source_image_path)})
    content.append({"text": prompt})
    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        },
        "parameters": {
            "size": requested_size,
            "n": 1,
            "seed": seed,
            "watermark": watermark,
        },
    }
    response = requests.post(
        base_url.rstrip("/") + "/services/aigc/multimodal-generation/generation",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(summarize_http_error(response))
    response_json = response.json()
    if response_json.get("code"):
        raise RuntimeError(f"{response_json.get('code')}: {response_json.get('message', '')}")
    return response_json, requested_size, input_size


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


def process_images(
    args: argparse.Namespace,
    image_paths_for_generate: List[Path],
    pending_image_paths: List[Path],
    input_dir: Path,
    output_dir: Path,
    albedo_dir: Path,
    meta_dir: Path,
    api_key: str,
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
        f"generation_mode={args.generation_mode} | "
        f"base_url={args.base_url}"
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
        albedo_prompt = make_albedo_prompt("wan")
        try:
            albedo_resp, requested_size, input_size = run_wan_image_generation(
                api_key,
                args.base_url,
                args.albedo_model,
                albedo_prompt,
                source_image_path,
                args.generation_mode,
                args.watermark,
                args.seed,
                args.timeout,
            )
        except Exception as exc:
            if should_skip_image_error(exc):
                print(f"      跳过：generation 阶段触发内容审核 -> {extract_error_code(exc)}")
                record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation", exc, args)
                continue
            if is_skippable_http_client_error(exc):
                print(f"      跳过：generation 请求失败 -> {exc}")
                record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation_http", exc, args)
                continue
            raise
        ensure_parent_dir(out_albedo)
        image_url = save_wan_image_response(albedo_resp, out_albedo, timeout=args.timeout)
        usage = albedo_resp.get("usage", {}) if isinstance(albedo_resp, dict) else {}
        upsert_manifest_entry(manifest, {
            "image_name": image_path.name,
            "relative_image_path": image_path.relative_to(input_dir).as_posix(),
            "num_parts": args.num_parts,
            "part_index": args.part_index,
            "albedo_model": args.albedo_model,
            "base_url": args.base_url,
            "generation_mode": args.generation_mode,
            "requested_size": requested_size,
            "input_size": list(input_size),
            "returned_size": usage.get("size"),
            "source_image_path": source_image_path.as_posix(),
            "albedo_output": out_albedo.relative_to(output_dir).as_posix(),
            "image_url": image_url,
            "request_id": albedo_resp.get("request_id") if isinstance(albedo_resp, dict) else None,
            "albedo_prompt": albedo_prompt,
        })
        time.sleep(max(0.0, args.sleep))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_parts(args.num_parts, args.part_index)
    api_key = ensure_api_key(args.api_key)
    gt_root = Path(args.gt_root)
    requested_input_dir = Path(args.input_dir)
    input_dir = resolve_listing_input_dir(requested_input_dir, gt_root)
    output_dir = Path(args.output_dir)
    albedo_dir = output_dir / "albedo"
    meta_dir = output_dir / "meta"
    albedo_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(input_dir, args.filename_suffix, args.recursive)
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
        "执行 wan2.7-image 生成/编辑"
    )
    if requested_input_dir != input_dir:
        print(
            f"[info] 检测到 input_dir 已迁移，自动映射: "
            f"{requested_input_dir.as_posix()} -> {input_dir.as_posix()}"
        )
    process_images(
        args=args,
        image_paths_for_generate=image_paths_for_generate,
        pending_image_paths=pending_image_paths,
        input_dir=input_dir,
        output_dir=output_dir,
        albedo_dir=albedo_dir,
        meta_dir=meta_dir,
        api_key=api_key,
        gt_root=gt_root,
        intro_text=intro_text,
    )
    print("完成。")
    print(f"albedo 输出目录: {albedo_dir.as_posix()}")
    print(f"meta 输出目录: {meta_dir.as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        sys.exit(130)
