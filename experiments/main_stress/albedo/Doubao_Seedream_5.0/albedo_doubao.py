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
    from volcenginesdkarkruntime import Ark
except Exception as e:
    Ark = None
    _ARK_IMPORT_ERROR = e
else:
    _ARK_IMPORT_ERROR = None

DEFAULT_BASE_URL = os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ALBEDO_MODEL = "doubao-seedream-5-0-260128"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多视角场景生成 albedo 贴图（独立版）")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--filename_suffix", type=str, default=None)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描 input_dir 下的所有子目录。",
    )
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)

    parser.add_argument("--albedo_model", type=str, default=DEFAULT_ALBEDO_MODEL)


    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--albedo_size", type=str, default="adaptive")
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
        help="将 input_dir 中每张图视为独立样本，逐张生成。",
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
    api_key = cli_api_key or os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 api key，请通过 --api_key 或环境变量 ARK_API_KEY 提供")
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


def run_image_generation(ark_client: Any, model: str, prompt: str, image_path: Path, size: str, watermark: bool, seed: int, guidance_scale: float) -> Any:
    image_data_uri = file_to_data_uri(image_path)
    model_lower = model.lower()
    if size:
        size = size.strip().lower()
    if not size or size == "adaptive":
        size = "2k"
    if "seedream" in model_lower:
        return ark_client.images.generate(model=model, prompt=prompt, image=[image_data_uri], size=size, watermark=watermark)
    try:
        return ark_client.images.generate(model=model, prompt=prompt, image=image_data_uri, seed=seed, guidance_scale=guidance_scale, size=size, watermark=watermark)
    except TypeError:
        return ark_client.images.generate(model=model, prompt=prompt, image=image_data_uri, size=size, watermark=watermark)

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

def main() -> None:
    args = parse_args()
    validate_parts(args.num_parts, args.part_index)
    api_key = ensure_api_key(args.api_key)
    if Ark is None:
        raise ImportError('未能导入 volcenginesdkarkruntime，请先安装: pip install "volcengine-python-sdk[ark]"') from _ARK_IMPORT_ERROR
    input_dir = Path(args.input_dir)
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
    ark_client = Ark(base_url=args.base_url, api_key=api_key)
    if args.independent_images:
        print(
            f"[1/2] 单图独立模式：找到 {len(image_paths)} 张图，"
            f"当前分片 {shard_label} 负责 {len(image_paths_for_generate)} 张，待处理 {len(pending_image_paths)} 张"
        )
        print(
            "      "
            f"filename_suffix={args.filename_suffix or 'N/A'} | "
            f"recursive={args.recursive} | "
            f"preserve_relative_dirs={args.preserve_relative_dirs} | "
            f"albedo_model={args.albedo_model} | "
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
            albedo_prompt = make_albedo_prompt("doubao")
            try:
                albedo_resp = run_image_generation(ark_client, args.albedo_model, albedo_prompt, image_path, args.albedo_size, args.watermark, args.seed, args.guidance_scale)
            except Exception as exc:
                if should_skip_image_error(exc):
                    print(f"      跳过：generation 阶段触发内容审核 -> {extract_error_code(exc)}")
                    record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation", exc, args)
                    continue
                raise
            ensure_parent_dir(out_albedo)
            save_image_response(albedo_resp.data[0], out_albedo, timeout=args.timeout)
            upsert_manifest_entry(manifest, {
                "image_name": image_path.name,
                "relative_image_path": image_path.relative_to(input_dir).as_posix(),
                "num_parts": args.num_parts,
                "part_index": args.part_index,
                "albedo_model": args.albedo_model,
                "albedo_output": out_albedo.relative_to(output_dir).as_posix(),
                "albedo_prompt": albedo_prompt,
            })
            time.sleep(max(0.0, args.sleep))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(
            f"[1/2] 找到 {len(image_paths)} 张图，"
            f"当前分片 {shard_label} 负责 {len(image_paths_for_generate)} 张，待处理 {len(pending_image_paths)} 张，"
            "直接执行 albedo 生成"
        )
        print(
            "      "
            f"filename_suffix={args.filename_suffix or 'N/A'} | "
            f"recursive={args.recursive} | "
            f"preserve_relative_dirs={args.preserve_relative_dirs} | "
            f"albedo_model={args.albedo_model} | "
            f"base_url={args.base_url}"
        )
        manifest_path = build_manifest_path(meta_dir, args.num_parts, args.part_index)
        skipped_path = build_skip_path(meta_dir, args.num_parts, args.part_index)
        manifest = load_manifest(manifest_path)
        skipped_images = load_manifest(skipped_path)
        if not pending_image_paths:
            print("[2/2] 检测到当前分片已全部完成，跳过生成。")
            print("完成。")
            print(f"albedo 输出目录: {albedo_dir.as_posix()}")
            print(f"meta 输出目录: {meta_dir.as_posix()}")
            return
        print(f"[2/2] 开始逐张导出 albedo（待处理 {len(pending_image_paths)} 张）")
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
            albedo_prompt = make_albedo_prompt("doubao")
            try:
                albedo_resp = run_image_generation(ark_client, args.albedo_model, albedo_prompt, image_path, args.albedo_size, args.watermark, args.seed, args.guidance_scale)
            except Exception as exc:
                if should_skip_image_error(exc):
                    print(f"      跳过：generation 阶段触发内容审核 -> {extract_error_code(exc)}")
                    record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation", exc, args)
                    continue
                raise
            ensure_parent_dir(out_albedo)
            save_image_response(albedo_resp.data[0], out_albedo, timeout=args.timeout)
            upsert_manifest_entry(manifest, {
                "image_name": image_path.name,
                "relative_image_path": image_path.relative_to(input_dir).as_posix(),
                "num_parts": args.num_parts,
                "part_index": args.part_index,
                "albedo_model": args.albedo_model,
                "albedo_output": out_albedo.relative_to(output_dir).as_posix(),
                "albedo_prompt": albedo_prompt,
            })
            time.sleep(max(0.0, args.sleep))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("完成。")
    print(f"albedo 输出目录: {albedo_dir.as_posix()}")
    print(f"meta 输出目录: {meta_dir.as_posix()}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        sys.exit(130)
