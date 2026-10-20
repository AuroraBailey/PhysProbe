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
from common.utils.image_policy import prepare_input, resized_upload
from common.utils.env import load_project_env as _load_project_env
_load_project_env()

from functools import partial as _prompt_partial
from experiments.main_stress.normal.prompts import MODEL_CONFIGS as _PROMPT_CONFIGS, make_one_shot_prompt as _prompt_builder
make_one_shot_prompt = _prompt_partial(_prompt_builder, model='gpt_image_15')


import argparse
import base64
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
try:
    from openai import AzureOpenAI
except Exception as e:
    AzureOpenAI = None
    _OPENAI_IMPORT_ERROR = e
else:
    _OPENAI_IMPORT_ERROR = None

try:
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None

DEFAULT_AZURE_ENDPOINT = None
DEFAULT_API_VERSION = "2025-04-01-preview"
DEFAULT_NORMAL_MODEL = "gpt-image-1.5"
DEFAULT_NORMAL_SIZE = "1536x1024"
DEFAULT_NORMAL_QUALITY = "high"
DEFAULT_EDIT_CANVAS_SIZE = (1536, 1024)
DEFAULT_EXAMPLE_RGB = str(Path(__file__).resolve().parents[1] / "examples/image.png")
DEFAULT_EXAMPLE_NORMAL = str(Path(__file__).resolve().parents[1] / "examples/normal.png")
DEFAULT_GT_ROOT = "/path/to/benchmark_data/GT"
DEFAULT_OUTPUT_DIR = "/path/to/benchmark_outputs/normal_gpt"
KNOWN_DATASET_NAMES = {
    "interiorverse_mainaxis",
    "interiorverse_stresstest",
    "openroomff_mainaxis",
    "openroomff_stresstest",
}
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate normal maps using a fixed reference RGB/normal pair.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input image directory.')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Output directory.')
    parser.add_argument('--filename_suffix', type=str, default='_im.png', help='Only process filenames ending with this suffix.')
    parser.add_argument('--recursive', action='store_true', help='Recursively scan input_dir.')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--azure_endpoint', '--base_url', dest='azure_endpoint', type=str, default=os.environ.get('AZURE_OPENAI_ENDPOINT', DEFAULT_AZURE_ENDPOINT), help='Azure OpenAI endpoint (shared fallback; role-specific endpoint has higher priority).')
    parser.add_argument('--api_version', type=str, default=os.environ.get('AZURE_OPENAI_API_VERSION', DEFAULT_API_VERSION))
    parser.add_argument('--gt_root', type=str, default=DEFAULT_GT_ROOT, help='Root directory of the original GT images. Legacy input_dir paths can be remapped through this root.')
    parser.add_argument('--normal_api_key', type=str, default=None)
    parser.add_argument('--normal_endpoint', '--normal_azure_endpoint', dest='normal_endpoint', type=str, default=None)
    parser.add_argument('--normal_api_version', type=str, default=None)
    parser.add_argument('--normal_model', type=str, default=DEFAULT_NORMAL_MODEL, help='Normal model.')
    parser.add_argument('--normal_deployment', type=str, default=None, help='Normal image deployment.')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--guidance_scale', type=float, default=5.5)
    parser.add_argument('--normal_size', type=str, default=DEFAULT_NORMAL_SIZE, help='Normal output size.')
    parser.add_argument('--normal_quality', type=str, default=DEFAULT_NORMAL_QUALITY, choices=['low', 'medium', 'high', 'auto'], help='Normal image quality.')
    parser.add_argument('--watermark', action='store_true', help='Keep watermark.')
    parser.add_argument('--max_generate', type=int, default=0, help='Max images to generate (0 = all).')
    parser.add_argument('--shard_count', type=int, default=1, help='Total number of stable shards for this input directory.')
    parser.add_argument('--shard_index', type=int, default=0, help='Shard index to process, in [0, shard_count).')
    parser.add_argument('--pending_start', type=int, default=0, help='Start offset within the pending image list, after completed outputs are filtered.')
    parser.add_argument('--pending_limit', type=int, default=0, help='Max pending images to process from --pending_start (0 = all remaining).')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs.')
    parser.add_argument('--independent_images', action='store_true', help='Process each query independently.')
    parser.add_argument('--preserve_relative_dirs', action='store_true', help='Preserve relative directory layout under input_dir.')
    parser.add_argument('--sleep', type=float, default=0.2, help='Sleep seconds between requests.')
    parser.add_argument('--timeout', type=int, default=120, help='Download timeout in seconds.')
    parser.add_argument('--generation_retries', type=int, default=3, help='Retry count for transient generation API failures.')
    parser.add_argument('--retry_sleep', type=float, default=5.0, help='Base sleep seconds between generation retries.')
    parser.add_argument('--example_rgb', type=str, default=DEFAULT_EXAMPLE_RGB, help='One-shot example RGB path. Default enables one-shot mode.')
    parser.add_argument('--example_normal', type=str, default=DEFAULT_EXAMPLE_NORMAL, help='One-shot example normal path. Default enables one-shot mode.')
    parser.add_argument('--save_debug_intermediates', action='store_true', help='Save an additional byte-exact provider response for debugging.')
    return parser.parse_args()


def resolve_azure_client_config(
    role: str,
    cli_api_key: Optional[str],
    cli_endpoint: Optional[str],
    cli_api_version: Optional[str],
    args: argparse.Namespace,
) -> Tuple[str, str, str]:
    role = role.lower().strip()
    if role not in {"normal"}:
        raise ValueError(f"Unknown role: {role}")

    api_key = cli_api_key or args.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Missing {role} API key: use --{role}_api_key, --api_key, or AZURE_OPENAI_API_KEY."
        )
    endpoint = cli_endpoint or args.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise ValueError("Azure endpoint is required: set AZURE_OPENAI_ENDPOINT or pass --azure_endpoint.")
    api_version = cli_api_version or args.api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or DEFAULT_API_VERSION
    return api_key, endpoint, api_version


def list_images(input_dir: Path, filename_suffix: Optional[str] = None, recursive: bool = False) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = [p for p in sorted(iterator) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    if filename_suffix:
        images = [p for p in images if p.name.endswith(filename_suffix)]
    if not images:
        suffix_msg = f" ending with {filename_suffix}" if filename_suffix else ""
        recursive_msg = " (recursive scan)" if recursive else " (top-level scan only)"
        raise FileNotFoundError(
            f"No supported image files found{suffix_msg}{recursive_msg}: {input_dir}"
        )
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


def get_image_size(path: Path) -> Optional[Tuple[int, int]]:
    if Image is None:
        return None
    with Image.open(path) as img:
        return img.size




def prepare_upload_image(src_path, dst_path, target_size=None):
    """Prepare the upload without padding; legacy name retained for callers."""
    return prepare_input(src_path, dst_path, gpt=True)




def postprocess_generated_image_bytes(image_bytes, *args, **kwargs):
    """Keep the complete provider response byte-for-byte."""
    return image_bytes


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
        f"input_dir does not exist and no matching dataset was found under gt_root: input_dir={input_dir.as_posix()} gt_root={gt_root.as_posix()}"
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
        f"Matched source image was not found: image={image_path.as_posix()} gt_root={gt_root.as_posix()}"
    )


def pil_resize_bytes(path: Path, max_side: int) -> Tuple[bytes, str]:
    if Image is None:
        data = path.read_bytes()
        return data, guess_mime(path)
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        long_side = max(w, h)
        if long_side > max_side:
            scale = max_side / float(long_side)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"


def file_to_data_uri(path: Path, max_side: Optional[int] = None) -> str:
    if max_side is not None:
        data, mime = pil_resize_bytes(path, max_side=max_side)
    else:
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


def upsert_manifest_entry(manifest: List[Dict[str, Any]], entry: Dict[str, Any], key_field: str = "relative_image_path") -> None:
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


def is_retryable_generation_error(exc: Exception) -> bool:
    error_name = type(exc).__name__
    if error_name in {
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "InternalServerError",
        "RateLimitError",
    }:
        return True
    text = str(exc).lower()
    retry_patterns = [
        "connection error",
        "connection refused",
        "connecttimeout",
        "request timed out",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "too many requests",
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
    ]
    return any(pattern in text for pattern in retry_patterns)


def build_skip_path(meta_dir: Path) -> Path:
    return meta_dir / "skipped_images.json"


def record_skipped_image(
    skipped_images: List[Dict[str, Any]],
    skipped_path: Path,
    image_path: Path,
    input_dir: Path,
    stage: str,
    exc: Exception,
) -> None:
    entry = {
        "skip_key": f"{image_path.relative_to(input_dir).as_posix()}::{stage}",
        "image_name": image_path.name,
        "relative_image_path": image_path.relative_to(input_dir).as_posix(),
        "stage": stage,
        "error_code": extract_error_code(exc),
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
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




def fetch_url_bytes(url: str, timeout: int = 120) -> bytes:
    output = io.BytesIO()
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
    return output.getvalue()


def get_image_response_bytes(image_item: Any, timeout: int = 120) -> bytes:
    url = getattr(image_item, "url", None)
    if url:
        return fetch_url_bytes(url, timeout=timeout)
    b64_json = getattr(image_item, "b64_json", None)
    if b64_json:
        return base64.b64decode(b64_json)
    raise RuntimeError("Response has neither url nor b64_json; cannot save image.")


def save_image_response(
    image_item: Any,
    save_path: Path,
    timeout: int = 120,
    crop_bbox: Optional[List[int]] = None,
    crop_reference_size: Tuple[int, int] = DEFAULT_EDIT_CANVAS_SIZE,
    final_size: Optional[Tuple[int, int]] = None,
) -> None:
    url = getattr(image_item, "url", None)
    if url:
        image_bytes = fetch_url_bytes(url, timeout=timeout)
        if crop_bbox is not None or final_size is not None:
            image_bytes = postprocess_generated_image_bytes(image_bytes)
        save_path.write_bytes(image_bytes)
        return
    b64_json = getattr(image_item, "b64_json", None)
    if b64_json:
        image_bytes = base64.b64decode(b64_json)
        if crop_bbox is not None or final_size is not None:
            image_bytes = postprocess_generated_image_bytes(image_bytes)
        save_path.write_bytes(image_bytes)
        return
    raise RuntimeError("Response has neither url nor b64_json; cannot save image.")


def normalize_normal_size(size: Optional[str]) -> str:
    if not size:
        return DEFAULT_NORMAL_SIZE
    size = size.strip().lower()
    if size in {"adaptive", "source"}:
        return DEFAULT_NORMAL_SIZE
    return size


def run_image_generation(
    client: Any,
    model: str,
    prompt: str,
    image_paths: List[Path],
    size: str,
    quality: str,
    watermark: bool,
    seed: int,
    guidance_scale: float,
) -> Any:
    del watermark, seed, guidance_scale
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    size = normalize_normal_size(size)

    def _edit_with_fallback(image_arg: Any) -> Any:
        base_kwargs: Dict[str, Any] = {
            "model": model,
            "image": image_arg,
            "prompt": prompt,
            "n": 1,
        }
        fallback_variants: List[Dict[str, Any]] = [
            {"size": size, "quality": quality, "output_format": "png", "response_format": "b64_json"},
            {"size": size, "quality": quality, "output_format": "png"},
            {"size": size, "quality": quality},
            {"size": size},
            {},
        ]
        last_error: Optional[Exception] = None
        for extra_kwargs in fallback_variants:
            try:
                return client.images.edit(**base_kwargs, **extra_kwargs,input_fidelity="high")
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                if "Unknown parameter" in str(exc):
                    last_error = exc
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("images.edit failed and no concrete fallback error was captured.")

    with contextlib.ExitStack() as stack:
        image_files = [stack.enter_context(path.open("rb")) for path in image_paths]
        image_arg: Any
        if len(image_files) == 1:
            image_arg = image_files[0]
        else:
            image_arg = image_files
        return _edit_with_fallback(image_arg)


def build_image_output_path(
    image_path: Path,
    input_dir: Path,
    base_dir: Path,
    suffix: str,
    preserve_relative_dirs: bool,
) -> Path:
    if preserve_relative_dirs:
        relative_path = image_path.relative_to(input_dir)
        return base_dir / relative_path.parent / f"{image_path.stem}{suffix}"
    return base_dir / f"{image_path.stem}{suffix}"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_pending_images(
    image_paths: List[Path],
    input_dir: Path,
    normal_dir: Path,
    overwrite: bool,
    preserve_relative_dirs: bool,
) -> List[Path]:
    if overwrite:
        return list(image_paths)
    pending: List[Path] = []
    for image_path in image_paths:
        out_normal = build_image_output_path(
            image_path=image_path,
            input_dir=input_dir,
            base_dir=normal_dir,
            suffix="_normal.png",
            preserve_relative_dirs=preserve_relative_dirs,
        )
        if is_completed_output(out_normal):
            continue
        pending.append(image_path)
    return pending


def main() -> None:
    args = parse_args()
    normal_api_key, normal_endpoint, normal_api_version = resolve_azure_client_config(role='normal', cli_api_key=args.normal_api_key, cli_endpoint=args.normal_endpoint, cli_api_version=args.normal_api_version, args=args)
    if AzureOpenAI is None:
        raise ImportError('Failed to import openai. Please install: pip install openai') from _OPENAI_IMPORT_ERROR
    example_rgb = Path(args.example_rgb) if args.example_rgb else None
    example_normal = Path(args.example_normal) if args.example_normal else None
    use_oneshot = True
    if example_rgb is None or example_normal is None:
        raise ValueError("Normal generation requires --example_rgb and --example_normal")
    if example_rgb and (not example_normal) or (example_normal and (not example_rgb)):
        raise ValueError('One-shot mode requires both --example_rgb and --example_normal.')
    if not example_rgb.exists():
        raise FileNotFoundError(f'example_rgb not found: {example_rgb}')
    if not example_normal.exists():
        raise FileNotFoundError(f'example_normal not found: {example_normal}')
    gt_root = Path(args.gt_root)
    requested_input_dir = Path(args.input_dir)
    input_dir = resolve_listing_input_dir(requested_input_dir, gt_root)
    output_dir = Path(args.output_dir)
    normal_dir = output_dir / 'normal'
    meta_dir = output_dir / 'meta'
    normal_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(input_dir, args.filename_suffix, args.recursive)
    image_paths_for_generate = image_paths[:args.max_generate] if args.max_generate > 0 else image_paths
    if args.shard_count < 1:
        raise ValueError('--shard_count must be >= 1')
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError('--shard_index must be in [0, shard_count)')
    sharded_image_paths = image_paths_for_generate[args.shard_index::args.shard_count]
    pending_image_paths = get_pending_images(image_paths=sharded_image_paths, input_dir=input_dir, normal_dir=normal_dir, overwrite=args.overwrite, preserve_relative_dirs=args.preserve_relative_dirs)
    if args.pending_start < 0:
        raise ValueError('--pending_start must be >= 0')
    if args.pending_limit < 0:
        raise ValueError('--pending_limit must be >= 0')
    pending_end = None if args.pending_limit == 0 else args.pending_start + args.pending_limit
    selected_image_paths = pending_image_paths[args.pending_start:pending_end]
    one_shot_prompt = make_one_shot_prompt()
    print(f'[1/3] Found {len(image_paths)} images; {len(image_paths_for_generate)} candidate generations; shard candidates: {len(sharded_image_paths)}; pending: {len(pending_image_paths)}; processing in this run: {len(selected_image_paths)}; shard={args.shard_index}/{args.shard_count}; pending_start={args.pending_start} pending_limit={args.pending_limit}; one_shot={use_oneshot}')
    print(f'      filename_suffix={args.filename_suffix or "N/A"} | recursive={args.recursive} | preserve_relative_dirs={args.preserve_relative_dirs} | independent_images={args.independent_images} | overwrite={args.overwrite} | normal_model={args.normal_model} | normal_deployment={args.normal_deployment or args.normal_model} | normal_endpoint={normal_endpoint} | normal_api_version={normal_api_version} | normal_quality={args.normal_quality} | normal_size={normalize_normal_size(args.normal_size)} | gt_root={gt_root.as_posix()}')
    normal_deployment = args.normal_deployment or args.normal_model
    normal_client = AzureOpenAI(api_version=normal_api_version, azure_endpoint=normal_endpoint, api_key=normal_api_key)
    manifest_path = meta_dir / 'manifest.json'
    skipped_path = build_skip_path(meta_dir)
    manifest = load_manifest(manifest_path)
    skipped_images = load_manifest(skipped_path)
    if requested_input_dir != input_dir:
        print(f'[info] Detected a migrated input_dir and remapped it automatically: {requested_input_dir.as_posix()} -> {input_dir.as_posix()}')
    meta_info = {'mode': 'one_shot_query_first_edgepad_v1', 'prompt_source': 'make_one_shot_prompt_query_first_edgepad', 'example_rgb': str(example_rgb), 'example_normal': str(example_normal)}
    (meta_dir / "reference_pair.json").write_text(json.dumps(meta_info, indent=2) + "\n", encoding="utf-8")
    print('[2/3] Generate each query using the selected reference pair and imported prompt.')
    print(f'[3/3] Start exporting normal maps ({len(selected_image_paths)} images)')
    if not pending_image_paths:
        print('  - The current directory is already complete; exiting.')
    elif not selected_image_paths:
        print('  - The current pending slice is empty; exiting.')
    for idx, image_path in enumerate(selected_image_paths, start=1):
        out_normal = build_image_output_path(image_path=image_path, input_dir=input_dir, base_dir=normal_dir, suffix='_normal.png', preserve_relative_dirs=args.preserve_relative_dirs)
        skipped_entry = None if args.overwrite else was_image_skipped(skipped_images, image_path, input_dir)
        if skipped_entry is not None:
            print(f'  - ({idx}/{len(selected_image_paths)}) {image_path.name} previously skipped, reason={skipped_entry.get("error_code", "N/A")}')
            continue
        if not args.overwrite and is_completed_output(out_normal):
            print(f'  - ({idx}/{len(selected_image_paths)}) {image_path.name} already exists; skipping')
            continue
        print(f'  - ({idx}/{len(selected_image_paths)}) {image_path.name}')
        try:
            normal_prompt = one_shot_prompt
            source_image_path = resolve_gt_image_path(image_path, input_dir, gt_root)
        except Exception as exc:
            print(f'      Skip: failed to read the GT source image -> {exc}')
            record_skipped_image(skipped_images, skipped_path, image_path, input_dir, 'gt_lookup', exc)
            continue
        try:
            with tempfile.TemporaryDirectory(prefix='gpt_edit_pad_') as temp_dir_str:
                temp_dir = Path(temp_dir_str)
                padded_example_rgb = temp_dir / 'example_rgb.png'
                padded_example_normal = temp_dir / 'example_normal.png'
                padded_query = temp_dir / 'query.png'
                example_rgb_bbox = prepare_upload_image(example_rgb, padded_example_rgb)
                example_normal_bbox = prepare_upload_image(example_normal, padded_example_normal)
                query_bbox = prepare_upload_image(source_image_path, padded_query)
                normal_input_paths = [padded_query, padded_example_rgb, padded_example_normal]
                max_attempts = max(1, args.generation_retries + 1)
                for attempt in range(1, max_attempts + 1):
                    try:
                        normal_resp = run_image_generation(client=normal_client, model=normal_deployment, prompt=normal_prompt, image_paths=normal_input_paths, size=args.normal_size, quality=args.normal_quality, watermark=args.watermark, seed=args.seed, guidance_scale=args.guidance_scale)
                        break
                    except Exception as exc:
                        if attempt < max_attempts and is_retryable_generation_error(exc):
                            wait_seconds = max(0.0, args.retry_sleep) * attempt
                            print(f'      Retry: transient generation error {type(exc).__name__}, {wait_seconds:.1f}s before retry {attempt}/{max_attempts - 1}')
                            time.sleep(wait_seconds)
                            continue
                        raise
            ensure_parent_dir(out_normal)
            raw_bytes = get_image_response_bytes(normal_resp.data[0], timeout=args.timeout)
            if args.save_debug_intermediates:
                debug_full = build_image_output_path(image_path=image_path, input_dir=input_dir, base_dir=normal_dir, suffix='_normal_full.png', preserve_relative_dirs=args.preserve_relative_dirs)
                ensure_parent_dir(debug_full)
                debug_full.write_bytes(raw_bytes)
            out_normal.write_bytes(postprocess_generated_image_bytes(raw_bytes))
        except Exception as exc:
            if should_skip_image_error(exc):
                print(f'      Skip: generation stage triggered a skippable error -> {extract_error_code(exc)}')
                record_skipped_image(skipped_images, skipped_path, image_path, input_dir, 'generation', exc)
                continue
            raise
        upsert_manifest_entry(manifest, {'image_name': image_path.name, 'relative_image_path': image_path.relative_to(input_dir).as_posix(), 'normal_model': args.normal_model, 'normal_deployment': normal_deployment, 'normal_endpoint': normal_endpoint, 'normal_api_version': normal_api_version, 'normal_quality': args.normal_quality, 'normal_size': normalize_normal_size(args.normal_size), 'preprocess_canvas_size': f'{DEFAULT_EDIT_CANVAS_SIZE[0]}x{DEFAULT_EDIT_CANVAS_SIZE[1]}', 'saved_normal_size': 'model_native', 'mode': 'one_shot_v3k_colorsem_texturepatch_v4', 'example_rgb': str(example_rgb) if example_rgb else '', 'example_normal': str(example_normal) if example_normal else '', 'example_rgb_content_bbox': example_rgb_bbox, 'example_normal_content_bbox': example_normal_bbox, 'query_content_bbox': query_bbox, 'source_image_path': source_image_path.as_posix(), 'normal_output': out_normal.relative_to(output_dir).as_posix(), 'normal_prompt': normal_prompt, 'input_order': ['query_rgb', 'example_rgb', 'example_normal'], 'padding_mode': 'none', 'save_debug_intermediates': bool(args.save_debug_intermediates)})
        time.sleep(max(0.0, args.sleep))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Done.')
    print(f'Normal output dir: {normal_dir.as_posix()}')
    print(f'Meta output dir: {meta_dir.as_posix()}')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\nInterrupted.", file=sys.stderr)
        sys.exit(130)
