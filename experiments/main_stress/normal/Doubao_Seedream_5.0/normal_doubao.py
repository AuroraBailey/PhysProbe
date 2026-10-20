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

from functools import partial as _prompt_partial
from experiments.main_stress.normal.prompts import MODEL_CONFIGS as _PROMPT_CONFIGS, make_one_shot_prompt as _prompt_builder
make_one_shot_prompt = _prompt_partial(_prompt_builder, model='doubao')


import argparse
import base64
import io
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

try:
    from volcenginesdkarkruntime import Ark
except Exception as e:
    Ark = None
    _ARK_IMPORT_ERROR = e
else:
    _ARK_IMPORT_ERROR = None

DEFAULT_BASE_URL = os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_NORMAL_MODEL = "doubao-seedream-5-0-260128"
DEFAULT_EXAMPLE_RGB = str(Path(__file__).resolve().parents[1] / "examples/image.png")
DEFAULT_EXAMPLE_NORMAL = str(Path(__file__).resolve().parents[1] / "examples/normal.png")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate normal maps using a fixed reference RGB/normal pair.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input image directory.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('--filename_suffix', type=str, default='_im.png', help='Only process filenames ending with this suffix.')
    parser.add_argument('--recursive', action='store_true', help='Recursively scan input_dir.')
    parser.add_argument('--base_url', type=str, default=DEFAULT_BASE_URL, help='Ark base URL.')
    parser.add_argument('--normal_model', type=str, default=DEFAULT_NORMAL_MODEL, help='Normal model.')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--guidance_scale', type=float, default=5.5)
    parser.add_argument('--normal_size', type=str, default='adaptive', help='Normal output size.')
    parser.add_argument('--watermark', action='store_true', help='Keep watermark.')
    parser.add_argument('--max_generate', type=int, default=0, help='Max images to generate (0 = all).')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs.')
    parser.add_argument('--independent_images', action='store_true', help='Process each query independently.')
    parser.add_argument('--preserve_relative_dirs', action='store_true', help='Preserve relative directory layout under input_dir.')
    parser.add_argument('--sleep', type=float, default=0.2, help='Sleep seconds between requests.')
    parser.add_argument('--timeout', type=int, default=120, help='Download timeout in seconds.')
    parser.add_argument('--example_rgb', type=str, default=DEFAULT_EXAMPLE_RGB, help='One-shot example RGB path. Default enables one-shot mode.')
    parser.add_argument('--example_normal', type=str, default=DEFAULT_EXAMPLE_NORMAL, help='One-shot example normal path. Default enables one-shot mode.')
    return parser.parse_args()


def ensure_api_key() -> str:
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ARK_API_KEY in the environment.")
    return api_key


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
    raise RuntimeError("Response has neither url nor b64_json; cannot save image.")


def run_image_generation(
    ark_client: Any,
    model: str,
    prompt: str,
    image_paths: List[Path],
    size: str,
    watermark: bool,
    seed: int,
    guidance_scale: float,
) -> Any:
    image_uris = [file_to_data_uri(p, max_side=None) for p in image_paths]
    model_lower = model.lower()

    if size:
        size = size.strip().lower()
    if not size or size == "adaptive":
        size = "2k"

    if "seedream" in model_lower:
        return ark_client.images.generate(
            model=model,
            prompt=prompt,
            image=image_uris,
            size=size,
            watermark=watermark,
        )

    try:
        return ark_client.images.generate(
            model=model,
            prompt=prompt,
            image=image_uris if len(image_uris) > 1 else image_uris[0],
            seed=seed,
            guidance_scale=guidance_scale,
            size=size,
            watermark=watermark,
        )
    except TypeError as exc:
        raise RuntimeError("Normal generation requires preserving the complete query-first input list; backend rejected the request.") from exc


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
    api_key = ensure_api_key()
    if Ark is None:
        raise ImportError('Failed to import volcenginesdkarkruntime. Please install: pip install "volcengine-python-sdk[ark]"') from _ARK_IMPORT_ERROR
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
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    normal_dir = output_dir / 'normal'
    meta_dir = output_dir / 'meta'
    normal_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(input_dir, args.filename_suffix, args.recursive)
    image_paths_for_generate = image_paths[:args.max_generate] if args.max_generate > 0 else image_paths
    pending_image_paths = get_pending_images(image_paths=image_paths_for_generate, input_dir=input_dir, normal_dir=normal_dir, overwrite=args.overwrite, preserve_relative_dirs=args.preserve_relative_dirs)
    print(f'[1/3] Found {len(image_paths)} images; candidate generations: {len(image_paths_for_generate)}; pending: {len(pending_image_paths)}; one_shot={use_oneshot}')
    print(f'      filename_suffix={args.filename_suffix or "N/A"} | recursive={args.recursive} | preserve_relative_dirs={args.preserve_relative_dirs} | independent_images={args.independent_images} | overwrite={args.overwrite}')
    ark_client = Ark(base_url=args.base_url, api_key=api_key)
    manifest_path = meta_dir / 'manifest.json'
    skipped_path = build_skip_path(meta_dir)
    manifest = load_manifest(manifest_path)
    skipped_images = load_manifest(skipped_path)
    print('[2/3] Process each query independently using the selected reference pair.')
    print(f'[3/3] Start exporting normal maps one by one (pending: {len(pending_image_paths)} images)')
    if not pending_image_paths:
        print('  - The current directory is already complete; exiting.')
    for idx, image_path in enumerate(image_paths_for_generate, start=1):
        out_normal = build_image_output_path(image_path=image_path, input_dir=input_dir, base_dir=normal_dir, suffix='_normal.png', preserve_relative_dirs=args.preserve_relative_dirs)
        skipped_entry = None if args.overwrite else was_image_skipped(skipped_images, image_path, input_dir)
        if skipped_entry is not None:
            print(f'  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} previously skipped, reason={skipped_entry.get("error_code", "N/A")}')
            continue
        if not args.overwrite and is_completed_output(out_normal):
            print(f'  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} already exists; skipping')
            continue
        print(f'  - ({idx}/{len(image_paths_for_generate)}) {image_path.name}')
        normal_prompt = make_one_shot_prompt()
        normal_input_paths = [image_path, example_rgb, example_normal]
        try:
            normal_resp = run_image_generation(ark_client=ark_client, model=args.normal_model, prompt=normal_prompt, image_paths=normal_input_paths, size=args.normal_size, watermark=args.watermark, seed=args.seed, guidance_scale=args.guidance_scale)
            ensure_parent_dir(out_normal)
            save_image_response(normal_resp.data[0], out_normal, timeout=args.timeout)
        except Exception as exc:
            if should_skip_image_error(exc):
                print(f'      Skip: generation stage triggered a skippable error -> {extract_error_code(exc)}')
                record_skipped_image(skipped_images, skipped_path, image_path, input_dir, 'generation', exc)
                continue
            raise
        upsert_manifest_entry(manifest, {
            'relative_image_path': image_path.relative_to(input_dir).as_posix(),
            'normal_model': args.normal_model, 'requested_size': args.normal_size,
            'seed': args.seed, 'seed_sent': 'seedream' not in args.normal_model.lower(),
            'normal_prompt': normal_prompt, 'status': 'completed',
            'input_order': ['query_rgb', 'example_rgb', 'example_normal'],
            'input_paths': [str(path) for path in normal_input_paths],
            'normal_output': out_normal.relative_to(output_dir).as_posix(),
            'output_policy': 'provider_bytes',
        })
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
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
