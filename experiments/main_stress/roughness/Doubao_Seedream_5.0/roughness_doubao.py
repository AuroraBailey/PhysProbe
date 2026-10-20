#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Main pipeline: direct roughness generation from RGB plus an externally provided
# segmentation prior. In our experiments, the prior masks are SAM3 outputs, but
# this script only consumes the prior and does not generate segmentation itself.

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
from experiments.main_stress.roughness.prompts import MODEL_CONFIGS as _PROMPT_CONFIGS, build_roughness_prompt as _prompt_builder
build_roughness_prompt = _prompt_partial(_prompt_builder, model='doubao')
PROMPT_VERSION = _PROMPT_CONFIGS['doubao']['PROMPT_VERSION']


import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

try:
    from volcenginesdkarkruntime import Ark
except Exception as e:
    Ark = None
    _ARK_IMPORT_ERROR = e
else:
    _ARK_IMPORT_ERROR = None


DEFAULT_BASE_URL = os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_IMAGE_MODEL = "doubao-seedream-5-0-260128"
DEFAULT_TIMEOUT = 1800
SEG_CANDIDATE_SUFFIXES = ["_seg", "_mask", "_sam", "_semantic", "_label", ""]
INPUT_MODE_CHOICES = ("rgb_only",)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate roughness maps with SeedDream from RGB-only, RGB+example, or RGB+seg inputs.')
    parser.add_argument('--input_dir', type=str, required=True, help='RGB image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--input_mode', type=str, default='rgb_only', choices=INPUT_MODE_CHOICES, help='Main/stress RGB-only generation.')
    parser.add_argument('--base_url', type=str, default=DEFAULT_BASE_URL, help='Ark base URL')
    parser.add_argument('--image_model', type=str, default=DEFAULT_IMAGE_MODEL, help='Image generation model')
    parser.add_argument('--size', type=str, default='adaptive', help='Output size; adaptive -> 2k')
    parser.add_argument('--watermark', action='store_true', help='Keep watermark')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='Request and download timeout')
    parser.add_argument('--sleep', type=float, default=0.2, help='Sleep between requests')
    parser.add_argument('--max_generate', type=int, default=0, help='0 means process all images')
    parser.add_argument('--filename_suffix', type=str, default=None, help='Only process files ending with this suffix')
    parser.add_argument('--recursive', action='store_true', help='Recursively scan input_dir')
    parser.add_argument('--preserve_relative_dirs', action='store_true', help='Preserve paths relative to input_dir')
    parser.add_argument('--num_parts', type=int, default=1, help='Number of shards for parallel runs')
    parser.add_argument('--part_index', type=int, default=0, help='Shard index, 0-based')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate existing outputs. By default, completed outputs are skipped for resume.')
    parser.add_argument('--skip_existing', action='store_true', help='Compatibility flag. Existing completed outputs are skipped by default unless --overwrite is set.')
    return parser.parse_args()


def ensure_api_key() -> str:
    api_key = os.environ.get("ARK_API_KEY")
    if api_key:
        api_key = api_key.strip()
        try:
            api_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError(
                "ARK_API_KEY contains non-ASCII characters and cannot be written to the HTTP header. "
                "Please replace any placeholder text with the real key."
            ) from e
        return api_key
    raise RuntimeError("Missing authentication: please provide ARK_API_KEY.")


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


def validate_parts(num_parts: int, part_index: int) -> None:
    if num_parts < 1:
        raise ValueError("--num_parts must be >= 1")
    if part_index < 0 or part_index >= num_parts:
        raise ValueError("--part_index must satisfy 0 <= part_index < --num_parts")


def shard_paths(paths: List[Path], num_parts: int, part_index: int) -> List[Path]:
    if num_parts == 1:
        return list(paths)
    return [path for idx, path in enumerate(paths) if idx % num_parts == part_index]


def list_images(input_dir: Path, filename_suffix: Optional[str] = None, recursive: bool = False) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    candidates = [p for p in sorted(iterator) if p.is_file()]

    if filename_suffix:
        images = [
            p for p in candidates
            if p.suffix.lower() in SUPPORTED_EXTS and p.name.endswith(filename_suffix)
        ]
        if images:
            return images
        recursive_msg = " recursively" if recursive else ""
        raise FileNotFoundError(
            f"No RGB images ending with {filename_suffix} were found{recursive_msg} in {input_dir}"
        )

    preferred = [
        p for p in candidates
        if p.suffix.lower() == ".png" and p.name.lower().endswith("_im.png")
    ]
    if preferred:
        return preferred

    image_prefix = [
        p for p in candidates
        if p.is_file() and p.suffix.lower() == ".png" and p.name.lower().startswith("image_")
    ]
    if image_prefix:
        return image_prefix

    recursive_msg = " recursively" if recursive else ""
    raise FileNotFoundError(f"No RGB images matching *_im.png or Image_*.png were found{recursive_msg} in {input_dir}")


def run_image_generation(
    ark_client: Any,
    model: str,
    prompt: str,
    image_paths: List[Path],
    size: str,
    watermark: bool,
) -> Any:
    image_uris = [file_to_data_uri(p) for p in image_paths]
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

    return ark_client.images.generate(
        model=model,
        prompt=prompt,
        image=image_uris,
        size=size,
        watermark=watermark,
    )


def save_url_to_file(url: str, save_path: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def save_image_response(image_item: Any, save_path: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    url = getattr(image_item, "url", None)
    if url:
        save_url_to_file(url, save_path, timeout=timeout)
        return

    b64_json = getattr(image_item, "b64_json", None)
    if b64_json:
        save_path.write_bytes(base64.b64decode(b64_json))
        return

    if isinstance(image_item, dict):
        if image_item.get("url"):
            save_url_to_file(image_item["url"], save_path, timeout=timeout)
            return
        if image_item.get("b64_json"):
            save_path.write_bytes(base64.b64decode(image_item["b64_json"]))
            return

    raise RuntimeError("Image response contains neither url nor b64_json.")




def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def write_manifest(path: Path, manifest: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def upsert_manifest_entry(manifest: List[Dict[str, Any]], item: Dict[str, Any], key_field: str = "image_name") -> None:
    item_key = str(item.get(key_field, "") or item.get("image_name", "")).strip()
    if not item_key:
        manifest.append(item)
        return
    for idx, existing in enumerate(manifest):
        existing_key = str(existing.get(key_field, "") or existing.get("image_name", "")).strip()
        if existing_key == item_key:
            manifest[idx] = item
            return
    manifest.append(item)


def build_run_signature(*, input_mode: str, image_model: str, prompt_text: str, timeout: int, size: str, watermark: bool) -> Dict[str, Any]:
    route = 'seedream_rgb_only'
    return {'image_model': image_model, 'route': route, 'input_mode': input_mode, 'prompt_version': PROMPT_VERSION, 'prompt_text': prompt_text, 'segmentation_prior': 'unused', 'segmentation_assumption': 'unused', 'timeout': timeout, 'size': size, 'output_resolution_policy': 'model_native_output', 'watermark': watermark}


def should_skip_existing_output(
    *,
    roughness_path: Path,
    output_meta_path: Path,
    run_signature: Dict[str, Any],
) -> bool:
    del output_meta_path, run_signature
    return is_completed_output(roughness_path)


def is_completed_output(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def build_part_path(meta_dir: Path, stem: str, ext: str, num_parts: int, part_index: int) -> Path:
    filename = f"{stem}.{ext}"
    if num_parts > 1:
        filename = f"{stem}.part{part_index + 1}of{num_parts}.{ext}"
    return meta_dir / filename


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


def relative_output(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return path.as_posix()


def get_pending_images(
    image_paths: List[Path],
    input_dir: Path,
    roughness_dir: Path,
    output_meta_dir: Path,
    run_signature: Dict[str, Any],
    overwrite: bool,
    preserve_relative_dirs: bool,
) -> List[Path]:
    if overwrite:
        return list(image_paths)
    pending = []
    for rgb_path in image_paths:
        roughness_path = build_image_output_path(
            rgb_path, input_dir, roughness_dir, "_roughness.png", preserve_relative_dirs
        )
        output_meta_path = build_image_output_path(
            rgb_path, input_dir, output_meta_dir, "_roughness.json", preserve_relative_dirs
        )
        if should_skip_existing_output(
            roughness_path=roughness_path,
            output_meta_path=output_meta_path,
            run_signature=run_signature,
        ):
            continue
        pending.append(rgb_path)
    return pending


def generate_roughness_map_with_seedream(*, ark_client: Any, model: str, input_mode: str, rgb_path: Path, save_path: Path, size: str, watermark: bool, timeout: int) -> str:
    prompt = build_roughness_prompt(input_mode)
    image_paths = [rgb_path]
    mode = 'seedream_rgb_only'
    response = run_image_generation(ark_client=ark_client, model=model, prompt=prompt, image_paths=image_paths, size=size, watermark=watermark)
    save_image_response(response.data[0], save_path, timeout=timeout)
    return mode


def main() -> None:
    args = parse_args()
    validate_parts(args.num_parts, args.part_index)
    api_key = ensure_api_key()
    if Ark is None:
        raise ImportError('Failed to import volcenginesdkarkruntime. Please install pip install "volcengine-python-sdk[ark]"') from _ARK_IMPORT_ERROR
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roughness_dir = output_dir / 'roughness'
    meta_dir = output_dir / 'meta'
    output_meta_dir = meta_dir / 'per_image'
    roughness_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(input_dir, args.filename_suffix, args.recursive)
    image_paths_for_generate_all = image_paths[:args.max_generate] if args.max_generate > 0 else image_paths
    image_paths_for_generate = shard_paths(image_paths_for_generate_all, args.num_parts, args.part_index)
    shard_label = f'part {args.part_index + 1}/{args.num_parts}'
    prompt_text = build_roughness_prompt(args.input_mode)
    run_signature = build_run_signature(input_mode=args.input_mode, image_model=args.image_model, prompt_text=prompt_text, timeout=args.timeout, size=args.size, watermark=args.watermark)
    pending_image_paths = get_pending_images(image_paths_for_generate, input_dir, roughness_dir, output_meta_dir, run_signature, args.overwrite, args.preserve_relative_dirs)
    print(f'[1/3] found {len(image_paths)} RGB images; current shard {shard_label} owns {len(image_paths_for_generate)} images, pending {len(pending_image_paths)} roughness maps with model={args.image_model}, input_mode={args.input_mode}')
    print(f'      filename_suffix={args.filename_suffix or "N/A"} | recursive={args.recursive} | preserve_relative_dirs={args.preserve_relative_dirs} | overwrite={args.overwrite} | base_url={args.base_url}')
    ark_client = Ark(base_url=args.base_url, api_key=api_key)
    setup = dict(run_signature)
    setup.update({'filename_suffix': args.filename_suffix, 'recursive': args.recursive, 'preserve_relative_dirs': args.preserve_relative_dirs, 'num_parts': args.num_parts, 'part_index': args.part_index})
    setup_path = build_part_path(meta_dir, 'setup', 'json', args.num_parts, args.part_index)
    setup_path.write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest_path = build_part_path(meta_dir, 'manifest', 'json', args.num_parts, args.part_index)
    manifest: List[Dict[str, Any]] = load_manifest(manifest_path)
    print(f'[2/3] start generation for {len(pending_image_paths)} pending images')
    if not pending_image_paths:
        print('  - current shard is already complete; exiting.')
    failures = 0
    for idx, rgb_path in enumerate(image_paths_for_generate, start=1):
        relative_image_path = rgb_path.relative_to(input_dir).as_posix()
        roughness_path = build_image_output_path(rgb_path, input_dir, roughness_dir, '_roughness.png', args.preserve_relative_dirs)
        output_meta_path = build_image_output_path(rgb_path, input_dir, output_meta_dir, '_roughness.json', args.preserve_relative_dirs)
        item: Dict[str, Any] = {'image_name': rgb_path.name, 'relative_image_path': relative_image_path, 'prompt_version': PROMPT_VERSION, 'input_mode': args.input_mode, 'num_parts': args.num_parts, 'part_index': args.part_index}
        try:
            if not args.overwrite and should_skip_existing_output(roughness_path=roughness_path, output_meta_path=output_meta_path, run_signature=run_signature):
                item['skipped'] = True
                item['status'] = 'skipped'
                item['skip_reason'] = 'existing_completed_output'
                item['roughness_output'] = relative_output(roughness_path, output_dir)
                print(f'  - ({idx}/{len(image_paths_for_generate)}) {relative_image_path} exists, skip')
                if not output_meta_path.exists():
                    ensure_parent_dir(output_meta_path)
                    output_meta = {'image_name': rgb_path.name, 'relative_image_path': relative_image_path, 'roughness_output': relative_output(roughness_path, output_dir), 'input_mode': args.input_mode, 'run_signature': run_signature, 'recovered_existing_output': True}
                    output_meta_path.write_text(json.dumps(output_meta, ensure_ascii=False, indent=2), encoding='utf-8')
                upsert_manifest_entry(manifest, item, key_field='relative_image_path')
                write_manifest(manifest_path, manifest)
                continue
            print(f'  - ({idx}/{len(image_paths_for_generate)}) {relative_image_path}')
            ensure_parent_dir(roughness_path)
            ensure_parent_dir(output_meta_path)
            item['roughness_mode'] = generate_roughness_map_with_seedream(ark_client=ark_client, model=args.image_model, input_mode=args.input_mode, rgb_path=rgb_path, save_path=roughness_path, size=args.size, watermark=args.watermark, timeout=args.timeout)
            item['status'] = 'done'
            item['roughness_output'] = relative_output(roughness_path, output_dir)
            output_meta = {'image_name': rgb_path.name, 'relative_image_path': relative_image_path, 'roughness_output': relative_output(roughness_path, output_dir), 'input_mode': args.input_mode, 'run_signature': run_signature, 'num_parts': args.num_parts, 'part_index': args.part_index}
            output_meta_path.write_text(json.dumps(output_meta, ensure_ascii=False, indent=2), encoding='utf-8')
            time.sleep(max(0.0, args.sleep))
        except Exception as e:
            failures += 1
            item['error'] = str(e)
            item['status'] = 'error'
            print(f'[error] {relative_image_path}: {e}')
        upsert_manifest_entry(manifest, item, key_field='relative_image_path')
        write_manifest(manifest_path, manifest)
    if failures:
        raise RuntimeError(f"{failures} image generation(s) failed; see the task manifest and log")
    print('[3/3] done')
    print(f'roughness output dir: {roughness_dir.as_posix()}')
    print(f'meta output dir: {meta_dir.as_posix()}')
    print(f'output_dir: {output_dir.as_posix()}')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
