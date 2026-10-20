#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Main pipeline: direct relative depth generation from RGB only.
# Modified version:
# - near_white only (white=near, black=far)
# - no inversion in postprocess
# - unified prompt semantics across modes

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

from experiments.main_stress.depth.prompts import (
    FIXED_DEPTH_POLARITY, PROMPT_VERSION, build_relative_depth_prompt, get_depth_semantics,
)


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

DEPTH_POLARITY_CHOICES = ("near_white",)

INPUT_MODE_CHOICES = ("rgb_only",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate relative depth maps with SeedDream from RGB inputs.')
    parser.add_argument('--input_dir', type=str, required=True, help='RGB image directory')
    parser.add_argument('--seg_dir', type=str, default='', help='Deprecated and ignored. Kept only for backward compatibility with older commands.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--input_mode', type=str, default='rgb_only', choices=INPUT_MODE_CHOICES, help='Strict RGB-only baseline or analysis/example-assisted baseline.')
    parser.add_argument('--depth_polarity', type=str, default=FIXED_DEPTH_POLARITY, choices=DEPTH_POLARITY_CHOICES, help='Fixed grayscale convention: near_white only. White means near, black means far.')
    parser.add_argument('--base_url', type=str, default=DEFAULT_BASE_URL, help='Ark base URL')
    parser.add_argument('--image_model', type=str, default=DEFAULT_IMAGE_MODEL, help='Image generation model')
    parser.add_argument('--size', type=str, default='adaptive', help='Output size; adaptive -> 2k')
    parser.add_argument('--watermark', action='store_true', help='Keep watermark')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='Request and download timeout')
    parser.add_argument('--sleep', type=float, default=0.2, help='Sleep between requests')
    parser.add_argument('--max_generate', type=int, default=0, help='0 means process all images')
    parser.add_argument('--skip_existing', action='store_true', help='Skip images whose outputs already exist')
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


def list_images(input_dir: Path) -> List[Path]:
    preferred = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.name.lower().endswith("_im.png")
    ]
    if preferred:
        return preferred

    image_prefix = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() == ".png" and p.name.lower().startswith("image_")
    ]
    if image_prefix:
        return image_prefix

    raise FileNotFoundError(f"No RGB images matching *_im.png or Image_*.png were found in {input_dir}")


def looks_like_object_seg_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() == ".png" and p.name.lower().startswith("objectsegmentation_"):
            return True
    return False


def infer_rgb_dir_from_seg_dir(seg_dir: Path) -> Optional[Path]:
    parts = list(seg_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "objectsegmentation":
            parts[idx] = "Image"
            candidate = Path(*parts)
            if candidate.exists() and candidate.is_dir():
                return candidate
            break
    return None






def run_image_generation(
    ark_client: Any,
    model: str,
    prompt: str,
    image_paths: List[Path],
    size: str,
    watermark: bool,
) -> Any:
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    image_uris = [file_to_data_uri(p) for p in image_paths]

    if size:
        size = size.strip().lower()
    if not size or size == "adaptive":
        size = "2k"

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


def save_image_response(
    image_item: Any,
    save_path: Path,
    timeout: int = DEFAULT_TIMEOUT,
) -> None:
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
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def upsert_manifest_entry(manifest: List[Dict[str, Any]], item: Dict[str, Any]) -> None:
    image_name = str(item.get("image_name", "")).strip()
    if not image_name:
        manifest.append(item)
        return
    for idx, existing in enumerate(manifest):
        if str(existing.get("image_name", "")).strip() == image_name:
            manifest[idx] = item
            return
    manifest.append(item)


def build_run_signature(*, input_mode: str, image_model: str, prompt_text: str, depth_polarity: str, timeout: int, size: str, watermark: bool) -> Dict[str, Any]:
    route = 'seedream_rgb_relative_depth_direct'
    return {'image_model': image_model, 'route': route, 'input_mode': input_mode, 'prompt_version': PROMPT_VERSION, 'prompt_text': prompt_text, 'depth_representation': 'relative_depth', 'depth_polarity': depth_polarity, 'evaluation_protocol': 'single_image_relative_depth_prediction', 'evaluation_note': 'Near-white only. No polarity inversion, scale fit, or shift alignment is applied during generation.', 'timeout': timeout, 'size': size, 'input_resolution_policy': 'source_image_without_padding', 'output_resolution_policy': 'model_native_output_without_crop_or_resize', 'watermark': watermark}


def should_skip_existing_output(
    *,
    depth_path: Path,
    output_meta_path: Path,
    run_signature: Dict[str, Any],
) -> bool:
    if not depth_path.exists() or not output_meta_path.exists():
        return False
    saved = load_json_dict(output_meta_path)
    return saved.get("run_signature") == run_signature


def generate_relative_depth_map_with_seedream(*, ark_client: Any, model: str, input_mode: str, rgb_path: Path, save_path: Path, size: str, watermark: bool, timeout: int, depth_polarity: str) -> str:
    prompt = build_relative_depth_prompt(depth_polarity)
    image_paths = [rgb_path]
    mode = 'seedream_rgb_relative_depth_direct'
    response = run_image_generation(ark_client=ark_client, model=model, prompt=prompt, image_paths=image_paths, size=size, watermark=watermark)
    save_image_response(response.data[0], save_path, timeout=timeout)
    return mode


def main() -> None:
    args = parse_args()
    if args.depth_polarity != FIXED_DEPTH_POLARITY:
        raise ValueError(f'This script only supports {FIXED_DEPTH_POLARITY}, got: {args.depth_polarity}')
    api_key = ensure_api_key()
    if Ark is None:
        raise ImportError('Failed to import volcenginesdkarkruntime. Please install pip install "volcengine-python-sdk[ark]"') from _ARK_IMPORT_ERROR
    ark_client = Ark(base_url=args.base_url, api_key=api_key)
    input_dir = Path(args.input_dir)
    if looks_like_object_seg_dir(input_dir):
        inferred_rgb_dir = infer_rgb_dir_from_seg_dir(input_dir)
        if inferred_rgb_dir is None:
            raise RuntimeError(f'input_dir looks like an ObjectSegmentation directory but the paired RGB directory could not be inferred: {input_dir}')
        print(f'[info] detected segmentation directory as input_dir, switching RGB directory to: {inferred_rgb_dir}')
        input_dir = inferred_rgb_dir
    if args.seg_dir:
        print('[info] --seg_dir is deprecated for depth_generation.py and will be ignored.')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / 'relative_depth'
    meta_dir = output_dir / 'meta'
    output_meta_dir = meta_dir / 'per_image'
    depth_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(input_dir)
    image_paths_for_generate = image_paths[:args.max_generate] if args.max_generate > 0 else image_paths
    print(f'[1/3] found {len(image_paths)} RGB images; generating {len(image_paths_for_generate)} relative depth maps with model={args.image_model}, input_mode={args.input_mode}, depth_polarity={args.depth_polarity}')
    prompt_text = build_relative_depth_prompt(args.depth_polarity)
    run_signature = build_run_signature(input_mode=args.input_mode, image_model=args.image_model, prompt_text=prompt_text, depth_polarity=args.depth_polarity, timeout=args.timeout, size=args.size, watermark=args.watermark)
    setup = dict(run_signature)
    (meta_dir / 'setup.json').write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest_path = meta_dir / 'manifest.json'
    manifest: List[Dict[str, Any]] = load_manifest(manifest_path)
    print(f'[2/3] start generation for {len(image_paths_for_generate)} images')
    for idx, rgb_path in enumerate(image_paths_for_generate, start=1):
        item: Dict[str, Any] = {'image_name': rgb_path.name}
        try:
            print(f'  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name}')
            depth_path = depth_dir / f'{rgb_path.stem}_relative_depth.png'
            output_meta_path = output_meta_dir / f'{rgb_path.stem}_relative_depth.json'
            item['prompt_version'] = PROMPT_VERSION
            item['input_mode'] = args.input_mode
            item['depth_representation'] = 'relative_depth'
            item['depth_polarity'] = args.depth_polarity
            if args.skip_existing and should_skip_existing_output(depth_path=depth_path, output_meta_path=output_meta_path, run_signature=run_signature):
                item['skipped'] = True
                item['skip_reason'] = 'matching_output_and_signature'
                item['relative_depth_output'] = depth_path.name
            else:
                item['depth_mode'] = generate_relative_depth_map_with_seedream(ark_client=ark_client, model=args.image_model, input_mode=args.input_mode, rgb_path=rgb_path, save_path=depth_path, size=args.size, watermark=args.watermark, timeout=args.timeout, depth_polarity=args.depth_polarity)
                item['relative_depth_output'] = depth_path.name
                output_meta = {'image_name': rgb_path.name, 'relative_depth_output': depth_path.name, 'input_mode': args.input_mode, 'run_signature': run_signature}
                output_meta_path.write_text(json.dumps(output_meta, ensure_ascii=False, indent=2), encoding='utf-8')
                time.sleep(max(0.0, args.sleep))
        except Exception as e:
            item['error'] = str(e)
            item['status'] = 'error'
            print(f'[error] {rgb_path.name}: {e}')
        upsert_manifest_entry(manifest, item)
        write_manifest(manifest_path, manifest)
    print('[3/3] done')
    print(f'output_dir: {output_dir.as_posix()}')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
