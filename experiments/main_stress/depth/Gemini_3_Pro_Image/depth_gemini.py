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
import contextlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from common.utils.gemini_client import GEMINI_IMAGE_MODEL, GeminiImageClient, gemini_request_record


DEFAULT_IMAGE_MODEL = GEMINI_IMAGE_MODEL
DEFAULT_TIMEOUT = 1800

DEPTH_POLARITY_CHOICES = ("near_white",)

INPUT_MODE_CHOICES = ("rgb_only",)


DEFAULT_IMAGE_SIZE = "1536x1024"

DEFAULT_EDIT_CANVAS_SIZE = (1536, 1024)

SUPPORTED_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate relative depth maps with Gemini from RGB inputs.')
    parser.add_argument('--input_dir', type=str, default='', help='Deprecated compatibility argument; --scene_list controls RGB inputs.')
    parser.add_argument('--scene_list', type=str, required=True, help='Scene list with paths relative to --dataset_root.')
    parser.add_argument('--dataset_root', type=str, default=str(Path(__file__).resolve().parents[4] / 'data' / 'raw'), help='Dataset root used to resolve --scene_list entries.')
    parser.add_argument('--seg_dir', type=str, default='', help='Deprecated and ignored. Kept only for backward compatibility with older commands.')
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).resolve().parent / 'output' / 'depth'), help='Output directory (default: gemini/output/depth).')
    parser.add_argument('--input_mode', type=str, default='rgb_only', choices=INPUT_MODE_CHOICES, help='Strict RGB-only baseline or analysis/example-assisted baseline.')
    parser.add_argument('--depth_polarity', type=str, default=FIXED_DEPTH_POLARITY, choices=DEPTH_POLARITY_CHOICES, help='Fixed grayscale convention: near_white only. White means near, black means far.')
    parser.add_argument('--base_url', type=str, default=os.environ.get('AIHUBMIX_BASE_URL'), help='Aihubmix endpoint; defaults to AIHUBMIX_BASE_URL.')
    parser.add_argument('--api_key', type=str, default=None, help='Aihubmix API key')
    parser.add_argument('--image_model', type=str, default=DEFAULT_IMAGE_MODEL, help='Image generation model')
    parser.add_argument('--size', type=str, default=DEFAULT_IMAGE_SIZE, help='Gemini output size')
    parser.add_argument('--watermark', action='store_true', help='Keep watermark')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='Request and download timeout')
    parser.add_argument('--sleep', type=float, default=0.2, help='Sleep between requests')
    parser.add_argument('--max_generate', type=int, default=0, help='0 means process all images')
    parser.add_argument('--skip_existing', action='store_true', help='Skip images whose outputs already exist')
    return parser.parse_args()


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


def resolve_image_client_config(args: argparse.Namespace) -> Any:
    api_key = ensure_api_key(args.api_key)
    return GeminiImageClient(api_key, base_url=args.base_url, aspect_ratio="4:3", timeout=args.timeout)


def get_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size






def prepare_upload_image(src_path, dst_path, target_size=None):
    """Prepare the upload without padding; legacy name retained for callers."""
    return prepare_input(src_path, dst_path, gpt=False)




def postprocess_generated_image_bytes(image_bytes, *args, **kwargs):
    """Keep the complete provider response byte-for-byte."""
    return image_bytes


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


def list_images_from_scene_list(scene_list_path: Path, dataset_root: Path) -> List[Path]:
    """List RGB inputs recursively from scene paths relative to dataset_root."""
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
        scene_images = sorted(path for path in scene_path.rglob("*_im.png") if path.is_file())
        if not scene_images:
            scene_images = sorted(path for path in scene_path.rglob("Image_*.png") if path.is_file())
        if not scene_images:
            raise FileNotFoundError(f"No RGB images were found in scene: {scene_path}")
        images.extend(scene_images)
    return sorted(images)


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
    image_client: Any,
    model: str,
    prompt: str,
    image_paths: List[Path],
    size: str,
    watermark: bool,
) -> Any:
    del watermark
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    requested_size = normalize_image_size(size)

    with contextlib.ExitStack() as stack:
        image_files = [stack.enter_context(path.open("rb")) for path in image_paths]
        image_arg: Any = image_files[0] if len(image_files) == 1 else image_files
        return image_client.images.edit(
            model=model,
            image=image_arg,
            prompt=prompt,
            n=1,
            size=requested_size,
                    )


def fetch_url_bytes(url: str, timeout: int = DEFAULT_TIMEOUT) -> bytes:
    output = io.BytesIO()
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
    return output.getvalue()


def get_image_response_bytes(image_item: Any, timeout: int = DEFAULT_TIMEOUT) -> bytes:
    url = getattr(image_item, "url", None)
    if url:
        return fetch_url_bytes(url, timeout=timeout)

    b64_json = getattr(image_item, "b64_json", None)
    if b64_json:
        return base64.b64decode(b64_json)

    if isinstance(image_item, dict):
        if image_item.get("url"):
            return fetch_url_bytes(image_item["url"], timeout=timeout)
        if image_item.get("b64_json"):
            return base64.b64decode(image_item["b64_json"])

    raise RuntimeError("Image response contains neither url nor b64_json.")


def save_image_response(image_item, save_path, timeout=120):
    """Save the provider response without decoding or altering its image pixels."""
    save_path.write_bytes(get_image_response_bytes(image_item, timeout=timeout))




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
    item = gemini_request_record(item)
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
    route = 'gemini_rgb_relative_depth_direct'
    return gemini_request_record({'image_model': image_model, 'route': route, 'input_mode': input_mode, 'prompt_version': PROMPT_VERSION, 'prompt_text': prompt_text, 'depth_representation': 'relative_depth', 'depth_polarity': depth_polarity, 'evaluation_protocol': 'single_image_relative_depth_prediction', 'evaluation_note': 'Near-white only. No polarity inversion, scale fit, or shift alignment is applied during generation.', 'timeout': timeout, 'size': size, 'preprocess_canvas_size': 'model_native', 'padding_mode': 'none', 'input_resize_policy': 'model_input_native', 'output_resolution_policy': 'provider_native_bytes', 'saved_depth_size': 'model_native', 'watermark': watermark})


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


def generate_relative_depth_map_with_gemini(*, image_client: Any, model: str, input_mode: str, rgb_path: Path, save_path: Path, size: str, watermark: bool, timeout: int, depth_polarity: str) -> str:
    prompt = build_relative_depth_prompt(depth_polarity)
    mode = 'gemini_rgb_relative_depth_direct'
    with tempfile.TemporaryDirectory(prefix='gemini_edit_pad_') as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        padded_rgb = temp_dir / 'query.png'
        query_bbox = prepare_upload_image(rgb_path, padded_rgb)
        image_paths = [padded_rgb]
        response = run_image_generation(image_client=image_client, model=model, prompt=prompt, image_paths=image_paths, size=size, watermark=watermark)
    save_image_response(response.data[0], save_path, timeout=timeout)
    return mode


def main() -> None:
    args = parse_args()
    if args.depth_polarity != FIXED_DEPTH_POLARITY:
        raise ValueError(f'This script only supports {FIXED_DEPTH_POLARITY}, got: {args.depth_polarity}')
    image_client = resolve_image_client_config(args)
    input_dir = Path(args.dataset_root).expanduser()
    if not input_dir.is_dir():
        raise FileNotFoundError(f'Dataset root was not found: {input_dir}')
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
    depth_dir = output_dir
    meta_dir = output_dir / 'meta'
    output_meta_dir = meta_dir / 'per_image'
    depth_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)
    if args.scene_list:
        scene_list_path = Path(args.scene_list).expanduser()
        image_paths = list_images_from_scene_list(scene_list_path, input_dir)
        print(f'[info] scene_list={scene_list_path.as_posix()}')
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
        image_relative_path = rgb_path.relative_to(input_dir)
        item: Dict[str, Any] = {'image_name': image_relative_path.as_posix()}
        try:
            print(f'  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name}')
            depth_path = depth_dir / image_relative_path.with_name(f'{rgb_path.stem}_relative_depth.png')
            output_meta_path = output_meta_dir / image_relative_path.with_name(f'{rgb_path.stem}_relative_depth.json')
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            output_meta_path.parent.mkdir(parents=True, exist_ok=True)
            item['prompt_version'] = PROMPT_VERSION
            item['input_mode'] = args.input_mode
            item['depth_representation'] = 'relative_depth'
            item['depth_polarity'] = args.depth_polarity
            item['input_resize_policy'] = 'model_input_native'
            item['output_resolution_policy'] = 'provider_native_bytes'
            item['saved_depth_size'] = 'model_native'
            if args.skip_existing and should_skip_existing_output(depth_path=depth_path, output_meta_path=output_meta_path, run_signature=run_signature):
                item['skipped'] = True
                item['skip_reason'] = 'matching_output_and_signature'
                item['relative_depth_output'] = depth_path.relative_to(depth_dir).as_posix()
            else:
                item['depth_mode'] = generate_relative_depth_map_with_gemini(image_client=image_client, model=args.image_model, input_mode=args.input_mode, rgb_path=rgb_path, save_path=depth_path, size=args.size, watermark=args.watermark, timeout=args.timeout, depth_polarity=args.depth_polarity)
                item['relative_depth_output'] = depth_path.relative_to(depth_dir).as_posix()
                output_meta = gemini_request_record({'image_name': image_relative_path.as_posix(), 'relative_depth_output': depth_path.relative_to(depth_dir).as_posix(), 'input_mode': args.input_mode, 'run_signature': run_signature, 'input_resize_policy': 'model_input_native', 'output_resolution_policy': 'provider_native_bytes', 'saved_depth_size': 'model_native'})
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





def normalize_image_size(size: Optional[str]) -> str:
    normalized = (size or DEFAULT_IMAGE_SIZE).strip().lower()
    if normalized in {"adaptive", "source", "2k"}:
        normalized = DEFAULT_IMAGE_SIZE
    if normalized not in SUPPORTED_IMAGE_SIZES:
        raise ValueError(
            f"Unsupported --size for Gemini: {size}. "
            f"Choices: {', '.join(sorted(SUPPORTED_IMAGE_SIZES))}"
        )
    return normalized



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
