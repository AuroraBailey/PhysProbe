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
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_IMAGE_MODEL = "qwen-image-2.0"
DEFAULT_GENERATION_MODE = "edit"
DEFAULT_IMAGE_SIZE = "1280*960"
DEFAULT_TIMEOUT = 1800
WAN_MIN_PIXELS = 768 * 768
WAN_MAX_PIXELS = 2048 * 2048
DEFAULT_WAN_OUTPUT_WIDTH = 1280
DEFAULT_WAN_OUTPUT_HEIGHT = 960
DEFAULT_WAN_FALLBACK_SIZE = f"{DEFAULT_WAN_OUTPUT_WIDTH}*{DEFAULT_WAN_OUTPUT_HEIGHT}"

DEPTH_POLARITY_CHOICES = ("near_white",)

INPUT_MODE_CHOICES = ("rgb_only",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate relative depth maps with Wan/DashScope from RGB inputs.')
    parser.add_argument('--input_dir', type=str, required=True, help='RGB image directory')
    parser.add_argument('--seg_dir', type=str, default='', help='Deprecated and ignored. Kept only for backward compatibility with older commands.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--input_mode', type=str, default='rgb_only', choices=INPUT_MODE_CHOICES, help='Strict RGB-only baseline or analysis/example-assisted baseline.')
    parser.add_argument('--depth_polarity', type=str, default=FIXED_DEPTH_POLARITY, choices=DEPTH_POLARITY_CHOICES, help='Fixed grayscale convention: near_white only. White means near, black means far.')
    parser.add_argument('--api_key', type=str, default=None, help='DashScope API key; defaults to DASHSCOPE_API_KEY')
    parser.add_argument('--base_url', type=str, default=os.environ.get('DASHSCOPE_BASE_URL', DEFAULT_BASE_URL), help='DashScope base URL')
    parser.add_argument('--image_model', type=str, default=DEFAULT_IMAGE_MODEL, help='DashScope image generation model')
    parser.add_argument('--generation_mode', type=str, default=DEFAULT_GENERATION_MODE, choices=['generate', 'edit'], help='generate: text-only generation; edit: attach input image(s), recommended for depth.')
    parser.add_argument('--size', type=str, default=DEFAULT_IMAGE_SIZE, help='Wan output size. source/original uses the target RGB size; explicit form: 1280*960.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for DashScope generation')
    parser.add_argument('--watermark', action='store_true', help='Keep watermark')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='Request and download timeout')
    parser.add_argument('--sleep', type=float, default=0.2, help='Sleep between requests')
    parser.add_argument('--max_generate', type=int, default=0, help='0 means process all images')
    parser.add_argument('--skip_existing', action='store_true', help='Skip images whose outputs already exist')
    return parser.parse_args()


def ensure_api_key(cli_api_key: Optional[str] = None) -> str:
    api_key = cli_api_key or os.environ.get("DASHSCOPE_API_KEY")
    if api_key:
        api_key = api_key.strip()
        try:
            api_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError(
                "DASHSCOPE_API_KEY contains non-ASCII characters and cannot be written to the HTTP header. "
                "Please replace any placeholder text with the real key."
            ) from e
        return api_key
    raise RuntimeError("Missing authentication: please provide --api_key or DASHSCOPE_API_KEY.")


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
    with Image.open(path) as img:
        return img.size


def validate_dashscope_output_size(width: int, height: int) -> None:
    pixels = width * height
    if pixels < WAN_MIN_PIXELS or pixels > WAN_MAX_PIXELS:
        raise ValueError(
            f"wan2.7-image output pixels must be in [{WAN_MIN_PIXELS}, {WAN_MAX_PIXELS}], "
            f"got {width}x{height} ({pixels})."
        )
    ratio = width / float(height)
    if ratio < 1.0 / 8.0 or ratio > 8.0:
        raise ValueError(f"wan2.7-image output aspect ratio must be in [1:8, 8:1], got {width}x{height}.")


def normalize_dashscope_output_size(size: str, source_size: Tuple[int, int]) -> str:
    normalized = str(size or "").strip().lower()
    if not normalized or normalized in {"source", "original", "input", "auto"}:
        width, height = source_size
    elif normalized in {"adaptive", "2k"}:
        normalized = DEFAULT_WAN_FALLBACK_SIZE
        width, height = DEFAULT_WAN_OUTPUT_WIDTH, DEFAULT_WAN_OUTPUT_HEIGHT
    else:
        normalized = normalized.replace("x", "*")
        parts = normalized.split("*")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f"wan2.7-image output size must look like 1280*960, got: {size}")
        width, height = int(parts[0]), int(parts[1])
    validate_dashscope_output_size(width, height)
    return f"{width}*{height}"


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


def extract_qwen_image_url(response_json: Dict[str, Any]) -> str:
    for choice in response_json.get("output", {}).get("choices", []):
        message = choice.get("message", {})
        for item in message.get("content", []):
            if item.get("type") == "image" and item.get("image"):
                return item["image"]
            if item.get("image"):
                return item["image"]
    raise RuntimeError(f"Model response contains no image URL: {json.dumps(response_json, ensure_ascii=False)}")


def run_qwen_image_generation(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    image_paths: List[Path],
    generation_mode: str,
    size: str,
    watermark: bool,
    seed: int,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[Dict[str, Any], str, List[Tuple[int, int]]]:
    if model != "qwen-image-2.0":
        raise ValueError("This runner requires qwen-image-2.0")
    mode = (generation_mode or DEFAULT_GENERATION_MODE).strip().lower()
    if mode not in {"generate", "edit"}:
        raise ValueError(f"Unsupported generation_mode: {generation_mode}")
    if not image_paths:
        raise ValueError("image_paths cannot be empty")

    input_sizes = [get_image_size(path) for path in image_paths]
    requested_size = normalize_dashscope_output_size(size, input_sizes[-1])
    content: List[Dict[str, Any]] = []
    if mode == "edit":
        content.extend({"image": file_to_data_uri(path)} for path in image_paths)
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
    return response_json, requested_size, input_sizes


def save_url_to_file(url: str, save_path: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with save_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def save_qwen_image_response(response_json: Dict[str, Any], save_path: Path, timeout: int = DEFAULT_TIMEOUT) -> str:
    image_url = extract_qwen_image_url(response_json)
    save_url_to_file(image_url, save_path, timeout=timeout)
    return image_url




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


def build_run_signature(*, input_mode: str, base_url: str, image_model: str, prompt_text: str, depth_polarity: str, timeout: int, generation_mode: str, size: str, seed: int, watermark: bool) -> Dict[str, Any]:
    route = 'wan_rgb_relative_depth_direct'
    return {'image_model': image_model, 'base_url': base_url, 'route': route, 'input_mode': input_mode, 'prompt_version': PROMPT_VERSION, 'prompt_text': prompt_text, 'depth_representation': 'relative_depth', 'depth_polarity': depth_polarity, 'evaluation_protocol': 'single_image_relative_depth_prediction', 'evaluation_note': 'Near-white only. No polarity inversion, scale fit, or shift alignment is applied during generation.', 'timeout': timeout, 'generation_mode': generation_mode, 'size': size, 'seed': seed, 'padding_mode': 'none', 'output_resolution_policy': 'provider_native_bytes', 'watermark': watermark}


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


def generate_relative_depth_map_with_qwen2(*, api_key: str, base_url: str, model: str, generation_mode: str, input_mode: str, rgb_path: Path, save_path: Path, size: str, watermark: bool, seed: int, timeout: int, depth_polarity: str) -> Dict[str, Any]:
    prompt = build_relative_depth_prompt(depth_polarity)
    image_paths = [rgb_path]
    mode = 'wan_rgb_relative_depth_direct'
    source_size = get_image_size(rgb_path)
    response_json, requested_size, input_sizes = run_qwen_image_generation(api_key=api_key, base_url=base_url, model=model, prompt=prompt, image_paths=image_paths, generation_mode=generation_mode, size=size, watermark=watermark, seed=seed, timeout=timeout)
    image_url = save_qwen_image_response(response_json, save_path, timeout=timeout)
    usage = response_json.get('usage', {}) if isinstance(response_json, dict) else {}
    return {'mode': mode, 'image_url': image_url, 'requested_size': requested_size, 'input_sizes': [list(size_pair) for size_pair in input_sizes], 'source_size': list(source_size), 'final_size': 'model_native', 'returned_size': usage.get('size'), 'request_id': response_json.get('request_id') if isinstance(response_json, dict) else None}


def main() -> None:
    args = parse_args()
    if args.depth_polarity != FIXED_DEPTH_POLARITY:
        raise ValueError(f'This script only supports {FIXED_DEPTH_POLARITY}, got: {args.depth_polarity}')
    api_key = ensure_api_key(args.api_key)
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
    run_signature = build_run_signature(input_mode=args.input_mode, base_url=args.base_url, image_model=args.image_model, prompt_text=prompt_text, depth_polarity=args.depth_polarity, timeout=args.timeout, generation_mode=args.generation_mode, size=args.size, seed=args.seed, watermark=args.watermark)
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
                print(f'      skip existing: {depth_path.as_posix()}')
            else:
                generation_result = generate_relative_depth_map_with_qwen2(api_key=api_key, base_url=args.base_url, model=args.image_model, generation_mode=args.generation_mode, input_mode=args.input_mode, rgb_path=rgb_path, save_path=depth_path, size=args.size, watermark=args.watermark, seed=args.seed, timeout=args.timeout, depth_polarity=args.depth_polarity)
                item['depth_mode'] = generation_result['mode']
                item['image_url'] = generation_result['image_url']
                item['request_id'] = generation_result['request_id']
                item['relative_depth_output'] = depth_path.name
                output_meta = {'image_name': rgb_path.name, 'relative_depth_output': depth_path.name, 'input_mode': args.input_mode, 'run_signature': run_signature, 'generation_result': generation_result}
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
