#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Metallic generation with WAN2.7-Image from RGB-only, built-in final
# RGB+prompt, or RGB+example inputs. This script freezes the final
# surface_material_windowpane_filltight prompt variant.

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
from experiments.main_stress.metallic.prompts import MODEL_CONFIGS as _PROMPT_CONFIGS, build_metallic_prompt as _prompt_builder
build_metallic_prompt = _prompt_partial(_prompt_builder, model='wan')
DEFAULT_IMAGE_MODEL = _PROMPT_CONFIGS['wan']['DEFAULT_IMAGE_MODEL']
PROMPT_PRESET = _PROMPT_CONFIGS['wan']['PROMPT_PRESET']
PROMPT_VERSION = _PROMPT_CONFIGS['wan']['PROMPT_VERSION']


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
from PIL import Image


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_GENERATION_MODE = "edit"
DEFAULT_TIMEOUT = 1800
DEFAULT_WAN_OUTPUT_WIDTH = 1280
DEFAULT_WAN_OUTPUT_HEIGHT = 720
DEFAULT_WAN_OUTPUT_SIZE = "same_as_input"
WAN_MIN_PIXELS = 768 * 768
WAN_MAX_PIXELS = 2048 * 2048
INPUT_MODE_CHOICES = ("rgb_only",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate metallic maps with WAN2.7-Image from RGB-only, RGB+prompt, or RGB+example inputs.')
    parser.add_argument('--input_dir', type=str, required=True, help='RGB image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--input_mode', type=str, default='rgb_only', choices=INPUT_MODE_CHOICES, help='Main/stress RGB-only generation.')
    parser.add_argument('--api_key', type=str, default=None, help='DashScope API key')
    parser.add_argument('--base_url', type=str, default=os.environ.get('DASHSCOPE_BASE_URL', DEFAULT_BASE_URL), help='DashScope base URL')
    parser.add_argument('--image_model', type=str, default=DEFAULT_IMAGE_MODEL, help='Image generation model')
    parser.add_argument('--generation_mode', type=str, default=DEFAULT_GENERATION_MODE, choices=['generate', 'edit'], help='Generation mode for wan2.7-image. edit is recommended for image-conditioned metallic generation.')
    parser.add_argument('--size', type=str, default=DEFAULT_WAN_OUTPUT_SIZE, help='Output size in WIDTH*HEIGHT format, or same_as_input / adaptive.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for generation')
    parser.add_argument('--watermark', action='store_true', help='Keep watermark')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='Request and download timeout')
    parser.add_argument('--sleep', type=float, default=0.2, help='Sleep between requests')
    parser.add_argument('--max_generate', type=int, default=0, help='0 means process all images')
    parser.add_argument('--skip_existing', action='store_true', help='Skip images whose outputs already exist')
    return parser.parse_args()


def ensure_api_key(cli_api_key: Optional[str] = None) -> str:
    api_key = cli_api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing authentication: please provide --api_key or set DASHSCOPE_API_KEY.")
    return api_key


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
    images = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.name.startswith("Image_") and p.suffix.lower() == ".png"]
    if images:
        return images
    raise FileNotFoundError(f"No PNG images were found in {input_dir}")


def parse_wan_size(size_text: str) -> Tuple[int, int]:
    text = str(size_text or "").strip().lower()
    if not text or text in {"adaptive", "same_as_input"}:
        return DEFAULT_WAN_OUTPUT_WIDTH, DEFAULT_WAN_OUTPUT_HEIGHT
    match = re.fullmatch(r"(\d+)\s*[*xX]\s*(\d+)", text)
    if not match:
        raise ValueError(f"Invalid size format: {size_text}. Expected WIDTH*HEIGHT, for example 1280*960.")
    return int(match.group(1)), int(match.group(2))


def validate_wan_output_size(width: int, height: int) -> None:
    pixels = width * height
    if pixels < WAN_MIN_PIXELS or pixels > WAN_MAX_PIXELS:
        raise ValueError(
            f"wan2.7-image output pixels must be within [{WAN_MIN_PIXELS}, {WAN_MAX_PIXELS}], got {width}x{height} ({pixels})."
        )
    ratio = width / float(height)
    if ratio < 1.0 / 8.0 or ratio > 8.0:
        raise ValueError(f"wan2.7-image output aspect ratio must be within [1:8, 8:1], got {width}x{height}.")


def build_wan_size(size_text: str, input_size: Optional[Tuple[int, int]] = None) -> str:
    text = str(size_text or "").strip().lower()
    if text == "same_as_input":
        if input_size is None:
            raise ValueError("same_as_input requires input_size.")
        width, height = input_size
    else:
        width, height = parse_wan_size(size_text)
    validate_wan_output_size(width, height)
    return f"{width}*{height}"


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


def extract_wan_image_url(response_json: Dict[str, Any]) -> str:
    for choice in response_json.get("output", {}).get("choices", []):
        message = choice.get("message", {})
        for item in message.get("content", []):
            if item.get("type") == "image" and item.get("image"):
                return item["image"]
            if item.get("image"):
                return item["image"]
    raise RuntimeError(f"Could not find image URL in model response: {json.dumps(response_json, ensure_ascii=False)}")


def save_url_to_file(url: str, save_path: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def save_wan_image_response(response_json: Dict[str, Any], save_path: Path, timeout: int = DEFAULT_TIMEOUT) -> str:
    image_url = extract_wan_image_url(response_json)
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


def build_run_signature(*, input_mode: str, image_model: str, prompt_text: str, timeout: int, size: str, generation_mode: str, watermark: bool, seed: int) -> Dict[str, Any]:
    route = 'wan_rgb_only_metallic'
    return {'image_model': image_model, 'route': route, 'input_mode': input_mode, 'prompt_version': PROMPT_VERSION, 'prompt_preset': PROMPT_PRESET, 'prompt_text': prompt_text, 'timeout': timeout, 'size': size, 'generation_mode': generation_mode, 'output_resolution_policy': 'requested_fixed_output_size', 'watermark': watermark, 'seed': seed}


def should_skip_existing_output(
    *,
    metallic_path: Path,
    output_meta_path: Path,
    run_signature: Dict[str, Any],
) -> bool:
    if not metallic_path.exists() or not output_meta_path.exists():
        return False
    saved = load_json_dict(output_meta_path)
    return saved.get("run_signature") == run_signature


def get_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def run_wan_image_generation(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    image_paths: List[Path],
    generation_mode: str,
    size: str,
    watermark: bool,
    seed: int,
    timeout: int,
) -> Dict[str, Any]:
    if model != "wan2.7-image":
        raise ValueError("This runner requires wan2.7-image")
    mode = (generation_mode or DEFAULT_GENERATION_MODE).strip().lower()
    if mode not in {"generate", "edit"}:
        raise ValueError(f"Unsupported generation_mode: {generation_mode}")

    input_size = get_image_size(image_paths[-1]) if image_paths else None
    requested_size = build_wan_size(size, input_size=input_size)
    content: List[Dict[str, str]] = []
    if mode == "edit":
        for image_path in image_paths:
            content.append({"image": file_to_data_uri(image_path)})
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
    return response_json


def generate_metallic_map_with_wan(*, api_key: str, base_url: str, model: str, input_mode: str, rgb_path: Path, save_path: Path, size: str, generation_mode: str, watermark: bool, seed: int, timeout: int) -> Dict[str, Any]:
    prompt = build_metallic_prompt(input_mode=input_mode, image_model=model)
    input_size = get_image_size(rgb_path)
    image_paths = [rgb_path]
    mode = 'wan_rgb_only_metallic'
    response_json = run_wan_image_generation(api_key=api_key, base_url=base_url, model=model, prompt=prompt, image_paths=image_paths, generation_mode=generation_mode, size=size, watermark=watermark, seed=seed, timeout=timeout)
    image_url = save_wan_image_response(response_json, save_path, timeout=timeout)
    return {'metallic_mode': mode, 'prompt_text': prompt, 'request_id': response_json.get('request_id'), 'image_url': image_url, 'requested_size': build_wan_size(size, input_size=input_size), 'input_size': list(input_size), 'returned_size': (response_json.get('usage') or {}).get('size')}


def main() -> None:
    args = parse_args()
    api_key = ensure_api_key(args.api_key)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metallic_dir = output_dir / 'metallic'
    meta_dir = output_dir / 'meta'
    output_meta_dir = meta_dir / 'per_image'
    metallic_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(input_dir)
    image_paths_for_generate = image_paths[:args.max_generate] if args.max_generate > 0 else image_paths
    print(f'[1/3] found {len(image_paths)} RGB images; generating {len(image_paths_for_generate)} metallic maps with model={args.image_model}, input_mode={args.input_mode}, generation_mode={args.generation_mode}')
    prompt_text = build_metallic_prompt(args.input_mode, image_model=args.image_model)
    signature_input_size = get_image_size(image_paths_for_generate[0]) if image_paths_for_generate else (DEFAULT_WAN_OUTPUT_WIDTH, DEFAULT_WAN_OUTPUT_HEIGHT)
    run_signature = build_run_signature(input_mode=args.input_mode, image_model=args.image_model, prompt_text=prompt_text, timeout=args.timeout, size=build_wan_size(args.size, input_size=signature_input_size), generation_mode=args.generation_mode, watermark=args.watermark, seed=args.seed)
    setup = dict(run_signature)
    setup['base_url'] = args.base_url
    (meta_dir / 'setup.json').write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest_path = meta_dir / 'manifest.json'
    manifest: List[Dict[str, Any]] = load_manifest(manifest_path)
    print(f'[2/3] start generation for {len(image_paths_for_generate)} images')
    failures = 0
    for idx, rgb_path in enumerate(image_paths_for_generate, start=1):
        item: Dict[str, Any] = {'image_name': rgb_path.name}
        try:
            print(f'  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name}')
            metallic_path = metallic_dir / f'{rgb_path.stem}_metallic.png'
            output_meta_path = output_meta_dir / f'{rgb_path.stem}_metallic.json'
            item['prompt_version'] = PROMPT_VERSION
            item['input_mode'] = args.input_mode
            if args.skip_existing and should_skip_existing_output(metallic_path=metallic_path, output_meta_path=output_meta_path, run_signature=run_signature):
                item['skipped'] = True
                item['skip_reason'] = 'matching_output_and_signature'
                item['metallic_output'] = metallic_path.name
            else:
                result = generate_metallic_map_with_wan(api_key=api_key, base_url=args.base_url, model=args.image_model, input_mode=args.input_mode, rgb_path=rgb_path, save_path=metallic_path, size=args.size, generation_mode=args.generation_mode, watermark=args.watermark, seed=args.seed, timeout=args.timeout)
                item.update(result)
                item['metallic_output'] = metallic_path.name
                output_meta = {'image_name': rgb_path.name, 'metallic_output': metallic_path.name, 'input_mode': args.input_mode, 'run_signature': run_signature, 'request_id': result.get('request_id'), 'image_url': result.get('image_url'), 'requested_size': result.get('requested_size'), 'input_size': result.get('input_size'), 'returned_size': result.get('returned_size')}
                output_meta_path.write_text(json.dumps(output_meta, ensure_ascii=False, indent=2), encoding='utf-8')
                time.sleep(max(0.0, args.sleep))
        except Exception as e:
            failures += 1
            item['error'] = str(e)
            item['status'] = 'error'
            print(f'[error] {rgb_path.name}: {e}')
        upsert_manifest_entry(manifest, item)
        write_manifest(manifest_path, manifest)
    if failures:
        raise RuntimeError(f"{failures} image generation(s) failed; see the task manifest and log")
    print('[3/3] done')
    print(f'output_dir: {output_dir.as_posix()}')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
