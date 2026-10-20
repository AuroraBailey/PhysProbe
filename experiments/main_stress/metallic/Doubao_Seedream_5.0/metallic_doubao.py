#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Final Doubao metallic generation script fixed to the gate_black prompt variant.

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
build_metallic_prompt = _prompt_partial(_prompt_builder, model='doubao')
PROMPT_FAMILY = _PROMPT_CONFIGS['doubao']['PROMPT_FAMILY']
PROMPT_PRESET_CHOICES = _PROMPT_CONFIGS['doubao']['PROMPT_PRESET_CHOICES']


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
INPUT_MODE_CHOICES = ("rgb_only",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate metallic maps with SeedDream using the final gate_black prompt.')
    parser.add_argument('--input_dir', type=str, required=True, help='RGB image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--input_mode', type=str, default='rgb_only', choices=INPUT_MODE_CHOICES, help='Main/stress RGB-only generation.')
    parser.add_argument('--prompt_preset', type=str, default='v3_visualprior_noboundary', choices=PROMPT_PRESET_CHOICES, help='Built-in metallic prompt preset. This final script fixes the prompt to the gate_black wording.')
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
    images = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.name.startswith("Image_") and p.suffix.lower() == ".png"
    ]
    if images:
        return images
    raise FileNotFoundError(f"No PNG images were found in {input_dir}")


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


def build_run_signature(*, input_mode: str, prompt_version: str, prompt_preset: str, image_model: str, prompt_text: str, timeout: int, size: str, watermark: bool) -> Dict[str, Any]:
    route = 'seedream_rgb_only_metallic'
    return {'image_model': image_model, 'route': route, 'input_mode': input_mode, 'prompt_version': prompt_version, 'prompt_preset': prompt_preset, 'prompt_text': prompt_text, 'timeout': timeout, 'size': size, 'output_resolution_policy': 'model_native_output', 'watermark': watermark}


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


def generate_metallic_map_with_seedream(*, ark_client: Any, model: str, input_mode: str, prompt_preset: str, rgb_path: Path, save_path: Path, size: str, watermark: bool, timeout: int) -> str:
    prompt = build_metallic_prompt(input_mode=input_mode, prompt_preset=prompt_preset)
    image_paths = [rgb_path]
    mode = 'seedream_rgb_only_metallic'
    response = run_image_generation(ark_client=ark_client, model=model, prompt=prompt, image_paths=image_paths, size=size, watermark=watermark)
    save_image_response(response.data[0], save_path, timeout=timeout)
    return mode


def main() -> None:
    args = parse_args()
    api_key = ensure_api_key()
    prompt_version = f'{PROMPT_FAMILY}_{args.prompt_preset}'
    if Ark is None:
        raise ImportError('Failed to import volcenginesdkarkruntime. Please install pip install "volcengine-python-sdk[ark]"') from _ARK_IMPORT_ERROR
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
    print(f'[1/3] found {len(image_paths)} RGB images; generating {len(image_paths_for_generate)} metallic maps with model={args.image_model}, input_mode={args.input_mode}')
    ark_client = Ark(base_url=args.base_url, api_key=api_key)
    prompt_text = build_metallic_prompt(args.input_mode, prompt_preset=args.prompt_preset)
    run_signature = build_run_signature(input_mode=args.input_mode, prompt_version=prompt_version, prompt_preset=args.prompt_preset, image_model=args.image_model, prompt_text=prompt_text, timeout=args.timeout, size=args.size, watermark=args.watermark)
    setup = dict(run_signature)
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
            item['prompt_version'] = prompt_version
            item['input_mode'] = args.input_mode
            if args.skip_existing and should_skip_existing_output(metallic_path=metallic_path, output_meta_path=output_meta_path, run_signature=run_signature):
                item['skipped'] = True
                item['skip_reason'] = 'matching_output_and_signature'
                item['metallic_output'] = metallic_path.name
            else:
                item['metallic_mode'] = generate_metallic_map_with_seedream(ark_client=ark_client, model=args.image_model, input_mode=args.input_mode, prompt_preset=args.prompt_preset, rgb_path=rgb_path, save_path=metallic_path, size=args.size, watermark=args.watermark, timeout=args.timeout)
                item['metallic_output'] = metallic_path.name
                output_meta = {'image_name': rgb_path.name, 'metallic_output': metallic_path.name, 'input_mode': args.input_mode, 'run_signature': run_signature}
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
