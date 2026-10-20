#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Metallic generation script using GPT Image 2 with letterbox preprocessing.

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
build_metallic_prompt = _prompt_partial(_prompt_builder, model='gpt_image_2')
PROMPT_FAMILY = _PROMPT_CONFIGS['gpt_image_2']['PROMPT_FAMILY']
PROMPT_PRESET_CHOICES = _PROMPT_CONFIGS['gpt_image_2']['PROMPT_PRESET_CHOICES']


import argparse
import base64
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

try:
    from openai import AzureOpenAI
except Exception as e:
    AzureOpenAI = None
    _OPENAI_IMPORT_ERROR = e
else:
    _OPENAI_IMPORT_ERROR = None


DEFAULT_AZURE_ENDPOINT = None
DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_BASE_URL = DEFAULT_AZURE_ENDPOINT
DEFAULT_IMAGE_MODEL = "gpt-image-2"
DEFAULT_IMAGE_SIZE = "1536x1024"
DEFAULT_IMAGE_QUALITY = "medium"
DEFAULT_TIMEOUT = 1800
MODEL_INPUT_SIZE = (1536, 1024)
INPUT_MODE_CHOICES = ("rgb_only",)
SUPPORTED_GPT_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate metallic maps with GPT Image 2 using the final gate_black prompt.')
    parser.add_argument('--input_dir', type=str, required=True, help='RGB image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--input_mode', type=str, default='rgb_only', choices=INPUT_MODE_CHOICES, help='Main/stress RGB-only generation.')
    parser.add_argument('--prompt_preset', type=str, default='v3_visualprior_noboundary', choices=PROMPT_PRESET_CHOICES, help='Built-in metallic prompt preset. This final script fixes the prompt to the gate_black wording.')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--azure_endpoint', '--base_url', dest='azure_endpoint', type=str, default=os.environ.get('AZURE_OPENAI_ENDPOINT') or DEFAULT_AZURE_ENDPOINT, help='Azure OpenAI endpoint; --base_url is kept as a compatibility alias.')
    parser.add_argument('--api_version', type=str, default=os.environ.get('AZURE_OPENAI_API_VERSION') or DEFAULT_API_VERSION, help='Azure OpenAI API version.')
    parser.add_argument('--image_model', type=str, default=DEFAULT_IMAGE_MODEL, help='Image generation model')
    parser.add_argument('--image_deployment', type=str, default=None, help='Azure deployment name, defaults to --image_model')
    parser.add_argument('--generation_mode', type=str, default='edit', choices=['edit'], help='GPT Image call path follows roughness_gpt_image_15.py and uses images.edit.')
    parser.add_argument('--generate_requires_image', action='store_true', help='In generate mode, fail if the SDK cannot send the image, instead of silently falling back to text-only generation.')
    parser.add_argument('--size', type=str, default=DEFAULT_IMAGE_SIZE, help='Output size for GPT Image 2')
    parser.add_argument('--quality', type=str, default=DEFAULT_IMAGE_QUALITY, choices=['low', 'medium', 'high', 'auto'])
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--watermark', action='store_true', help='Kept for compatibility; ignored by GPT Image 2.')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='Request and download timeout')
    parser.add_argument('--sleep', type=float, default=0.2, help='Sleep between requests')
    parser.add_argument('--max_generate', type=int, default=0, help='0 means process all images')
    parser.add_argument('--skip_existing', action='store_true', help='Skip images whose outputs already exist')
    parser.add_argument('--save_debug_intermediates', action='store_true', help='Save input/upload and raw response copies for debugging.')
    return parser.parse_args()


def ensure_api_key(cli_api_key: Optional[str] = None) -> str:
    api_key = (
        cli_api_key
        or os.environ.get("AZURE_OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Missing authentication: please provide --api_key or set "
            "AZURE_OPENAI_API_KEY."
        )
    return api_key.strip()


def resolve_image_client_config(args: argparse.Namespace) -> Tuple[Any, str, str, Optional[str], Optional[str]]:
    api_key = ensure_api_key(args.api_key)
    endpoint = (
        args.azure_endpoint
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
    )
    api_version = (
        args.api_version
        or os.environ.get("AZURE_OPENAI_API_VERSION")
    )
    deployment = args.image_deployment or args.image_model
    if not endpoint:
        raise ValueError("Azure endpoint is required: set AZURE_OPENAI_ENDPOINT or pass the endpoint argument.")
    if AzureOpenAI is None:
        raise ImportError('Failed to import openai. Please install it with: pip install openai') from _OPENAI_IMPORT_ERROR
    client = AzureOpenAI(
        api_version=api_version or DEFAULT_API_VERSION,
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    return client, deployment, "azure", endpoint, api_version or DEFAULT_API_VERSION


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


def normalize_gpt_image_size(size: Optional[str]) -> str:
    normalized = (size or DEFAULT_IMAGE_SIZE).strip().lower()
    if normalized not in SUPPORTED_GPT_IMAGE_SIZES:
        raise ValueError(
            f"Unsupported --size: {size}. Expected one of: {', '.join(sorted(SUPPORTED_GPT_IMAGE_SIZES))}"
        )
    return normalized


def parse_size_tuple(size: str) -> Tuple[int, int]:
    normalized = normalize_gpt_image_size(size)
    if normalized == "auto":
        return MODEL_INPUT_SIZE
    width_text, height_text = normalized.split("x", 1)
    return int(width_text), int(height_text)




def prepare_upload_image(image_path, *, target_size=None):
    """Prepare the upload; no padding or source-size restriction."""
    temp_dir = Path(tempfile.mkdtemp(prefix="image_upload_"))
    temp_path = temp_dir / image_path.name
    box = prepare_input(image_path, temp_path, gpt=True)
    with Image.open(image_path) as image:
        original_size = list(image.size)
    return temp_path, {"original_size": original_size, "canvas_size": box[2:],
                      "content_box": box, "resized_size": box[2:], "temp_path": str(temp_path)}




def postprocess_generated_image_bytes(image_bytes, *args, **kwargs):
    """Keep the complete provider response byte-for-byte."""
    return image_bytes


def list_images(input_dir: Path) -> List[Path]:
    images = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.name.startswith("Image_") and p.suffix.lower() == ".png"
    ]
    if images:
        return images
    raise FileNotFoundError(f"No PNG images were found in {input_dir}")


def run_image_generation(
    image_client: Any,
    model: str,
    prompt: str,
    image_paths: List[Path],
    size: str,
    watermark: bool,
    quality: str,
) -> Tuple[Any, str]:
    del watermark
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    requested_size = normalize_gpt_image_size(size)

    with contextlib.ExitStack() as stack:
        image_files = [stack.enter_context(path.open("rb")) for path in image_paths]
        image_arg: Any = image_files[0] if len(image_files) == 1 else image_files
        response = image_client.images.edit(
            model=model,
            image=image_arg,
            prompt=prompt,
            n=1,
            size=requested_size,
            quality=quality,
            output_format="png",
        )
        return response, requested_size


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


def copy_image_for_debug(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def save_debug_bundle(
    *,
    rgb_path: Path,
    prepared_target_path: Path,
    raw_image_bytes: bytes,
    final_output_path: Path,
) -> None:
    debug_dir = final_output_path.parent.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = final_output_path.stem
    copy_image_for_debug(rgb_path, debug_dir / f"{stem}_01_original.png")
    copy_image_for_debug(prepared_target_path, debug_dir / f"{stem}_02_padded_input.png")
    (debug_dir / f"{stem}_03_raw_generated.png").write_bytes(raw_image_bytes)
    copy_image_for_debug(final_output_path, debug_dir / f"{stem}_04_restored_output.png")




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


def build_run_signature(*, input_mode: str, prompt_version: str, prompt_preset: str, image_model: str, prompt_text: str, timeout: int, size: str, quality: str, seed: int, watermark: bool) -> Dict[str, Any]:
    route = 'gpt_image_rgb_only_metallic'
    return {'image_model': image_model, 'route': route, 'input_mode': input_mode, 'prompt_version': prompt_version, 'prompt_preset': prompt_preset, 'prompt_text': prompt_text, 'timeout': timeout, 'size': size, 'quality': quality, 'seed': seed, 'model_input_size': list(MODEL_INPUT_SIZE), 'output_resolution_policy': 'crop_to_original_after_letterbox', 'watermark': watermark}


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


def format_exception_with_context(
    exc: Exception,
    *,
    endpoint: Optional[str],
    model: str,
    input_mode: str,
    image_paths: List[Path],
) -> str:
    exc_type = type(exc).__name__
    exc_text = str(exc).strip() or repr(exc)
    image_names = [path.name for path in image_paths]
    parts = [
        f"{exc_type}: {exc_text}",
        f"model={model}",
        f"input_mode={input_mode}",
        f"endpoint={endpoint or 'N/A'}",
        f"images={image_names}",
    ]
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        cause_text = str(cause).strip() or repr(cause)
        parts.append(f"cause={type(cause).__name__}: {cause_text}")
    return " | ".join(parts)


def generate_metallic_map_with_gpt_image(*, client: Any, endpoint: Optional[str], model: str, input_mode: str, prompt_preset: str, rgb_path: Path, save_path: Path, size: str, quality: str, seed: int, timeout: int, save_debug_intermediates: bool) -> str:
    prompt = build_metallic_prompt(input_mode=input_mode, prompt_preset=prompt_preset)
    prepared_paths: List[Path] = []
    temp_dirs: List[Path] = []
    primary_preprocess_meta: Optional[Dict[str, Any]] = None
    target_prepared_path: Optional[Path] = None
    try:
        prepared_path, primary_preprocess_meta = prepare_upload_image(rgb_path)
        prepared_paths = [prepared_path]
        temp_dirs.append(prepared_path.parent)
        target_prepared_path = prepared_path
        mode = 'gpt_image_rgb_only_metallic'
        response, requested_size = run_image_generation(image_client=client, model=model, prompt=prompt, image_paths=prepared_paths, size=size, watermark=False, quality=quality)
        final_size = tuple(primary_preprocess_meta['original_size']) if primary_preprocess_meta is not None else None
        raw_image_bytes = get_image_response_bytes(response.data[0], timeout=timeout)
        if primary_preprocess_meta is not None:
            final_image_bytes = postprocess_generated_image_bytes(raw_image_bytes)
        else:
            final_image_bytes = raw_image_bytes
        save_path.write_bytes(final_image_bytes)
        if save_debug_intermediates and target_prepared_path is not None:
            save_debug_bundle(rgb_path=rgb_path, prepared_target_path=target_prepared_path, raw_image_bytes=raw_image_bytes, final_output_path=save_path)
        if primary_preprocess_meta is None:
            raise RuntimeError('Missing preprocess metadata for target RGB image.')
        return mode
    except Exception as exc:
        detail_paths = prepared_paths if prepared_paths else [rgb_path]
        raise RuntimeError(format_exception_with_context(exc, endpoint=endpoint, model=model, input_mode=input_mode, image_paths=detail_paths)) from exc
    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    client, image_model, client_kind, resolved_endpoint, resolved_api_version = resolve_image_client_config(args)
    prompt_version = f'{PROMPT_FAMILY}_{args.prompt_preset}'
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
    print(f'[1/3] found {len(image_paths)} RGB images; generating {len(image_paths_for_generate)} metallic maps with model={image_model}, input_mode={args.input_mode}')
    print(f'      client_kind={client_kind} | endpoint={resolved_endpoint or "N/A"} | api_version={resolved_api_version or "N/A"} | quality={args.quality} | size={normalize_gpt_image_size(args.size)}')
    prompt_text = build_metallic_prompt(args.input_mode, prompt_preset=args.prompt_preset)
    run_signature = build_run_signature(input_mode=args.input_mode, prompt_version=prompt_version, prompt_preset=args.prompt_preset, image_model=image_model, prompt_text=prompt_text, timeout=args.timeout, size=args.size, quality=args.quality, seed=args.seed, watermark=args.watermark)
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
                item['metallic_mode'] = generate_metallic_map_with_gpt_image(client=client, endpoint=resolved_endpoint, model=image_model, input_mode=args.input_mode, prompt_preset=args.prompt_preset, rgb_path=rgb_path, save_path=metallic_path, size=args.size, quality=args.quality, seed=args.seed, timeout=args.timeout, save_debug_intermediates=args.save_debug_intermediates)
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
