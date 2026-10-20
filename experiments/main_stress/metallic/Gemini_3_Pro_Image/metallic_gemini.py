#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Metallic generation script using Gemini with letterbox preprocessing.

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
build_metallic_prompt = _prompt_partial(_prompt_builder, model='gemini')
PROMPT_FAMILY = _PROMPT_CONFIGS['gemini']['PROMPT_FAMILY']
PROMPT_PRESET_CHOICES = _PROMPT_CONFIGS['gemini']['PROMPT_PRESET_CHOICES']


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
from common.utils.gemini_client import GEMINI_IMAGE_MODEL, GeminiImageClient, gemini_request_record


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_IMAGE_MODEL = GEMINI_IMAGE_MODEL
DEFAULT_TIMEOUT = 1800
DEFAULT_DATASET_ROOT = str(PROJECT_ROOT / "data" / "raw" / "pbr_rooms")
MODEL_INPUT_SIZE = (1536, 1024)
INPUT_MODE_CHOICES = ("rgb_only",)


DEFAULT_IMAGE_SIZE = "1536x1024"

DEFAULT_IMAGE_QUALITY = "medium"

SUPPORTED_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate metallic maps with Gemini using the final gate_black prompt.')
    parser.add_argument('--input_dir', type=str, default='', help='Deprecated compatibility argument; scene_list controls the RGB inputs.')
    parser.add_argument('--scene_list', type=str, required=True, help='Text file of scene paths relative to --dataset_root.')
    parser.add_argument('--dataset_root', type=str, default=DEFAULT_DATASET_ROOT, help='Root directory used to resolve paths in --scene_list.')
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).resolve().parent / 'output' / 'metallic'), help='Output directory (default: gemini/output/metallic).')
    parser.add_argument('--input_mode', type=str, default='rgb_only', choices=INPUT_MODE_CHOICES, help='Main/stress RGB-only generation.')
    parser.add_argument('--prompt_preset', type=str, default='v3_visualprior_noboundary', choices=PROMPT_PRESET_CHOICES, help='Built-in metallic prompt preset. This final script fixes the prompt to the gate_black wording.')
    parser.add_argument('--base_url', type=str, default=os.environ.get('AIHUBMIX_BASE_URL'), help='Aihubmix endpoint; defaults to AIHUBMIX_BASE_URL.')
    parser.add_argument('--api_key', type=str, default=None, help='Aihubmix API key')
    parser.add_argument('--image_model', type=str, default=DEFAULT_IMAGE_MODEL, help='Image generation model')
    parser.add_argument('--generation_mode', type=str, default='edit', choices=['edit'], help='Gemini image editing path.')
    parser.add_argument('--generate_requires_image', action='store_true', help='In generate mode, fail if the SDK cannot send the image, instead of silently falling back to text-only generation.')
    parser.add_argument('--size', type=str, default=DEFAULT_IMAGE_SIZE, help='Output size for Gemini')
    parser.add_argument('--quality', type=str, default=DEFAULT_IMAGE_QUALITY, choices=['low', 'medium', 'high', 'auto'])
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--watermark', action='store_true', help='Kept for compatibility; ignored by Gemini.')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='Request and download timeout')
    parser.add_argument('--sleep', type=float, default=0.2, help='Sleep between requests')
    parser.add_argument('--max_generate', type=int, default=0, help='0 means process all images')
    parser.add_argument('--skip_existing', action='store_true', help='Skip images whose outputs already exist')
    parser.add_argument('--save_debug_intermediates', action='store_true', help='Save input/upload and raw response copies for debugging.')
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


def resolve_image_client_config(args: argparse.Namespace) -> Tuple[Any, str, str, Optional[str], Optional[str]]:
    api_key = ensure_api_key(getattr(args, "albedo_api_key", None) or getattr(args, "api_key", None))
    return GeminiImageClient(api_key, base_url=args.base_url, aspect_ratio="16:9", timeout=args.timeout), args.image_model, "aihubmix", args.base_url or "https://aihubmix.com/v1", None


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








def prepare_upload_image(image_path, *, target_size=None):
    """Prepare the upload; no padding or source-size restriction."""
    temp_dir = Path(tempfile.mkdtemp(prefix="image_upload_"))
    temp_path = temp_dir / image_path.name
    box = prepare_input(image_path, temp_path, gpt=False)
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


def list_images_from_scene_list(scene_list_path: Path, dataset_root: Path) -> List[Path]:
    """List PNG RGB inputs only from sampled scenes V*/Image folders."""
    if not scene_list_path.is_file():
        raise FileNotFoundError(f"Scene list was not found: {scene_list_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root was not found: {dataset_root}")
    scene_paths = [dataset_root / line.strip() for line in scene_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not scene_paths:
        raise RuntimeError(f"Scene list is empty: {scene_list_path}")
    images: List[Path] = []
    for scene_path in scene_paths:
        if not scene_path.is_dir():
            raise FileNotFoundError(f"Sampled scene was not found: {scene_path}")
        if scene_path.parent.name == "Image" and scene_path.name.startswith("camera_"):
            images.extend(sorted(scene_path.glob("Image_*.png")))
        else:
            images.extend(path for path in sorted(scene_path.rglob("Image_*.png"))
                          if path.parent.parent.name == "Image")
    if not images:
        raise FileNotFoundError("No PNG files were found in sampled scenes V*/Image folders")
    return images


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
    requested_size = normalize_image_size(size)

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


def build_run_signature(*, input_mode: str, prompt_version: str, prompt_preset: str, image_model: str, prompt_text: str, timeout: int, size: str, quality: str, seed: int, watermark: bool) -> Dict[str, Any]:
    route = 'gemini_image_rgb_only_metallic'
    return gemini_request_record({'image_model': image_model, 'route': route, 'input_mode': input_mode, 'prompt_version': prompt_version, 'prompt_preset': prompt_preset, 'prompt_text': prompt_text, 'timeout': timeout, 'size': size, 'quality': quality, 'seed': seed, 'input_resize_policy': 'model_input_native', 'output_resolution_policy': 'provider_native_bytes', 'watermark': watermark})


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


def generate_metallic_map_with_gemini_image(*, client: Any, endpoint: Optional[str], model: str, input_mode: str, prompt_preset: str, rgb_path: Path, save_path: Path, size: str, quality: str, seed: int, timeout: int, save_debug_intermediates: bool) -> str:
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
        mode = 'gemini_image_rgb_only_metallic'
        response, requested_size = run_image_generation(image_client=client, model=model, prompt=prompt, image_paths=prepared_paths, size=size, watermark=False, quality=quality)
        final_size = None
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
    client, image_model, client_kind, resolved_endpoint, _resolved_api_version = resolve_image_client_config(args)
    prompt_version = f'{PROMPT_FAMILY}_{args.prompt_preset}'
    dataset_root = Path(args.dataset_root)
    scene_list_path = Path(args.scene_list)
    input_dir = Path(args.input_dir) if args.input_dir else dataset_root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metallic_dir = output_dir
    meta_dir = output_dir / 'meta'
    output_meta_dir = meta_dir / 'per_image'
    metallic_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images_from_scene_list(scene_list_path, dataset_root)
    image_paths_for_generate = image_paths[:args.max_generate] if args.max_generate > 0 else image_paths
    prompt_text = build_metallic_prompt(args.input_mode, prompt_preset=args.prompt_preset)
    run_signature = build_run_signature(input_mode=args.input_mode, prompt_version=prompt_version, prompt_preset=args.prompt_preset, image_model=image_model, prompt_text=prompt_text, timeout=args.timeout, size=args.size, quality=args.quality, seed=args.seed, watermark=args.watermark)
    skip_count = 0
    pending_count = 0
    if args.skip_existing:
        for rgb_path in image_paths_for_generate:
            image_relative_path = rgb_path.relative_to(dataset_root)
            metallic_path = metallic_dir / image_relative_path.with_name(f'{rgb_path.stem}_metallic.png')
            output_meta_path = output_meta_dir / image_relative_path.with_name(f'{rgb_path.stem}_metallic.json')
            if should_skip_existing_output(metallic_path=metallic_path, output_meta_path=output_meta_path, run_signature=run_signature):
                skip_count += 1
            else:
                pending_count += 1
    else:
        pending_count = len(image_paths_for_generate)
    print(f'[1/3] found {len(image_paths)} RGB images; candidates={len(image_paths_for_generate)}; pending={pending_count}; skipped={skip_count}; model={image_model}, input_mode={args.input_mode}')
    print(f'      client_kind={client_kind} | endpoint={resolved_endpoint or "N/A"} | quality={args.quality} | size={normalize_image_size(args.size)}')
    setup = dict(run_signature)
    (meta_dir / 'setup.json').write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest_path = meta_dir / 'manifest.json'
    manifest: List[Dict[str, Any]] = load_manifest(manifest_path)
    print(f'[2/3] process {len(image_paths_for_generate)} candidates; generate {pending_count} pending images')
    failures = 0
    for idx, rgb_path in enumerate(image_paths_for_generate, start=1):
        image_relative_path = rgb_path.relative_to(dataset_root)
        item: Dict[str, Any] = {'image_name': image_relative_path.as_posix()}
        try:
            print(f'  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name}')
            metallic_path = metallic_dir / image_relative_path.with_name(f'{rgb_path.stem}_metallic.png')
            output_meta_path = output_meta_dir / image_relative_path.with_name(f'{rgb_path.stem}_metallic.json')
            metallic_path.parent.mkdir(parents=True, exist_ok=True)
            output_meta_path.parent.mkdir(parents=True, exist_ok=True)
            item['prompt_version'] = prompt_version
            item['input_mode'] = args.input_mode
            if args.skip_existing and should_skip_existing_output(metallic_path=metallic_path, output_meta_path=output_meta_path, run_signature=run_signature):
                item['skipped'] = True
                item['skip_reason'] = 'matching_output_and_signature'
                item['metallic_output'] = metallic_path.relative_to(metallic_dir).as_posix()
                print(f'      skipped=matching_output_and_signature | output={item["metallic_output"]}', flush=True)
            else:
                item['metallic_mode'] = generate_metallic_map_with_gemini_image(client=client, endpoint=resolved_endpoint, model=image_model, input_mode=args.input_mode, prompt_preset=args.prompt_preset, rgb_path=rgb_path, save_path=metallic_path, size=args.size, quality=args.quality, seed=args.seed, timeout=args.timeout, save_debug_intermediates=args.save_debug_intermediates)
                item['metallic_output'] = metallic_path.relative_to(metallic_dir).as_posix()
                output_meta = gemini_request_record({'image_name': image_relative_path.as_posix(), 'metallic_output': metallic_path.relative_to(metallic_dir).as_posix(), 'input_mode': args.input_mode, 'run_signature': run_signature})
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





def normalize_image_size(size: Optional[str]) -> str:
    normalized = (size or DEFAULT_IMAGE_SIZE).strip().lower()
    if normalized not in SUPPORTED_IMAGE_SIZES:
        raise ValueError(
            f"Unsupported --size: {size}. Expected one of: {', '.join(sorted(SUPPORTED_IMAGE_SIZES))}"
        )
    return normalized



def parse_size_tuple(size: str) -> Tuple[int, int]:
    normalized = normalize_image_size(size)
    if normalized == "auto":
        return MODEL_INPUT_SIZE
    width_text, height_text = normalized.split("x", 1)
    return int(width_text), int(height_text)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
