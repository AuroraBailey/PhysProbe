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


import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.ablation.normal.normal_provider_utils import (
    SUPPORTED_EXTS,
    add_provider_args,
    extract_error_code,
    normalize_provider_name,
    resolve_provider_runtime,
    run_normal_generation,
    save_generated_image,
    should_skip_image_error,
)


from experiments.ablation.normal.prompts import FULL_RGB_ONLY_PROMPT, FULL_RGB_PLUS_EXAMPLE_PROMPT, MINIMAL_RGB_ONLY_PROMPT, MINIMAL_RGB_PLUS_EXAMPLE_PROMPT

@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    variant_name: str
    prompt_version: str
    description: str
    prompt_text: str
    use_example_pair: bool
    prompt_level: str


def parse_args(config: VariantConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--filename_suffix", type=str, default="_im.png")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--api_key", type=str, default=None)
    add_provider_args(parser)
    parser.add_argument("--normal_model", type=str, default=None)
    parser.add_argument("--normal_size", type=str, default="adaptive")
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--max_generate", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preserve_relative_dirs", action="store_true")
    parser.add_argument("--max_retries", type=int, default=3, help="Retry count per image on generation failure.")
    parser.add_argument("--retry_sleep", type=float, default=3.0, help="Sleep seconds between retries.")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--example_rgb",
        type=str,
        required=config.use_example_pair,
        default="",
        help="Required for variants that use a fixed example pair.",
    )
    parser.add_argument(
        "--example_normal",
        type=str,
        required=config.use_example_pair,
        default="",
        help="Required for variants that use a fixed example pair.",
    )
    return parser.parse_args()


def list_images(input_dir: Path, filename_suffix: Optional[str], recursive: bool) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = [path for path in sorted(iterator) if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS]
    if filename_suffix:
        images = [path for path in images if path.name.endswith(filename_suffix)]
    if not images:
        raise FileNotFoundError(f"No supported images found in {input_dir}")
    return images


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = load_json_file(path)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def upsert_manifest_entry(
    manifest: List[Dict[str, Any]],
    entry: Dict[str, Any],
    key_field: str = "relative_image_path",
) -> None:
    entry_key = entry.get(key_field)
    if entry_key is None:
        manifest.append(entry)
        return
    for idx, item in enumerate(manifest):
        if item.get(key_field) == entry_key:
            manifest[idx] = entry
            return
    manifest.append(entry)


def build_skip_path(meta_dir: Path) -> Path:
    return meta_dir / "skipped_images.json"


def record_skipped_image(
    skipped_images: List[Dict[str, Any]],
    skipped_path: Path,
    image_path: Path,
    input_dir: Path,
    config: VariantConfig,
    exc: Exception,
    attempts: int,
) -> None:
    entry = {
        "skip_key": f"{config.variant_id}::{image_path.relative_to(input_dir).as_posix()}",
        "variant_id": config.variant_id,
        "image_name": image_path.name,
        "relative_image_path": image_path.relative_to(input_dir).as_posix(),
        "error_code": extract_error_code(exc),
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
        "attempts": attempts,
        "skipped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    upsert_manifest_entry(skipped_images, entry, key_field="skip_key")
    skipped_path.write_text(json.dumps(skipped_images, ensure_ascii=False, indent=2), encoding="utf-8")


def is_completed_output(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


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
            image_path,
            input_dir,
            normal_dir,
            "_normal.png",
            preserve_relative_dirs,
        )
        if is_completed_output(out_normal):
            continue
        pending.append(image_path)
    return pending


def build_setup_payload(config: VariantConfig, args: argparse.Namespace, runtime: Any) -> Dict[str, Any]:
    return {
        "variant_id": config.variant_id,
        "variant_name": config.variant_name,
        "prompt_version": config.prompt_version,
        "prompt_level": config.prompt_level,
        "use_example_pair": config.use_example_pair,
        "prompt_text": config.prompt_text,
        "provider": runtime.provider,
        "normal_model": runtime.normal_model,
        "normal_base_url": runtime.normal_base_url,
        "generation_mode": runtime.generation_mode,
        "normal_quality": runtime.normal_quality,
        "normal_size": args.normal_size,
        "watermark": args.watermark,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "max_generate": args.max_generate,
        "overwrite": args.overwrite,
        "preserve_relative_dirs": args.preserve_relative_dirs,
        "max_retries": args.max_retries,
        "retry_sleep": args.retry_sleep,
        "filename_suffix": args.filename_suffix,
        "recursive": args.recursive,
        "example_rgb": args.example_rgb if config.use_example_pair else "",
        "example_normal": args.example_normal if config.use_example_pair else "",
    }


def run_variant(config: VariantConfig) -> None:
    args = parse_args(config)
    runtime = resolve_provider_runtime(args)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    normal_dir = output_dir / "normal"
    meta_dir = output_dir / "meta"
    normal_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    example_rgb = Path(args.example_rgb) if config.use_example_pair else None
    example_normal = Path(args.example_normal) if config.use_example_pair else None
    if config.use_example_pair:
        if not example_rgb or not example_normal:
            raise ValueError("This variant requires both --example_rgb and --example_normal")
        if not example_rgb.exists():
            raise FileNotFoundError(f"example_rgb not found: {example_rgb}")
        if not example_normal.exists():
            raise FileNotFoundError(f"example_normal not found: {example_normal}")

    image_paths = list_images(input_dir, args.filename_suffix, args.recursive)
    image_paths_for_generate = image_paths[: args.max_generate] if args.max_generate > 0 else image_paths
    pending_image_paths = get_pending_images(
        image_paths_for_generate,
        input_dir,
        normal_dir,
        args.overwrite,
        args.preserve_relative_dirs,
    )

    manifest_path = meta_dir / "manifest.json"
    skipped_path = build_skip_path(meta_dir)
    manifest = load_manifest(manifest_path)
    skipped_images = load_manifest(skipped_path)
    setup_path = meta_dir / "setup.json"
    setup_path.write_text(
        json.dumps(build_setup_payload(config, args, runtime), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[1/3] found {len(image_paths)} images; selected {len(image_paths_for_generate)}; "
        f"pending {len(pending_image_paths)}; variant={config.variant_id}; "
        f"provider={runtime.provider}; prompt_level={config.prompt_level}; example_pair={config.use_example_pair}"
    )
    print(
        "      "
        f"filename_suffix={args.filename_suffix or 'N/A'} | "
        f"recursive={args.recursive} | "
        f"preserve_relative_dirs={args.preserve_relative_dirs} | "
        f"normal_model={runtime.normal_model} | "
        f"normal_base_url={runtime.normal_base_url} | "
        f"generation_mode={runtime.generation_mode}"
    )
    print("[2/3] fixed prompt ablation mode; analysis disabled")
    print(f"[3/3] start normal generation for {len(pending_image_paths)} pending images")

    for idx, image_path in enumerate(image_paths_for_generate, start=1):
        out_normal = build_image_output_path(
            image_path,
            input_dir,
            normal_dir,
            "_normal.png",
            args.preserve_relative_dirs,
        )
        if not args.overwrite and is_completed_output(out_normal):
            print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} exists, skip")
            continue

        print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name}")
        input_images = [image_path]
        if config.use_example_pair and example_rgb and example_normal:
            input_images = [image_path, example_rgb, example_normal]

        last_exc: Optional[Exception] = None
        response = None
        for attempt in range(1, max(1, args.max_retries) + 1):
            try:
                response = run_normal_generation(
                    runtime=runtime,
                    prompt=config.prompt_text,
                    image_paths=input_images,
                    size=args.normal_size,
                    watermark=args.watermark,
                    seed=args.seed,
                    guidance_scale=args.guidance_scale,
                    timeout=args.timeout,
                )
                break
            except Exception as exc:
                last_exc = exc
                if should_skip_image_error(exc) or attempt >= max(1, args.max_retries):
                    break
                print(
                    f"      generation failed on attempt {attempt}/{max(1, args.max_retries)}: {exc} | retrying"
                )
                time.sleep(max(0.0, args.retry_sleep))

        if response is None:
            assert last_exc is not None
            print(f"      skip after {max(1, args.max_retries)} failed attempt(s): {last_exc}")
            record_skipped_image(
                skipped_images=skipped_images,
                skipped_path=skipped_path,
                image_path=image_path,
                input_dir=input_dir,
                config=config,
                exc=last_exc,
                attempts=max(1, args.max_retries),
            )
            continue

        ensure_parent_dir(out_normal)
        image_url = save_generated_image(runtime, response, out_normal, timeout=args.timeout)

        per_image_meta = {
            "image_name": image_path.name,
            "relative_image_path": image_path.relative_to(input_dir).as_posix(),
            "variant_id": config.variant_id,
            "variant_name": config.variant_name,
            "prompt_version": config.prompt_version,
            "prompt_level": config.prompt_level,
            "use_example_pair": config.use_example_pair,
            "provider": runtime.provider,
            "normal_model": runtime.normal_model,
            "normal_base_url": runtime.normal_base_url,
            "generation_mode": runtime.generation_mode,
            "normal_quality": runtime.normal_quality,
            "normal_output": out_normal.relative_to(output_dir).as_posix(),
            "normal_prompt": config.prompt_text,
            "image_url": image_url,
                "seed": args.seed,
                "input_order": [str(path) for path in input_images],
            "example_rgb": str(example_rgb) if example_rgb else "",
            "example_normal": str(example_normal) if example_normal else "",
        }
        upsert_manifest_entry(manifest, per_image_meta)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(max(0.0, args.sleep))

    print("Done.")
    print(f"Normal output dir: {normal_dir.as_posix()}")
    print(f"Meta output dir: {meta_dir.as_posix()}")












def main() -> None:
    raise RuntimeError("Import this module from a specific ablation script instead of running it directly.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
