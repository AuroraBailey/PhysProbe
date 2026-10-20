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

from experiments.ablation.albedo.albedo_provider_utils import (
    SUPPORTED_EXTS,
    add_provider_args,
    extract_error_code,
    normalize_provider_name,
    resolve_provider_runtime,
    run_albedo_generation_with_retry,
    save_generated_image,
    should_skip_image_error,
)


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    variant_name: str
    prompt_version: str
    description: str
    prompt_text: str
    use_example_pair: bool = False
    prompt_level: str = "fixed"


from experiments.ablation.albedo.prompts import FULL_PROMPT_TEXT, WEAKENED_PROMPT_TEXT, MINIMAL_PROMPT_TEXT, MINIMAL_FIXED_EXEMPLAR_PROMPT_TEXT
from common.utils.gemini_client import provider_request_record

PROJECT_ROOT = Path(__file__).resolve().parents[3]




DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ablation" / "albedo"
DEFAULT_INPUT_DIR = str(PROJECT_ROOT / "data" / "raw" / "ablation_ADN")


def parse_args(config: VariantConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--filename_suffix", type=str, default="_im.png")
    parser.add_argument("--recursive", dest="recursive", action="store_true")
    parser.add_argument("--non_recursive", dest="recursive", action="store_false")
    parser.set_defaults(recursive=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    add_provider_args(parser, include_analysis=False)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--albedo_size", type=str, default="adaptive")
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--max_generate", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_parts", type=int, default=1)
    parser.add_argument("--part_index", type=int, default=0)
    parser.add_argument("--preserve_relative_dirs", dest="preserve_relative_dirs", action="store_true")
    parser.add_argument("--flat_output", dest="preserve_relative_dirs", action="store_false")
    parser.set_defaults(preserve_relative_dirs=True)
    parser.add_argument("--sleep", type=float, default=40)
    parser.add_argument("--timeout", type=int, default=200)
    parser.add_argument("--request_retries", type=int, default=5, help="generation 阶段遇到网络抖动、超时、限流、5xx 时的重试次数。")
    parser.add_argument("--retry_backoff", type=float, default=45, help="generation 阶段首次重试前的等待秒数，后续按指数退避。")
    parser.add_argument("--retry_max_backoff", type=float, default=90.0, help="generation 阶段单次重试等待的最大秒数。")
    parser.add_argument(
        "--example_rgb",
        type=str,
        required=config.use_example_pair,
        default="",
        help="固定 exemplar 版本要求提供 example RGB。",
    )
    parser.add_argument(
        "--example_albedo",
        type=str,
        required=config.use_example_pair,
        default="",
        help="固定 exemplar 版本要求提供 example albedo。",
    )
    return parser.parse_args()


def resolve_default_output_dir(config: VariantConfig, provider: str) -> Path:
    subdir_map = {
        "B0": "b0",
    }
    variant_dir = subdir_map.get(config.variant_id, config.variant_id.lower())
    return DEFAULT_OUTPUT_ROOT / f"{provider}_result" / variant_dir


def validate_parts(num_parts: int, part_index: int) -> None:
    if num_parts < 1:
        raise ValueError("--num_parts 必须 >= 1")
    if part_index < 0 or part_index >= num_parts:
        raise ValueError("--part_index 必须满足 0 <= part_index < num_parts")


def shard_paths(paths: List[Path], num_parts: int, part_index: int) -> List[Path]:
    if num_parts == 1:
        return list(paths)
    return [path for idx, path in enumerate(paths) if idx % num_parts == part_index]


def list_images(input_dir: Path, filename_suffix: Optional[str], recursive: bool) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = [path for path in sorted(iterator) if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS]
    if filename_suffix:
        images = [path for path in images if path.name.endswith(filename_suffix)]
    if not images:
        raise FileNotFoundError(f"目录中没有找到支持的图片文件: {input_dir}")
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


def upsert_manifest_entry(manifest: List[Dict[str, Any]], entry: Dict[str, Any], key_field: str = "relative_image_path") -> None:
    entry_key = entry.get(key_field)
    if entry_key is None:
        manifest.append(entry)
        return
    for index, item in enumerate(manifest):
        if item.get(key_field) == entry_key:
            manifest[index] = entry
            return
    manifest.append(entry)


def is_completed_output(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def build_skip_path(meta_dir: Path, num_parts: int, part_index: int) -> Path:
    if num_parts == 1:
        return meta_dir / "skipped_images.json"
    return meta_dir / f"skipped_images.part{part_index + 1}of{num_parts}.json"


def record_skipped_image(
    skipped_images: List[Dict[str, Any]],
    skipped_path: Path,
    image_path: Path,
    input_dir: Path,
    stage: str,
    exc: Exception,
    args: argparse.Namespace,
) -> None:
    entry = {
        "skip_key": f"{image_path.relative_to(input_dir).as_posix()}::{stage}",
        "image_name": image_path.name,
        "relative_image_path": image_path.relative_to(input_dir).as_posix(),
        "stage": stage,
        "error_code": extract_error_code(exc),
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
        "part_index": args.part_index,
        "num_parts": args.num_parts,
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


def build_image_output_path(image_path: Path, input_dir: Path, base_dir: Path, suffix: str, preserve_relative_dirs: bool) -> Path:
    if preserve_relative_dirs:
        relative_path = image_path.relative_to(input_dir)
        return base_dir / relative_path.parent / f"{image_path.stem}{suffix}"
    return base_dir / f"{image_path.stem}{suffix}"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_pending_images(
    image_paths: List[Path],
    input_dir: Path,
    albedo_dir: Path,
    overwrite: bool,
    preserve_relative_dirs: bool,
) -> List[Path]:
    if overwrite:
        return list(image_paths)
    pending: List[Path] = []
    for image_path in image_paths:
        out_albedo = build_image_output_path(image_path, input_dir, albedo_dir, "_albedo.png", preserve_relative_dirs)
        if is_completed_output(out_albedo):
            continue
        pending.append(image_path)
    return pending


def build_setup_payload(config: VariantConfig, args: argparse.Namespace, runtime: Any) -> Dict[str, Any]:
    return provider_request_record({
        "variant_id": config.variant_id,
        "variant_name": config.variant_name,
        "prompt_version": config.prompt_version,
        "prompt_level": config.prompt_level,
        "analysis_enabled": False,
        "use_example_pair": config.use_example_pair,
        "prompt_text": config.prompt_text,
        "provider": runtime.provider,
        "analysis_model": runtime.analysis_model,
        "albedo_model": runtime.albedo_model,
        "analysis_base_url": runtime.analysis_base_url,
        "albedo_base_url": runtime.albedo_base_url,
        "generation_mode": runtime.generation_mode,
        "albedo_quality": runtime.albedo_quality,
        "example_rgb": args.example_rgb if config.use_example_pair else "",
        "example_albedo": args.example_albedo if config.use_example_pair else "",
    }, runtime.provider)


def run_variant(config: VariantConfig) -> None:
    args = parse_args(config)
    validate_parts(args.num_parts, args.part_index)
    if not args.output_dir:
        provider = normalize_provider_name(args.provider)
        args.output_dir = str(resolve_default_output_dir(config, provider))
    runtime = resolve_provider_runtime(args, analysis_enabled=False)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    albedo_dir = output_dir / "albedo"
    meta_dir = output_dir / "meta"
    per_image_meta_dir = meta_dir / "per_image"
    albedo_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    per_image_meta_dir.mkdir(parents=True, exist_ok=True)

    example_rgb = Path(args.example_rgb) if config.use_example_pair else None
    example_albedo = Path(args.example_albedo) if config.use_example_pair else None
    if config.use_example_pair:
        if not example_rgb or not example_albedo:
            raise ValueError("固定 exemplar 版本需要同时提供 --example_rgb 和 --example_albedo")
        if not example_rgb.exists():
            raise FileNotFoundError(f"example_rgb not found: {example_rgb}")
        if not example_albedo.exists():
            raise FileNotFoundError(f"example_albedo not found: {example_albedo}")

    image_paths = list_images(input_dir, args.filename_suffix, args.recursive)
    image_paths_all = image_paths[: args.max_generate] if args.max_generate > 0 else image_paths
    image_paths_for_generate = shard_paths(image_paths_all, args.num_parts, args.part_index)
    pending_image_paths = get_pending_images(
        image_paths_for_generate,
        input_dir,
        albedo_dir,
        args.overwrite,
        args.preserve_relative_dirs,
    )
    shard_label = f"part {args.part_index + 1}/{args.num_parts}"

    print(
        f"[1/3] found {len(image_paths)} images; current {shard_label} handles "
        f"{len(image_paths_for_generate)} images; pending {len(pending_image_paths)}; "
        f"variant={config.variant_id} ({config.variant_name}) | provider={runtime.provider} | "
        f"example_pair={config.use_example_pair}"
    )
    print(
        "      "
        f"output_dir={output_dir.as_posix()} | "
        f"filename_suffix={args.filename_suffix or 'N/A'} | "
        f"recursive={args.recursive} | "
        f"preserve_relative_dirs={args.preserve_relative_dirs} | "
        f"albedo_model={runtime.albedo_model} | "
        f"albedo_base_url={runtime.albedo_base_url} | "
        f"generation_mode={runtime.generation_mode}"
    )

    setup_path = meta_dir / "setup.json"
    setup_path.write_text(
        json.dumps(build_setup_payload(config, args, runtime), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest_path = meta_dir / "manifest.json"
    skipped_path = build_skip_path(meta_dir, args.num_parts, args.part_index)
    manifest = load_manifest(manifest_path)
    skipped_images = load_manifest(skipped_path)

    if not pending_image_paths:
        print("[2/3] nothing to do; all outputs already exist for this shard")
        print("[3/3] done")
        return

    print("[2/3] analysis disabled for this ablation; using fixed prompt only")
    print(f"[3/3] start albedo generation for {len(pending_image_paths)} images")

    for idx, image_path in enumerate(image_paths_for_generate, start=1):
        out_albedo = build_image_output_path(
            image_path,
            input_dir,
            albedo_dir,
            "_albedo.png",
            args.preserve_relative_dirs,
        )
        out_meta = build_image_output_path(
            image_path,
            input_dir,
            per_image_meta_dir,
            "_albedo.json",
            args.preserve_relative_dirs,
        )

        skipped_entry = None if args.overwrite else was_image_skipped(skipped_images, image_path, input_dir)
        if skipped_entry is not None:
            print(
                f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} "
                f"此前已跳过，原因={skipped_entry.get('error_code', 'N/A')}"
            )
            continue
        if not args.overwrite and is_completed_output(out_albedo):
            print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} exists, skip")
            continue

        print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name}")
        input_images = [image_path]
        if config.use_example_pair and example_rgb and example_albedo:
            input_images = [example_rgb, example_albedo, image_path]
        try:
            response = run_albedo_generation_with_retry(
                runtime,
                config.prompt_text,
                input_images,
                args.albedo_size,
                args.watermark,
                args.seed,
                args.guidance_scale,
                args.timeout,
                args.request_retries,
                args.retry_backoff,
                args.retry_max_backoff,
            )
        except Exception as exc:
            if should_skip_image_error(exc):
                print(f"      跳过：generation 阶段触发内容审核 -> {extract_error_code(exc)}")
                record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation", exc, args)
                continue
            raise

        ensure_parent_dir(out_albedo)
        image_url = save_generated_image(runtime, response, out_albedo, timeout=args.timeout)

        per_image_meta = provider_request_record({
            "image_name": image_path.name,
            "relative_image_path": image_path.relative_to(input_dir).as_posix(),
            "variant_id": config.variant_id,
            "variant_name": config.variant_name,
            "prompt_version": config.prompt_version,
            "prompt_level": config.prompt_level,
            "analysis_enabled": False,
            "use_example_pair": config.use_example_pair,
            "provider": runtime.provider,
            "analysis_model": runtime.analysis_model,
            "albedo_model": runtime.albedo_model,
            "analysis_base_url": runtime.analysis_base_url,
            "albedo_base_url": runtime.albedo_base_url,
            "generation_mode": runtime.generation_mode,
            "albedo_output": out_albedo.relative_to(output_dir).as_posix(),
            "albedo_prompt": config.prompt_text,
            "example_rgb": str(example_rgb) if example_rgb else "",
            "example_albedo": str(example_albedo) if example_albedo else "",
            "image_url": image_url,
            "seed": args.seed,
            "input_order": [str(path) for path in ([example_rgb, example_albedo, image_path] if config.use_example_pair else [image_path])],
            "num_parts": args.num_parts,
            "part_index": args.part_index,
        }, runtime.provider)
        ensure_parent_dir(out_meta)
        out_meta.write_text(json.dumps(per_image_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        upsert_manifest_entry(manifest, provider_request_record({**per_image_meta, "per_image_meta": out_meta.relative_to(output_dir).as_posix()}, runtime.provider))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(max(0.0, args.sleep))

    print("done")


def main() -> None:
    raise RuntimeError("Import this module from a specific ablation script instead of running it directly.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
