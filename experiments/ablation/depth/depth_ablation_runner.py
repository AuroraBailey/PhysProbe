#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from experiments.ablation.depth.depth_provider_utils import (
    add_provider_args,
    resolve_provider_runtime,
    run_depth_generation_with_retry,
    save_generated_image,
)


from experiments.ablation.depth.prompts import build_direct_prompt, build_example_prompt, build_seg_prompt
from common.utils.gemini_client import provider_request_record

PROMPT_VERSION_A0 = "relative_depth_benchmark_v10_near_white_only"
PROMPT_VERSION_A1 = "relative_depth_benchmark_v10_near_white_only"
PROMPT_VERSION_A2 = "relative_depth_benchmark_v11_rgb_plus_seg_near_white"
FIXED_DEPTH_POLARITY = "near_white"
SCRIPT_DIR = Path(__file__).resolve().parent
FIXED_EXAMPLE_RGB = SCRIPT_DIR / "examples/image.png"
FIXED_EXAMPLE_DEPTH = SCRIPT_DIR / "examples/depth.png"








@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    variant_name: str
    description: str
    prompt_version: str
    prompt_kind: str
    input_mode: str


def parse_args(config: VariantConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--input_dir", type=str, required=True, help="RGB image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--base_url", type=str, default=None, help="Provider base URL; 留空时按 provider 自动选择。")
    parser.add_argument("--image_model", type=str, default=None, help="Provider image model; 留空时按 provider 自动选择。")
    parser.add_argument("--size", type=str, default="adaptive", help="输出尺寸；不同 provider 会做各自适配。")
    parser.add_argument("--recursive", dest="recursive", action="store_true", help="递归扫描 input_dir 下的所有子目录。")
    parser.add_argument("--non_recursive", dest="recursive", action="store_false", help="仅扫描 input_dir 第一层。")
    parser.set_defaults(recursive=True)
    parser.add_argument("--preserve_relative_dirs", dest="preserve_relative_dirs", action="store_true", help="输出时保留相对目录结构。")
    parser.add_argument("--flat_output", dest="preserve_relative_dirs", action="store_false", help="输出时平铺到同一目录。")
    parser.set_defaults(preserve_relative_dirs=True)
    parser.add_argument("--watermark", action="store_true", help="Keep watermark")
    parser.add_argument("--sleep", type=float, default=40)
    parser.add_argument("--timeout", type=int, default=200)
    parser.add_argument("--request_retries", type=int, default=5, help="generation 阶段遇到网络抖动、超时、限流、5xx 时的重试次数。")
    parser.add_argument("--retry_backoff", type=float, default=45, help="generation 阶段首次重试前的等待秒数，后续按指数退避。")
    parser.add_argument("--retry_max_backoff", type=float, default=90.0, help="generation 阶段单次重试等待的最大秒数。")
    parser.add_argument("--max_generate", type=int, default=0, help="0 means process all images")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images whose outputs already exist with the same run signature")
    add_provider_args(parser)
    return parser.parse_args()


def list_images(input_dir: Path, recursive: bool) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    preferred = [p for p in sorted(iterator) if p.is_file() and p.name.lower().endswith("_im.png")]
    if preferred:
        return preferred
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    image_prefix = [p for p in sorted(iterator) if p.is_file() and p.suffix.lower() == ".png" and p.name.lower().startswith("image_")]
    if image_prefix:
        return image_prefix
    raise FileNotFoundError(f"No RGB images matching *_im.png or Image_*.png were found in {input_dir}")


def list_rgb_images(input_dir: Path, recursive: bool) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    image_paths = [p for p in sorted(iterator) if p.is_file() and p.suffix.lower() == ".png" and p.stem.lower().endswith("_im")]
    if not image_paths:
        raise FileNotFoundError(f"No RGB images matching *_im.png were found in {input_dir}")
    return image_paths


def get_paired_seg_path(rgb_path: Path) -> Path:
    seg_stem = rgb_path.stem[:-3] + "_seg"
    return rgb_path.with_name(seg_stem + rgb_path.suffix)


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


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


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
    image_name = str(item.get("relative_image_path") or item.get("image_name") or "").strip()
    if not image_name:
        manifest.append(item)
        return
    for idx, existing in enumerate(manifest):
        if str(existing.get("relative_image_path") or existing.get("image_name") or "").strip() == image_name:
            manifest[idx] = item
            return
    manifest.append(item)


def get_prompt_text(config: VariantConfig) -> str:
    if config.prompt_kind == "direct":
        return build_direct_prompt()
    if config.prompt_kind == "example":
        return build_example_prompt()
    if config.prompt_kind == "seg":
        return build_seg_prompt()
    raise ValueError(f"Unknown prompt_kind: {config.prompt_kind}")


def get_route_name(config: VariantConfig, provider: str) -> str:
    suffix_map = {
        "direct": "rgb_relative_depth_direct",
        "example": "rgb_relative_depth_example",
        "seg": "rgb_seg_relative_depth",
    }
    return f"{provider}_{suffix_map[config.prompt_kind]}"


def build_run_signature(
    config: VariantConfig,
    args: argparse.Namespace,
    runtime: Any,
    prompt_text: str,
    example_rgb: Optional[Path],
    example_depth: Optional[Path],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "provider": runtime.provider,
        "image_model": runtime.image_model,
        "base_url": runtime.base_url,
        "route": get_route_name(config, runtime.provider),
        "input_mode": config.input_mode,
        "prompt_version": config.prompt_version,
        "variant_id": config.variant_id,
        "seed": args.seed,
        "prompt_text": prompt_text,
        "depth_representation": "relative_depth",
        "depth_polarity": FIXED_DEPTH_POLARITY,
        "timeout": args.timeout,
        "sleep": args.sleep,
        "request_retries": args.request_retries,
        "retry_backoff": args.retry_backoff,
        "retry_max_backoff": args.retry_max_backoff,
        "size": args.size,
        "watermark": args.watermark,
        "recursive": args.recursive,
        "preserve_relative_dirs": args.preserve_relative_dirs,
    }
    if config.prompt_kind == "seg":
        payload["segmentation_policy"] = "paired_same_dir_suffix_im_to_seg"
        payload["evaluation_protocol"] = "single_image_relative_depth_prediction_with_segmentation_prior"
    else:
        payload["evaluation_protocol"] = "single_image_relative_depth_prediction"
    if example_rgb is not None:
        payload["example_rgb"] = str(example_rgb)
    if example_depth is not None:
        payload["example_depth"] = str(example_depth)
    if runtime.provider in {"gpt", "gpt2"}:
        payload["gpt_backend"] = runtime.gpt_backend
        payload["image_quality"] = runtime.image_quality
    return provider_request_record(payload, runtime.provider)


def should_skip_existing_output(depth_path: Path, output_meta_path: Path, run_signature: Dict[str, Any]) -> bool:
    return depth_path.is_file()


def run_variant(config: VariantConfig) -> None:
    args = parse_args(config)
    runtime = resolve_provider_runtime(args)
    input_dir = Path(args.input_dir)

    if config.prompt_kind in {"direct", "example"} and looks_like_object_seg_dir(input_dir):
        inferred_rgb_dir = infer_rgb_dir_from_seg_dir(input_dir)
        if inferred_rgb_dir is None:
            raise RuntimeError(f"input_dir looks like an ObjectSegmentation directory but the paired RGB directory could not be inferred: {input_dir}")
        print(f"[info] detected segmentation directory as input_dir, switching RGB directory to: {inferred_rgb_dir}")
        input_dir = inferred_rgb_dir

    example_rgb = FIXED_EXAMPLE_RGB if config.prompt_kind == "example" else None
    example_depth = FIXED_EXAMPLE_DEPTH if config.prompt_kind == "example" else None
    if example_rgb is not None and (not example_rgb.exists() or not example_rgb.is_file()):
        raise FileNotFoundError(f"example_rgb not found: {example_rgb}")
    if example_depth is not None and (not example_depth.exists() or not example_depth.is_file()):
        raise FileNotFoundError(f"example_depth not found: {example_depth}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "relative_depth"
    meta_dir = output_dir / "meta"
    output_meta_dir = meta_dir / "per_image"
    depth_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)

    if config.prompt_kind == "seg":
        image_paths = list_rgb_images(input_dir, args.recursive)
    else:
        image_paths = list_images(input_dir, args.recursive)
    image_paths_for_generate = image_paths[: args.max_generate] if args.max_generate > 0 else image_paths

    prompt_text = get_prompt_text(config)
    run_signature = build_run_signature(config, args, runtime, prompt_text, example_rgb, example_depth)
    (meta_dir / "setup.json").write_text(json.dumps(run_signature, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = meta_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    print(
        f"[1/3] found {len(image_paths)} RGB images; generating {len(image_paths_for_generate)} relative depth maps "
        f"with provider={runtime.provider}, model={runtime.image_model}, input_mode={config.input_mode}, depth_polarity={FIXED_DEPTH_POLARITY}"
    )
    print(
        "      "
        f"recursive={args.recursive} | preserve_relative_dirs={args.preserve_relative_dirs} | output_dir={output_dir.as_posix()}"
    )
    print(f"[2/3] start generation for {len(image_paths_for_generate)} images")

    failures = 0
    for idx, rgb_path in enumerate(image_paths_for_generate, start=1):
        item: Dict[str, Any] = {
            "image_name": rgb_path.name,
            "relative_image_path": rgb_path.relative_to(input_dir).as_posix(),
            "prompt_version": config.prompt_version,
        "variant_id": config.variant_id,
        "seed": args.seed,
            "input_mode": config.input_mode,
            "depth_representation": "relative_depth",
            "depth_polarity": FIXED_DEPTH_POLARITY,
            "provider": runtime.provider,
            "image_model": runtime.image_model,
        }
        try:
            depth_path = build_image_output_path(
                rgb_path,
                input_dir,
                depth_dir,
                "_relative_depth.png",
                args.preserve_relative_dirs,
            )
            output_meta_path = build_image_output_path(
                rgb_path,
                input_dir,
                output_meta_dir,
                "_relative_depth.json",
                args.preserve_relative_dirs,
            )
            input_images = [rgb_path]

            if config.prompt_kind == "example":
                assert example_rgb is not None and example_depth is not None
                input_images = [example_rgb, example_depth, rgb_path]
                item["example_rgb"] = example_rgb.name
                item["example_depth"] = example_depth.name
                print(f"  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name} | example={example_rgb.name}")
            elif config.prompt_kind == "seg":
                seg_path = get_paired_seg_path(rgb_path)
                if not seg_path.exists() or not seg_path.is_file():
                    raise FileNotFoundError(f"Paired segmentation map not found: {seg_path}")
                input_images = [rgb_path, seg_path]
                item["segmentation_name"] = seg_path.name
                item["relative_segmentation_path"] = seg_path.relative_to(input_dir).as_posix()
                print(f"  - ({idx}/{len(image_paths_for_generate)}) rgb={rgb_path.name} | seg={seg_path.name}")
            else:
                print(f"  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name}")

            item["input_order"] = [str(path) for path in input_images]
            if args.skip_existing and should_skip_existing_output(depth_path, output_meta_path, run_signature):
                item["skipped"] = True
                item["skip_reason"] = "existing_output_image"
                item["relative_depth_output"] = depth_path.relative_to(output_dir).as_posix()
                print(f"      skipped: output image already exists: {depth_path}")
            else:
                response = run_depth_generation_with_retry(
                    runtime,
                    prompt_text,
                    input_images,
                    args.size,
                    args.watermark,
                    args.seed,
                    args.timeout,
                    args.request_retries,
                    args.retry_backoff,
                    args.retry_max_backoff,
                )
                ensure_parent_dir(depth_path)
                ensure_parent_dir(output_meta_path)
                image_url = save_generated_image(runtime, response, depth_path, timeout=args.timeout)
                item["depth_mode"] = get_route_name(config, runtime.provider)
                item["relative_depth_output"] = depth_path.relative_to(output_dir).as_posix()
                if image_url:
                    item["image_url"] = image_url
                output_meta: Dict[str, Any] = {
                    "image_name": rgb_path.name,
                    "relative_image_path": rgb_path.relative_to(input_dir).as_posix(),
                    "relative_depth_output": depth_path.relative_to(output_dir).as_posix(),
                    "input_mode": config.input_mode,
                    "provider": runtime.provider,
                    "image_model": runtime.image_model,
                    "run_signature": run_signature,
                }
                if config.prompt_kind == "example":
                    output_meta["example_rgb"] = str(example_rgb)
                    output_meta["example_depth"] = str(example_depth)
                if config.prompt_kind == "seg":
                    output_meta["segmentation_name"] = item["segmentation_name"]
                    output_meta["relative_segmentation_path"] = item["relative_segmentation_path"]
                output_meta_path.write_text(json.dumps(output_meta, ensure_ascii=False, indent=2), encoding="utf-8")
                time.sleep(max(0.0, args.sleep))
        except Exception as e:
            failures += 1
            item["error"] = str(e)
            item["status"] = "error"
            print(f"[error] {rgb_path.name}: {e}")

        upsert_manifest_entry(manifest, provider_request_record(item, runtime.provider))
        write_manifest(manifest_path, manifest)
    if failures:
        raise RuntimeError(f"{failures} image generation(s) failed; see manifest and log")

    print("[3/3] done")
    print(f"output_dir: {output_dir.as_posix()}")
