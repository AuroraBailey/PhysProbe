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
import hashlib
import os
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.ablation.albedo.albedo_provider_utils import (
    DOUBAO_ANALYSIS_MODEL,
    DOUBAO_BASE_URL,
    SUPPORTED_EXTS,
    add_provider_args,
    extract_error_code,
    normalize_provider_name,
    resolve_provider_runtime,
    run_albedo_generation_with_retry,
    run_analysis_request,
    save_generated_image,
    should_skip_image_error,
)


from experiments.ablation.albedo.prompts import build_single_image_analysis_prompt, make_single_image_albedo_prompt
from common.utils.gemini_client import provider_request_record

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ablation" / "albedo"
DEFAULT_INPUT_DIR = str(PROJECT_ROOT / "data" / "raw" / "ablation_ADN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A1 ablation: single-image analysis + albedo generation")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--filename_suffix", type=str, default="_im.png")
    parser.add_argument("--recursive", dest="recursive", action="store_true", help="递归扫描 input_dir 下的所有子目录。")
    parser.add_argument("--non_recursive", dest="recursive", action="store_false", help="仅扫描 input_dir 第一层。")
    parser.set_defaults(recursive=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    add_provider_args(parser, include_analysis=True)
    parser.add_argument(
        "--analysis_dir",
        type=str,
        required=True,
        help=(
            "共享的单图光照分析缓存目录。"
            "必须先生成缓存；A1 不调用分析模型。"
        ),
    )
    parser.add_argument("--analysis_max_side", type=int, default=0, help="Legacy argument; analysis uploads now preserve original image bytes.")
    parser.add_argument("--detail", type=str, default="high", choices=["low", "high", "auto"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--albedo_size", type=str, default="adaptive")
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--max_generate", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_parts", type=int, default=1)
    parser.add_argument("--part_index", type=int, default=0)
    parser.add_argument("--independent_images", action="store_true", help="兼容旧命令保留参数。")
    parser.add_argument("--preserve_relative_dirs", dest="preserve_relative_dirs", action="store_true")
    parser.add_argument("--flat_output", dest="preserve_relative_dirs", action="store_false")
    parser.set_defaults(preserve_relative_dirs=True)
    parser.add_argument(
        "--analysis_scope",
        type=str,
        default="full",
        choices=["full", "part"],
        help="兼容旧命令保留参数；当前始终逐图单独分析。",
    )
    parser.add_argument("--sleep", type=float, default=40)
    parser.add_argument("--timeout", type=int, default=200)
    parser.add_argument("--request_retries", type=int, default=5, help="generation 阶段遇到网络抖动、超时、限流、5xx 时的重试次数。")
    parser.add_argument("--retry_backoff", type=float, default=45, help="generation 阶段首次重试前的等待秒数，后续按指数退避。")
    parser.add_argument("--retry_max_backoff", type=float, default=90.0, help="generation 阶段单次重试等待的最大秒数。")
    return parser.parse_args()


def resolve_default_output_dir(provider: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"{provider}_result" / "a1"




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


def clean_json_text(text: str) -> str:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    return text


def robust_json_loads(text: str) -> Dict[str, Any]:
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_completed_output(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


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




def analyze_single_image(runtime: Any, image_path: Path, detail: str, analysis_max_side: int) -> Dict[str, Any]:
    text = run_analysis_request(
        runtime,
        image_path,
        build_single_image_analysis_prompt(image_path.name),
        detail,
        analysis_max_side,
    )
    data = robust_json_loads(text)
    data.setdefault("scene_summary", "")
    data.setdefault("albedo_material_notes", [])
    data.setdefault("albedo_lighting_notes", [])
    data.setdefault("albedo_prompt_suffix", "")
    data.setdefault("albedo_hint", "")
    return data




def build_setup_payload(args: argparse.Namespace, runtime: Any) -> Dict[str, Any]:
    return provider_request_record({
        "variant_id": "A1",
        "variant_name": "Single Image Analysis + Full Prompt",
        "analysis_enabled": True,
        "analysis_cache_only": True,
        "analysis_api_called": False,
        "analysis_source_dir": args.analysis_dir or "",
        "provider": runtime.provider,
        "analysis_model": runtime.analysis_model,
        "albedo_model": runtime.albedo_model,
        "analysis_base_url": runtime.analysis_base_url,
        "albedo_base_url": runtime.albedo_base_url,
        "generation_mode": runtime.generation_mode,
        "albedo_quality": runtime.albedo_quality,
        "analysis_max_side": args.analysis_max_side,
        "detail": args.detail,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "albedo_size": args.albedo_size,
        "watermark": args.watermark,
        "max_generate": args.max_generate,
        "overwrite": args.overwrite,
        "num_parts": args.num_parts,
        "part_index": args.part_index,
        "preserve_relative_dirs": args.preserve_relative_dirs,
        "filename_suffix": args.filename_suffix,
        "recursive": args.recursive,
    }, runtime.provider)


def main() -> None:
    args = parse_args()
    validate_parts(args.num_parts, args.part_index)
    if not args.output_dir:
        provider = normalize_provider_name(args.provider)
        args.output_dir = str(resolve_default_output_dir(provider))
    external_analysis_dir = Path(args.analysis_dir).expanduser()
    if not args.analysis_dir or not external_analysis_dir.is_dir():
        raise FileNotFoundError("A1 requires --analysis_dir pointing to the shared illumination cache")
    args.analysis_model = DOUBAO_ANALYSIS_MODEL
    args.analysis_base_url = os.environ.get("ARK_BASE_URL") or DOUBAO_BASE_URL
    runtime = resolve_provider_runtime(args, analysis_enabled=False)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    albedo_dir = output_dir / "albedo"
    meta_dir = output_dir / "meta"
    albedo_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

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

    analysis_dir = meta_dir / "per_image_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = meta_dir / "manifest.json"
    skipped_path = build_skip_path(meta_dir, args.num_parts, args.part_index)
    setup_path = meta_dir / "setup.json"
    setup_path.write_text(json.dumps(build_setup_payload(args, runtime), ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = load_manifest(manifest_path)
    skipped_images = load_manifest(skipped_path)
    shard_label = f"part {args.part_index + 1}/{args.num_parts}"

    print(
        f"[1/3] 单图 analysis 模式：找到 {len(image_paths)} 张图，当前分片 {shard_label} 负责 "
        f"{len(image_paths_for_generate)} 张，待处理 {len(pending_image_paths)} 张 | provider={runtime.provider}"
    )
    print(
        "      "
        f"output_dir={output_dir.as_posix()} | "
        f"filename_suffix={args.filename_suffix or 'N/A'} | "
        f"recursive={args.recursive} | "
        f"preserve_relative_dirs={args.preserve_relative_dirs} | "
        f"analysis_model={runtime.analysis_model} | "
        f"albedo_model={runtime.albedo_model} | "
        f"analysis_base_url={runtime.analysis_base_url} | "
        f"albedo_base_url={runtime.albedo_base_url}"
    )
    if external_analysis_dir is not None:
        print(f"[2/3] 复用已有单图 analysis: {external_analysis_dir.as_posix()}")
    else:
        print("[2/3] 不再进行多视角联合分析，改为逐张单图 analysis")
    print(f"[3/3] 开始逐张导出 albedo（待处理 {len(pending_image_paths)} 张）")

    if not pending_image_paths:
        print("  - 检测到当前分片已全部完成，直接退出。")

    for idx, image_path in enumerate(image_paths_for_generate, start=1):
        out_albedo = build_image_output_path(image_path, input_dir, albedo_dir, "_albedo.png", args.preserve_relative_dirs)
        single_analysis_path = build_image_output_path(image_path, input_dir, analysis_dir, ".json", args.preserve_relative_dirs)
        skipped_entry = None if args.overwrite else was_image_skipped(skipped_images, image_path, input_dir)
        if skipped_entry is not None:
            print(
                f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} "
                f"此前已跳过，原因={skipped_entry.get('error_code', 'N/A')}"
            )
            continue
        if not args.overwrite and is_completed_output(out_albedo):
            print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} 已存在，跳过")
            continue

        print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name}")
        external_analysis_path = build_image_output_path(image_path, input_dir, external_analysis_dir, ".json", args.preserve_relative_dirs)
        if not external_analysis_path.is_file():
            raise FileNotFoundError(f"Shared analysis is missing for {image_path}: {external_analysis_path}")
        single_analysis = load_json_file(external_analysis_path)
        ensure_parent_dir(single_analysis_path)
        single_analysis_path.write_bytes(external_analysis_path.read_bytes())

        albedo_prompt = make_single_image_albedo_prompt(single_analysis)
        try:
            response = run_albedo_generation_with_retry(
                runtime,
                albedo_prompt,
                [image_path],
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
        upsert_manifest_entry(
            manifest,
            provider_request_record({
                "image_name": image_path.name,
                "relative_image_path": image_path.relative_to(input_dir).as_posix(),
                "variant_id": "A1",
                "variant_name": "Single Image Analysis + Full Prompt",
                "provider": runtime.provider,
                "analysis_model": runtime.analysis_model,
                "albedo_model": runtime.albedo_model,
                "analysis_base_url": runtime.analysis_base_url,
                "albedo_base_url": runtime.albedo_base_url,
                "generation_mode": runtime.generation_mode,
                "albedo_quality": runtime.albedo_quality,
                "albedo_output": out_albedo.relative_to(output_dir).as_posix(),
                "analysis_output": single_analysis_path.relative_to(output_dir).as_posix(),
                "albedo_prompt": albedo_prompt,
                "albedo_hint": single_analysis.get("albedo_hint", ""),
                "scene_summary": single_analysis.get("scene_summary", ""),
                "albedo_material_notes": single_analysis.get("albedo_material_notes", []),
                "albedo_lighting_notes": single_analysis.get("albedo_lighting_notes", []),
                "image_url": image_url,
                "seed": args.seed,
                "input_order": [str(image_path)],
                "analysis_source": str(external_analysis_path),
                "analysis_sha256": hashlib.sha256(external_analysis_path.read_bytes()).hexdigest(),
                "num_parts": args.num_parts,
                "part_index": args.part_index,
            }, runtime.provider),
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(max(0.0, args.sleep))

    print("完成。")
    print(f"albedo 输出目录: {albedo_dir.as_posix()}")
    print(f"meta 输出目录: {meta_dir.as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        sys.exit(130)
