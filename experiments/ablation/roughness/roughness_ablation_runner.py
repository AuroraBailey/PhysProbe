#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Paper roughness access-cue generation runner.
# A0 RGB-only; A1 soft segmentation; A2 connected-component region fill; A3 fixed exemplar.

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
import base64
import json
import os
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DEPTH_UTILS_DIR = SCRIPT_DIR.parent / "depth"
if str(DEPTH_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPTH_UTILS_DIR))

from experiments.ablation.depth.depth_provider_utils import (  # noqa: E402
    add_provider_args,
    resolve_provider_runtime,
    run_depth_generation_with_retry,
    save_generated_image,
)

try:
    from volcenginesdkarkruntime import Ark
except Exception as e:
    Ark = None
    _ARK_IMPORT_ERROR = e
else:
    _ARK_IMPORT_ERROR = None


DEFAULT_BASE_URL = os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_IMAGE_MODEL = "doubao-seedream-5-0-260128"
DEFAULT_TIMEOUT = 200
SEG_CANDIDATE_SUFFIXES = ["_seg", "_mask", "_sam", "_semantic", "_label", ""]
EXAMPLE_RGB_CANDIDATES = (
    "image.png",
    "image.jpg",
    "image.jpeg",
    "example_rgb.png",
    "example_rgb.jpg",
    "rgb.png",
)
EXAMPLE_ROUGHNESS_CANDIDATES = (
    "roughness.png",
    "roughness.jpg",
    "roughness.jpeg",
    "example_roughness.png",
    "example_roughness.jpg",
    "gt_roughness.png",
)


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    variant_name: str
    input_mode: str
    route: str
    segmentation_role: str
    use_example_pair: bool
    postprocess_mode: str
    prompt_version: str





from experiments.ablation.roughness.prompts import build_prompt
from common.utils.gemini_client import provider_request_record

VARIANT_CONFIGS = {
    "a0": VariantConfig("a0", "RGB only", "rgb_only", "roughness_rgb_only", "unused", False, "none", "roughness_a0_rgb_v1"),
    "a1": VariantConfig("a1", "Soft segmentation", "rgb_plus_seg", "roughness_seg_soft", "soft_prior", False, "none", "roughness_a1_soft_seg_v1"),
    "a2": VariantConfig("a2", "Segmentation region fill", "rgb_plus_seg", "roughness_seg_region_fill", "region_constraint", False, "connected_component_mean", "roughness_a2_region_fill_v1"),
    "a3": VariantConfig("a3", "Fixed exemplar", "rgb_plus_example", "roughness_example", "unused", True, "none", "roughness_a3_example_v1"),
}
DEFAULT_VARIANT_BY_INPUT_MODE = {"rgb_only": "a0", "rgb_plus_seg": "a1", "rgb_plus_example": "a3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper roughness access-cue settings across image providers."
    )
    parser.add_argument(
        "--variant",
        type=str.lower,
        required=True,
        choices=tuple(VARIANT_CONFIGS),
        help="Paper setting: a0 RGB-only, a1 soft segmentation, a2 region fill, a3 fixed exemplar.",
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Target RGB image directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output root. Results are written under <output_dir>/<variant>/.",
    )
    parser.add_argument(
        "--example_rgb",
        type=str,
        default="",
        help="Reference RGB image; required only for --variant a3.",
    )
    parser.add_argument(
        "--example_roughness",
        type=str,
        default="",
        help="Reference roughness map; required only for --variant a3.",
    )
    parser.add_argument("--seg_dir", default="", help="Segmentation root mirroring input_dir; default uses paired *_seg.png.")
    parser.add_argument("--base_url", type=str, default=None, help="Provider base URL; 留空时按 provider 自动选择。")
    parser.add_argument("--image_model", type=str, default=None, help="Provider image model; 留空时按 provider 自动选择。")
    parser.add_argument("--size", type=str, default="adaptive", help="Output size; adaptive -> 2k")
    parser.add_argument("--watermark", action="store_true", help="Keep watermark")
    parser.add_argument("--sleep", type=float, default=40)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--request_retries", type=int, default=5, help="generation 阶段遇到网络抖动、超时、限流、5xx 时的重试次数。")
    parser.add_argument("--retry_backoff", type=float, default=45, help="generation 阶段首次重试前的等待秒数，后续按指数退避。")
    parser.add_argument("--retry_max_backoff", type=float, default=90.0, help="generation 阶段单次重试等待的最大秒数。")
    parser.add_argument("--max_generate", type=int, default=0, help="0 means process all images")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images whose roughness output already exists")
    add_provider_args(parser, providers=("doubao", "gpt", "gpt2", "gemini"))
    args = parser.parse_args()
    if args.provider not in {"doubao", "gpt", "gpt2", "gemini"}:
        parser.error("The paper roughness ablation supports doubao/gpt/gpt2/gemini only")
    return args


def get_variant_config(variant_id: str) -> VariantConfig:
    key = str(variant_id or "").strip().lower()
    if key not in VARIANT_CONFIGS:
        raise ValueError(f"Unsupported variant_id: {variant_id}")
    return VARIANT_CONFIGS[key]


def get_default_variant_for_input_mode(input_mode: str) -> VariantConfig:
    key = str(input_mode or "").strip().lower()
    if key not in DEFAULT_VARIANT_BY_INPUT_MODE:
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    return get_variant_config(DEFAULT_VARIANT_BY_INPUT_MODE[key])


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
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"RGB image directory not found: {input_dir}")

    preferred = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.name.lower().endswith("_im.png")
    ]
    if preferred:
        return preferred

    recursive_preferred = [
        p for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.name.lower().endswith("_im.png")
    ]
    if recursive_preferred:
        return recursive_preferred

    image_prefix = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() == ".png" and p.name.lower().startswith("image_")
    ]
    if image_prefix:
        return image_prefix

    recursive_image_prefix = [
        p for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() == ".png" and p.name.lower().startswith("image_")
    ]
    if recursive_image_prefix:
        return recursive_image_prefix

    raise FileNotFoundError(f"No RGB images matching *_im.png or Image_*.png were found in {input_dir} or its subdirectories")


def filter_images_by_names(image_paths: Sequence[Path], include_names: Optional[Sequence[str]]) -> List[Path]:
    if not include_names:
        return list(image_paths)

    wanted = [str(name).strip() for name in include_names if str(name).strip()]
    if not wanted:
        return list(image_paths)

    image_map = {path.name: path for path in image_paths}
    image_map.update({path.as_posix(): path for path in image_paths})
    missing = [name for name in wanted if name not in image_map]
    if missing:
        raise FileNotFoundError(f"Requested image(s) not found in input_dir: {missing}")
    return [image_map[name] for name in wanted]


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

def resolve_fixed_example_pair(explicit_rgb: str, explicit_roughness: str) -> Tuple[Path, Path]:
    if explicit_rgb or explicit_roughness:
        if not explicit_rgb or not explicit_roughness:
            raise ValueError("Please provide both --example_rgb and --example_roughness together.")
        return Path(explicit_rgb).expanduser(), Path(explicit_roughness).expanduser()

    for rgb_name in EXAMPLE_RGB_CANDIDATES:
        rgb_path = SCRIPT_DIR / "examples" / rgb_name
        if not rgb_path.exists() or not rgb_path.is_file():
            continue
        for roughness_name in EXAMPLE_ROUGHNESS_CANDIDATES:
            roughness_path = SCRIPT_DIR / "examples" / roughness_name
            if roughness_path.exists() and roughness_path.is_file():
                return rgb_path, roughness_path

    raise FileNotFoundError(
        "Could not auto-detect the local roughness example pair. "
        f"Tried RGB candidates={list(EXAMPLE_RGB_CANDIDATES)} and roughness candidates={list(EXAMPLE_ROUGHNESS_CANDIDATES)} in {SCRIPT_DIR}"
    )


def normalize_seg_suffix(suffix: str) -> str:
    text = str(suffix or "").strip().lower()
    if not text:
        return ""
    if text in {"none", "bare", "no_suffix", "nosuffix"}:
        return ""
    if not text.startswith("_"):
        text = "_" + text.lstrip("_")
    return text


def get_seg_candidate_suffixes(preferred_suffix: str = "") -> List[str]:
    preferred = normalize_seg_suffix(preferred_suffix)
    if not preferred:
        return list(SEG_CANDIDATE_SUFFIXES)
    suffixes = [preferred]
    suffixes.extend([suffix for suffix in SEG_CANDIDATE_SUFFIXES if suffix != preferred])
    return suffixes


def find_matching_seg(rgb_path: Path, seg_dir: Path, preferred_suffix: str = "") -> Path:
    stem = rgb_path.stem
    candidates: List[Path] = []
    seg_suffixes = get_seg_candidate_suffixes(preferred_suffix)
    candidate_dirs = [seg_dir]

    if rgb_path.parent != seg_dir:
        candidate_dirs.append(rgb_path.parent)

    try:
        rel_parent = rgb_path.parent.relative_to(seg_dir)
    except ValueError:
        rel_parent = None
    if rel_parent is not None and rel_parent != Path("."):
        candidate_dirs.append(seg_dir / rel_parent)

    for candidate_dir in candidate_dirs:
        for suffix in seg_suffixes:
            for ext in [rgb_path.suffix, ".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
                candidates.append(candidate_dir / f"{stem}{suffix}{ext}")

    name_lower = rgb_path.name.lower()
    if name_lower.startswith("image_"):
        seg_name = "ObjectSegmentation_" + rgb_path.name[len("Image_"):]
        for candidate_dir in candidate_dirs:
            candidates.append(candidate_dir / seg_name)

    for tail in ["_im", "_rgb", "_image"]:
        if stem.endswith(tail):
            root = stem[: -len(tail)]
            for candidate_dir in candidate_dirs:
                for suffix in seg_suffixes:
                    for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
                        candidates.append(candidate_dir / f"{root}{suffix}{ext}")

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            if candidate.resolve() == rgb_path.resolve():
                continue
        except Exception:
            if candidate == rgb_path:
                continue
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Cannot find segmentation image for {rgb_path.as_posix()} in {seg_dir}")










def build_roughness_prompt(input_mode: str) -> str:
    config = get_default_variant_for_input_mode(input_mode)
    return build_prompt(config.variant_id)


def resolve_input_images(
    variant_id: str,
    rgb_path: Path,
    seg_path: Optional[Path] = None,
    example_pair: Optional[Tuple[Path, Path]] = None,
) -> List[Path]:
    config = get_variant_config(variant_id)
    if config.input_mode == "rgb_only":
        return [rgb_path]
    if config.input_mode == "rgb_plus_example":
        if example_pair is None:
            raise ValueError(f"{variant_id} requires an example_pair")
        example_rgb, example_roughness = example_pair
        return [example_rgb, example_roughness, rgb_path]
    if seg_path is None:
        raise ValueError(f"{variant_id} requires a matching segmentation image")
    return [rgb_path, seg_path]


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
    with urllib.request.urlopen(url, timeout=timeout) as response, open(save_path, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
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


def enforce_grayscale_png(path: Path) -> None:
    with Image.open(path) as image:
        image.convert("L").save(path)


def apply_region_fill_from_segmentation(pred_path: Path, seg_path: Path, save_path: Path) -> None:
    with Image.open(pred_path) as pred_img:
        pred_np = np.array(pred_img.convert("L"), dtype=np.float32)

    with Image.open(seg_path) as seg_img:
        seg_rgb_img = seg_img.convert("RGB")
        pred_height, pred_width = pred_np.shape[:2]
        if seg_rgb_img.size != (pred_width, pred_height):
            seg_rgb_img = seg_rgb_img.resize((pred_width, pred_height), Image.Resampling.NEAREST)
        seg_rgb = np.array(seg_rgb_img, dtype=np.uint32)

    label_map = (seg_rgb[..., 0] << 16) | (seg_rgb[..., 1] << 8) | seg_rgb[..., 2]
    height, width = label_map.shape
    visited = np.zeros((height, width), dtype=bool)
    output = pred_np.copy()

    for y in range(height):
        for x in range(width):
            if visited[y, x]:
                continue
            label_value = int(label_map[y, x])
            queue: deque[Tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            component_pixels: List[Tuple[int, int]] = []

            while queue:
                cy, cx = queue.popleft()
                component_pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or int(label_map[ny, nx]) != label_value:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))

            values = [pred_np[py, px] for py, px in component_pixels]
            mean_value = float(np.mean(values)) if values else float(pred_np[y, x])
            for py, px in component_pixels:
                output[py, px] = mean_value

    Image.fromarray(np.clip(np.round(output), 0, 255).astype(np.uint8), mode="L").save(save_path)


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
    image_key = str(item.get("image_relpath") or item.get("image_name") or "").strip()
    if not image_key:
        manifest.append(item)
        return
    for idx, existing in enumerate(manifest):
        existing_key = str(existing.get("image_relpath") or existing.get("image_name") or "").strip()
        if existing_key == image_key:
            manifest[idx] = item
            return
    manifest.append(item)


def get_route_name(config: VariantConfig, provider: str) -> str:
    route_suffix = config.route
    for prefix in ("seedream_", "doubao_"):
        if route_suffix.startswith(prefix):
            route_suffix = route_suffix[len(prefix):]
            break
    return f"{provider}_{route_suffix}"


def build_run_signature(
    *,
    config: VariantConfig,
    runtime: Any,
    prompt_text: str,
    example_rgb: str,
    example_roughness: str,
    timeout: int,
    sleep: float,
    request_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
    size: str,
    watermark: bool,
) -> Dict[str, Any]:
    signature = {
        "provider": runtime.provider,
        "variant_id": config.variant_id.upper(),
        "variant_name": config.variant_name,
        "image_model": runtime.image_model,
        "base_url": runtime.base_url,
        "route": get_route_name(config, runtime.provider),
        "input_mode": config.input_mode,
        "prompt_version": config.prompt_version,
        "prompt_text": prompt_text,
        "segmentation_role": config.segmentation_role,
        "postprocess_mode": config.postprocess_mode,
        "timeout": timeout,
        "sleep": sleep,
        "request_retries": request_retries,
        "retry_backoff": retry_backoff,
        "retry_max_backoff": retry_max_backoff,
        "size": size,
        "output_resolution_policy": "model_native_output",
        "watermark": watermark,
    }
    if config.use_example_pair:
        signature["example_rgb"] = example_rgb
        signature["example_roughness"] = example_roughness
        signature["example_pair_policy"] = "fixed_global_exemplar_pair"
    if runtime.provider in {"gpt", "gpt2"}:
        signature["gpt_backend"] = runtime.gpt_backend
        signature["image_quality"] = runtime.image_quality
    return provider_request_record(signature, runtime.provider)

def should_skip_existing_output(
    *,
    roughness_path: Path,
) -> bool:
    return roughness_path.exists()


def generate_one_case(
    *,
    runtime: Any,
    variant_id: str,
    rgb_path: Path,
    seg_path: Optional[Path],
    save_path: Path,
    example_rgb: Optional[Path],
    example_roughness: Optional[Path],
    size: str,
    watermark: bool,
    timeout: int,
    source_prediction_path: Optional[Path] = None,
    request_retries: int = 5,
    retry_backoff: float = 45,
    retry_max_backoff: float = 90.0,
) -> Dict[str, Any]:
    config = get_variant_config(variant_id)
    example_pair = None
    if config.use_example_pair:
        if example_rgb is None or example_roughness is None:
            raise ValueError(f"{variant_id} requires both example_rgb and example_roughness")
        example_pair = (example_rgb, example_roughness)

    prompt = build_prompt(config.variant_id)
    image_paths = resolve_input_images(
        config.variant_id,
        rgb_path=rgb_path,
        seg_path=seg_path,
        example_pair=example_pair,
    )

    source_prediction = None
    save_target = save_path
    if config.variant_id == "a2":
        if seg_path is None:
            raise ValueError("a2 requires a matching segmentation image")
        source_prediction = source_prediction_path or save_path.with_name(f"{save_path.stem}_source.png")
        save_target = source_prediction

    response = run_depth_generation_with_retry(
        runtime,
        prompt,
        image_paths,
        size,
        watermark,
        getattr(runtime, "seed", 123),
        timeout,
        request_retries,
        retry_backoff,
        retry_max_backoff,
    )
    image_url = save_generated_image(runtime, response, save_target, timeout=timeout)

    if source_prediction is not None:
        assert seg_path is not None
        apply_region_fill_from_segmentation(source_prediction, seg_path, save_path)

    result: Dict[str, Any] = {
        "variant_id": config.variant_id.upper(),
        "variant_name": config.variant_name,
        "route": get_route_name(config, runtime.provider),
        "input_mode": config.input_mode,
        "prompt_text": prompt,
        "segmentation_role": config.segmentation_role,
        "postprocess_mode": config.postprocess_mode,
        "example_rgb": str(example_rgb) if example_rgb else "",
        "example_roughness": str(example_roughness) if example_roughness else "",
        "seed": getattr(runtime, "seed", 123),
        "input_order": [str(path) for path in image_paths],
        "image_url": image_url,
    }
    if source_prediction is not None:
        result["source_prediction"] = source_prediction.name
    return provider_request_record(result, runtime.provider)

def generate_roughness_map_with_seedream(
    *,
    runtime: Any,
    input_mode: str,
    rgb_path: Path,
    seg_path: Optional[Path],
    save_path: Path,
    example_rgb: Optional[Path],
    example_roughness: Optional[Path],
    size: str,
    watermark: bool,
    timeout: int,
) -> str:
    config = get_default_variant_for_input_mode(input_mode)
    result = generate_one_case(
        runtime=runtime,
        variant_id=config.variant_id,
        rgb_path=rgb_path,
        seg_path=seg_path,
        save_path=save_path,
        example_rgb=example_rgb,
        example_roughness=example_roughness,
        size=size,
        watermark=watermark,
        timeout=timeout,
    )
    return str(result["route"])

def main() -> None:
    args = parse_args()
    config = get_variant_config(args.variant)
    runtime = resolve_provider_runtime(args)
    runtime.seed = args.seed

    input_dir = Path(args.input_dir).expanduser()
    example_rgb = None
    example_roughness = None
    if config.use_example_pair:
        example_rgb, example_roughness = resolve_fixed_example_pair(args.example_rgb, args.example_roughness)
        for path_obj, name in ((example_rgb, "example_rgb"), (example_roughness, "example_roughness")):
            if not path_obj.exists() or not path_obj.is_file():
                raise FileNotFoundError(f"{name} not found: {path_obj}")
    output_dir = Path(args.output_dir).expanduser() / config.variant_id.upper()
    output_dir.mkdir(parents=True, exist_ok=True)

    roughness_dir = output_dir / "roughness"
    meta_dir = output_dir / "meta"
    output_meta_dir = meta_dir / "per_image"
    roughness_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(input_dir)
    image_paths_for_generate = image_paths[: args.max_generate] if args.max_generate > 0 else image_paths

    print(
        f"[1/3] found {len(image_paths)} RGB images; generating {len(image_paths_for_generate)} roughness maps "
        f"with provider={runtime.provider}, model={runtime.image_model}, variant={config.variant_id}, input_mode={config.input_mode}"
    )

    prompt_text = build_prompt(config.variant_id)
    run_signature = build_run_signature(
        config=config,
        runtime=runtime,
        prompt_text=prompt_text,
        example_rgb=str(example_rgb) if example_rgb else "",
        example_roughness=str(example_roughness) if example_roughness else "",
        timeout=args.timeout,
        sleep=args.sleep,
        request_retries=args.request_retries,
        retry_backoff=args.retry_backoff,
        retry_max_backoff=args.retry_max_backoff,
        size=args.size,
        watermark=args.watermark,
    )

    setup = dict(run_signature)
    (meta_dir / "setup.json").write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = meta_dir / "manifest.json"
    manifest: List[Dict[str, Any]] = load_manifest(manifest_path)

    print(f"[2/3] start generation for {len(image_paths_for_generate)} images")
    failures = 0
    for idx, rgb_path in enumerate(image_paths_for_generate, start=1):
        try:
            image_relpath = rgb_path.relative_to(input_dir)
        except ValueError:
            image_relpath = Path(rgb_path.name)
        roughness_relpath = image_relpath.with_name(f"{image_relpath.stem}_roughness.png")

        item: Dict[str, Any] = {
            "image_name": rgb_path.name,
            "image_relpath": image_relpath.as_posix(),
            "provider": runtime.provider,
            "image_model": runtime.image_model,
        }
        try:
            print(f"  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name}")

            roughness_path = roughness_dir / roughness_relpath
            output_meta_path = output_meta_dir / roughness_relpath.with_suffix(".json")
            roughness_path.parent.mkdir(parents=True, exist_ok=True)
            output_meta_path.parent.mkdir(parents=True, exist_ok=True)
            item["prompt_version"] = config.prompt_version
            item["input_mode"] = config.input_mode
            item["variant_id"] = config.variant_id
            if config.use_example_pair:
                assert example_rgb is not None and example_roughness is not None
                item["example_rgb"] = example_rgb.name
                item["example_roughness"] = example_roughness.name
            if args.skip_existing and should_skip_existing_output(
                roughness_path=roughness_path,
            ):
                item["skipped"] = True
                item["skip_reason"] = "output_exists"
                item["roughness_output"] = roughness_relpath.as_posix()
                print(f"[skip] {rgb_path.name}: output exists")
            else:
                generated = generate_one_case(
                    runtime=runtime,
                    variant_id=config.variant_id,
                    rgb_path=rgb_path,
                    seg_path=(None if config.input_mode != "rgb_plus_seg" else find_matching_seg(rgb_path, (Path(args.seg_dir) / rgb_path.parent.relative_to(input_dir)) if args.seg_dir else rgb_path.parent)),
                    save_path=roughness_path,
                    example_rgb=example_rgb,
                    example_roughness=example_roughness,
                    size=args.size,
                    watermark=args.watermark,
                    timeout=args.timeout,
                    request_retries=args.request_retries,
                    retry_backoff=args.retry_backoff,
                    retry_max_backoff=args.retry_max_backoff,
                )
                item["roughness_mode"] = generated["route"]
                item["input_order"] = generated["input_order"]
                item["seed"] = args.seed
                item["roughness_output"] = roughness_relpath.as_posix()
                if generated.get("image_url"):
                    item["image_url"] = generated["image_url"]
                output_meta: Dict[str, Any] = provider_request_record({
                    "image_name": rgb_path.name,
                    "image_relpath": image_relpath.as_posix(),
                    "roughness_output": roughness_relpath.as_posix(),
                    "variant_id": config.variant_id.upper(),
                    "variant_name": config.variant_name,
                    "input_mode": config.input_mode,
                    "provider": runtime.provider,
                    "image_model": runtime.image_model,
                    "run_signature": run_signature,
                }, runtime.provider)
                if config.use_example_pair:
                    output_meta["example_rgb"] = str(example_rgb)
                    output_meta["example_roughness"] = str(example_roughness)
                if generated.get("source_prediction"):
                    output_meta["source_prediction"] = generated["source_prediction"]
                if generated.get("image_url"):
                    output_meta["image_url"] = generated["image_url"]
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

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
