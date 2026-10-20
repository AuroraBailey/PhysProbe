#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
try:
    from .mask_policy import resolve_required_mask
except ImportError:
    from mask_policy import resolve_required_mask


import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

try:
    import torch
except Exception:
    torch = None

try:
    import lpips
except Exception:
    lpips = None


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
ROUGHNESS_BIN_EDGES = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.000001], dtype=np.float32)
ROUGHNESS_BIN_NAMES = [
    "bin_0_02",
    "bin_02_04",
    "bin_04_06",
    "bin_06_08",
    "bin_08_10",
]


@dataclass
class ImageMetrics:
    name: str
    split: str
    scene_id: str
    width: int
    height: int
    valid_pixels: int
    mae: float
    rmse: float
    psnr: float
    ssim: float
    lpips: Optional[float]
    bin_bmae: float
    region_macro_mae: float
    region_macro_rmse: float
    region_count: int
    bin_0_02_mae: float
    bin_02_04_mae: float
    bin_04_06_mae: float
    bin_06_08_mae: float
    bin_08_10_mae: float
    bin_0_02_pixels: int
    bin_02_04_pixels: int
    bin_04_06_pixels: int
    bin_06_08_pixels: int
    bin_08_10_pixels: int


@dataclass
class SceneMetrics:
    scene_id: str
    split: str
    image_count: int
    valid_pixels: int
    mae: float
    rmse: float
    ssim_mean: float
    lpips_mean: float
    bin_bmae: float
    region_macro_mae: float
    region_macro_rmse: float
    region_count: int
    bin_0_02_mae: float
    bin_02_04_mae: float
    bin_04_06_mae: float
    bin_06_08_mae: float
    bin_08_10_mae: float
    bin_0_02_pixels: int
    bin_02_04_pixels: int
    bin_04_06_pixels: int
    bin_06_08_pixels: int
    bin_08_10_pixels: int


@dataclass
class PairDiagnostics:
    valid_pixels: int
    abs_sum: float
    sq_sum: float
    bin_abs_sums: np.ndarray
    bin_sq_sums: np.ndarray
    bin_counts: np.ndarray
    region_maes: List[float]
    region_rmses: List[float]
    ssim: float
    lpips: Optional[float]


class SceneAccumulator:
    def __init__(self, scene_id: str, split: str) -> None:
        self.scene_id = scene_id
        self.split = split
        self.image_count = 0
        self.valid_pixels = 0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.bin_abs_sums = np.zeros(5, dtype=np.float64)
        self.bin_sq_sums = np.zeros(5, dtype=np.float64)
        self.bin_counts = np.zeros(5, dtype=np.int64)
        self.region_maes: List[float] = []
        self.region_rmses: List[float] = []
        self.image_rmse_values: List[float] = []
        self.ssim_values: List[float] = []
        self.lpips_values: List[float] = []

    def update(self, diag: PairDiagnostics) -> None:
        self.image_count += 1
        self.valid_pixels += diag.valid_pixels
        self.abs_sum += diag.abs_sum
        self.sq_sum += diag.sq_sum
        if diag.valid_pixels > 0:
            self.image_rmse_values.append(float(math.sqrt(diag.sq_sum / diag.valid_pixels)))
        self.bin_abs_sums += diag.bin_abs_sums
        self.bin_sq_sums += diag.bin_sq_sums
        self.bin_counts += diag.bin_counts
        self.region_maes.extend(diag.region_maes)
        self.region_rmses.extend(diag.region_rmses)
        if np.isfinite(diag.ssim):
            self.ssim_values.append(float(diag.ssim))
        if diag.lpips is not None and np.isfinite(diag.lpips):
            self.lpips_values.append(float(diag.lpips))

    def finalize(self) -> SceneMetrics:
        mae = safe_div(self.abs_sum, self.valid_pixels)
        mse = safe_div(self.sq_sum, self.valid_pixels)
        rmse = float(math.sqrt(mse)) if np.isfinite(mse) else float("nan")
        bin_maes = per_bin_maes(self.bin_abs_sums, self.bin_counts)
        return SceneMetrics(
            scene_id=self.scene_id,
            split=self.split,
            image_count=self.image_count,
            valid_pixels=self.valid_pixels,
            mae=mae,
            rmse=rmse,
            ssim_mean=mean_valid(self.ssim_values),
            lpips_mean=mean_valid(self.lpips_values),
            bin_bmae=mean_valid(bin_maes),
            region_macro_mae=mean_valid(self.region_maes),
            region_macro_rmse=mean_valid(self.region_rmses),
            region_count=len(self.region_maes),
            bin_0_02_mae=bin_maes[0],
            bin_02_04_mae=bin_maes[1],
            bin_04_06_mae=bin_maes[2],
            bin_06_08_mae=bin_maes[3],
            bin_08_10_mae=bin_maes[4],
            bin_0_02_pixels=int(self.bin_counts[0]),
            bin_02_04_pixels=int(self.bin_counts[1]),
            bin_04_06_pixels=int(self.bin_counts[2]),
            bin_06_08_pixels=int(self.bin_counts[3]),
            bin_08_10_pixels=int(self.bin_counts[4]),
        )


class LPIPSEvaluator:
    def __init__(self, device: str = "cpu", net: str = "alex") -> None:
        self.available = torch is not None and lpips is not None
        self.device = device
        self.net = net
        self.model = None
        self.unavailable_reason: Optional[str] = None
        if not self.available:
            missing = []
            if torch is None:
                missing.append("torch")
            if lpips is None:
                missing.append("lpips")
            self.unavailable_reason = f"missing optional dependency: {', '.join(missing)}"
            return
        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()

    def __call__(self, gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Optional[float]:
        if not self.available or self.model is None:
            return None

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        gt_crop = gt[y0:y1, x0:x1].copy()
        pred_crop = pred[y0:y1, x0:x1].copy()
        mask_crop = mask[y0:y1, x0:x1]

        invalid = mask_crop == 0
        pred_crop[invalid] = gt_crop[invalid]

        gt_rgb = np.repeat(gt_crop[..., None], 3, axis=2)
        pred_rgb = np.repeat(pred_crop[..., None], 3, axis=2)

        gt_t = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).float()
        pred_t = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).float()
        gt_t = gt_t * 2.0 - 1.0
        pred_t = pred_t * 2.0 - 1.0

        with torch.no_grad():
            value = self.model(gt_t.to(self.device), pred_t.to(self.device))
        return float(value.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predicted roughness maps against aligned ground-truth roughness, "
            "with optional masks, bin-balanced diagnostics, optional region-macro errors, "
            "and optional constant / calibration-mean baselines."
        )
    )

    parser.add_argument("--gt_path", type=str, help="Single GT roughness path.")
    parser.add_argument("--pred_path", type=str, help="Single predicted roughness path.")
    parser.add_argument("--mask_path", type=str, help="Optional single valid-mask path.")
    parser.add_argument("--region_path", type=str, help="Optional single region-label path.")

    parser.add_argument("--gt_dir", type=str, help="GT roughness directory.")
    parser.add_argument("--pred_dir", type=str, help="Prediction directory.")
    parser.add_argument("--mask_dir", type=str, help="Optional valid-mask directory.")
    parser.add_argument("--region_dir", type=str, help="Optional region-label directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save CSV/JSON summaries.")

    parser.add_argument(
        "--pair_mode",
        type=str,
        default="stem",
        choices=["stem", "name"],
        help="How to match directory files: by stem or exact relative filename.",
    )
    parser.add_argument("--pred_suffix_to_strip", type=str, default="_roughness")
    parser.add_argument("--gt_suffix_to_strip", type=str, default="_roughness")
    parser.add_argument("--mask_suffix", type=str, default="_mask")
    parser.add_argument("--region_suffix", type=str, default="_region")
    parser.add_argument("--pred_path_component_to_strip", type=str)
    parser.add_argument("--gt_path_component_to_strip", type=str)
    parser.add_argument("--mask_path_component_to_strip", type=str)
    parser.add_argument("--region_path_component_to_strip", type=str)

    parser.add_argument("--save_resized_preds", action="store_true", help="Save resized predictions for inspection.")
    parser.add_argument("--recursive", dest="recursive", action="store_true", help="Recursively search directories.")
    parser.add_argument("--no_recursive", dest="recursive", action="store_false", help="Disable recursive search.")
    parser.set_defaults(recursive=True)

    parser.add_argument(
        "--resize_interp",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        help="Interpolation used when resizing predictions to GT resolution.",
    )
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Threshold in [0,1] for binarizing masks.")
    parser.add_argument(
        "--region_source",
        type=str,
        default="auto",
        choices=["none", "auto", "dir", "object_segmentation"],
        help=(
            "Region-label source for region-macro metrics. "
            "'auto' first uses --region_dir when provided, then tries OpenRooms-style ObjectSegmentation."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="none",
        choices=["none", "constant", "calibration_mean"],
        help="Optional synthetic baseline replacing prediction images.",
    )
    parser.add_argument("--constant_value", type=float, default=None, help="Constant prediction value in [0,1].")
    parser.add_argument(
        "--calibration_gt_dir",
        type=str,
        help="GT roughness directory used to estimate the calibration-set mean.",
    )
    parser.add_argument(
        "--calibration_mask_dir",
        type=str,
        help="Optional valid-mask directory aligned with --calibration_gt_dir.",
    )
    parser.add_argument("--bootstrap_iters", type=int, default=0, help="Optional scene-bootstrap iterations.")
    parser.add_argument("--bootstrap_seed", type=int, default=0, help="Random seed for scene bootstrap.")

    parser.add_argument(
        "--compute_lpips",
        dest="compute_lpips",
        action="store_true",
        help="Compute LPIPS if torch and lpips are available (default).",
    )
    parser.add_argument(
        "--no_compute_lpips",
        dest="compute_lpips",
        action="store_false",
        help="Disable LPIPS computation.",
    )
    parser.set_defaults(compute_lpips=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
        help="Device used for LPIPS.",
    )
    parser.add_argument(
        "--lpips_net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="Backbone used by LPIPS.",
    )
    parser.add_argument("--disable_mask", action="store_true", help="Explicitly evaluate without a valid-pixel mask.")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    single_mode = bool(args.gt_path and (args.pred_path or args.baseline != "none"))
    dir_mode = bool(args.gt_dir and (args.pred_dir or args.baseline != "none"))
    if single_mode == dir_mode:
        raise ValueError(
            "Use either single-pair mode (--gt_path plus --pred_path or synthetic baseline) "
            "or directory mode (--gt_dir plus --pred_dir or synthetic baseline)."
        )

    if single_mode:
        gt_path = Path(args.gt_path)
        if not gt_path.is_file():
            raise FileNotFoundError(f"GT file not found: {gt_path}")
        if args.baseline == "none":
            pred_path = Path(args.pred_path)
            if not pred_path.is_file():
                raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        if args.mask_path:
            mask_path = Path(args.mask_path)
            if not mask_path.is_file():
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
        if args.region_path:
            region_path = Path(args.region_path)
            if not region_path.is_file():
                raise FileNotFoundError(f"Region file not found: {region_path}")
    else:
        gt_dir = Path(args.gt_dir)
        if not gt_dir.is_dir():
            raise FileNotFoundError(f"GT directory not found: {gt_dir}")
        if args.baseline == "none":
            pred_dir = Path(args.pred_dir)
            if not pred_dir.is_dir():
                raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
        if args.mask_dir and not Path(args.mask_dir).is_dir():
            raise FileNotFoundError(f"Mask directory not found: {args.mask_dir}")
        if args.region_dir and not Path(args.region_dir).is_dir():
            raise FileNotFoundError(f"Region directory not found: {args.region_dir}")

    if args.baseline == "constant" and args.constant_value is None:
        raise ValueError("--baseline constant requires --constant_value")
    if args.baseline == "calibration_mean" and not args.calibration_gt_dir:
        raise ValueError("--baseline calibration_mean requires --calibration_gt_dir")


def pil_interp_mode(name: str) -> int:
    name = name.lower()
    if name == "nearest":
        return Image.Resampling.NEAREST
    if name == "bilinear":
        return Image.Resampling.BILINEAR
    if name == "bicubic":
        return Image.Resampling.BICUBIC
    if name == "lanczos":
        return Image.Resampling.LANCZOS
    raise ValueError(f"Unknown interpolation: {name}")


def list_images(directory: Path, recursive: bool) -> List[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted([p for p in iterator if p.is_file() and p.suffix.lower() in IMG_EXTS])


def normalize_match_key(name: str, suffix_to_strip: str) -> str:
    stem = Path(name).stem
    if suffix_to_strip and stem.endswith(suffix_to_strip):
        stem = stem[: -len(suffix_to_strip)]
    return stem


def load_gray(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("L")


def load_label(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img)


def normalize_region_labels(label: np.ndarray) -> np.ndarray:
    if label.ndim == 2:
        return label.astype(np.int64, copy=False)
    if label.ndim == 3:
        label_int = label.astype(np.int64, copy=False)
        if label_int.shape[2] == 1:
            return label_int[..., 0]
        # Pack multi-channel labels into a stable integer id per pixel.
        out = np.zeros(label_int.shape[:2], dtype=np.int64)
        for channel_idx in range(label_int.shape[2]):
            out = (out << 8) + label_int[..., channel_idx]
        return out
    raise ValueError(f"Unsupported region label shape: {label.shape}")


def image_to_float_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def split_from_name(name: str) -> str:
    for part in Path(name).parts:
        lower = part.lower()
        if "mainaxis" in lower:
            return "mainaxis"
        if "stresstest" in lower:
            return "stresstest"
    return "unknown"


def scene_id_from_name(name: str) -> str:
    parts = Path(name).parts
    for idx, part in enumerate(parts):
        lower = part.lower()
        if lower in {"mainaxis", "stresstest"} and idx + 2 < len(parts):
            return "/".join(parts[idx : idx + 3])
    if len(parts) >= 3:
        return "/".join(parts[:3])
    if parts:
        return parts[0]
    return "single"


def load_mask(path: Path, size: Tuple[int, int], threshold: float) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != size:
            img = img.resize(size, Image.Resampling.NEAREST)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return (arr >= threshold).astype(np.uint8)


def resize_label_to_size(label: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    h, w = label.shape[:2]
    if (w, h) == size:
        return normalize_region_labels(label)
    label_img = Image.fromarray(label)
    label_img = label_img.resize(size, Image.Resampling.NEAREST)
    return normalize_region_labels(np.asarray(label_img))


def is_same_existing_path(path_a: Path, path_b: Path) -> bool:
    try:
        return path_a.resolve() == path_b.resolve()
    except Exception:
        return path_a == path_b


def strip_path_component(rel: Path, component_to_strip: str) -> Path:
    if not component_to_strip:
        return rel
    return Path(*[part for part in rel.parts if part != component_to_strip])


def make_key(
    path: Path,
    root: Path,
    suffix_to_strip: str,
    path_component_to_strip: str,
    pair_mode: str,
) -> str:
    rel = path.relative_to(root)
    rel = strip_path_component(rel, path_component_to_strip)
    if pair_mode == "name":
        return rel.as_posix()
    stem = normalize_match_key(path.name, suffix_to_strip)
    if rel.parent == Path("."):
        return stem
    return (rel.parent / stem).as_posix()


def resolve_aux_path(
    gt_path: Path,
    aux_dir: Optional[Path],
    aux_map: Optional[Dict[str, Path]],
    pair_mode: str,
    gt_dir: Path,
    pair_key: str,
) -> Optional[Path]:
    if aux_dir is None:
        return None
    rel = gt_path.relative_to(gt_dir)
    direct_candidate = aux_dir / rel
    if direct_candidate.exists() and not is_same_existing_path(direct_candidate, gt_path):
        return direct_candidate
    for ext in IMG_EXTS:
        probe = direct_candidate.with_suffix(ext)
        if probe.exists() and not is_same_existing_path(probe, gt_path):
            return probe
    if aux_map:
        if pair_mode == "name":
            return aux_map.get(rel.as_posix()) or aux_map.get(pair_key)
        return aux_map.get(pair_key)
    return None


def build_pairs_from_dirs(
    gt_dir: Path,
    pred_dir: Optional[Path],
    mask_dir: Optional[Path],
    region_dir: Optional[Path],
    pair_mode: str,
    pred_suffix_to_strip: str,
    gt_suffix_to_strip: str,
    mask_suffix: str,
    region_suffix: str,
    pred_path_component_to_strip: str,
    gt_path_component_to_strip: str,
    mask_path_component_to_strip: str,
    region_path_component_to_strip: str,
    recursive: bool,
    synthetic_baseline: bool,
) -> List[Tuple[Path, Optional[Path], Optional[Path], Optional[Path]]]:
    gt_files = list_images(gt_dir, recursive)
    if synthetic_baseline:
        pred_files: List[Path] = []
    else:
        assert pred_dir is not None
        pred_files = list_images(pred_dir, recursive)
    mask_files = list_images(mask_dir, recursive) if mask_dir is not None else []
    region_files = list_images(region_dir, recursive) if region_dir is not None else []

    gt_map = {
        make_key(p, gt_dir, gt_suffix_to_strip, gt_path_component_to_strip, pair_mode): p for p in gt_files
    }
    pred_map = (
        {
            make_key(p, pred_dir, pred_suffix_to_strip, pred_path_component_to_strip, pair_mode): p
            for p in pred_files
        }
        if pred_dir is not None
        else {}
    )
    mask_map = (
        {
            make_key(p, mask_dir, mask_suffix, mask_path_component_to_strip, pair_mode): p
            for p in mask_files
        }
        if mask_files
        else {}
    )
    region_map = (
        {
            make_key(p, region_dir, region_suffix, region_path_component_to_strip, pair_mode): p
            for p in region_files
        }
        if region_files
        else {}
    )

    keys = sorted(gt_map.keys() if synthetic_baseline else (set(gt_map.keys()) & set(pred_map.keys())))
    pairs: List[Tuple[Path, Optional[Path], Optional[Path], Optional[Path]]] = []
    for key in keys:
        gt_path = gt_map[key]
        pred_path = None if synthetic_baseline else pred_map[key]
        mask_path = resolve_aux_path(
            gt_path=gt_path,
            aux_dir=mask_dir,
            aux_map=mask_map if mask_map else None,
            pair_mode=pair_mode,
            gt_dir=gt_dir,
            pair_key=key,
        )
        region_path = resolve_aux_path(
            gt_path=gt_path,
            aux_dir=region_dir,
            aux_map=region_map if region_map else None,
            pair_mode=pair_mode,
            gt_dir=gt_dir,
            pair_key=key,
        )
        pairs.append((gt_path, pred_path, mask_path, region_path))
    return pairs


def safe_div(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num / den)


def mean_valid(values: Sequence[Optional[float]]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def median_valid(values: Sequence[Optional[float]]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.median(vals))


def std_valid(values: Sequence[Optional[float]]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.std(vals))


def masked_values(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = mask > 0
    return gt[valid], pred[valid]


def masked_mae(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    gt_v, pred_v = masked_values(gt, pred, mask)
    if gt_v.size == 0:
        return float("nan")
    return float(np.mean(np.abs(gt_v - pred_v)))


def masked_mse(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    gt_v, pred_v = masked_values(gt, pred, mask)
    if gt_v.size == 0:
        return float("nan")
    return float(np.mean((gt_v - pred_v) ** 2))


def masked_rmse(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    mse = masked_mse(gt, pred, mask)
    if not np.isfinite(mse):
        return float("nan")
    return float(math.sqrt(mse))


def masked_psnr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> float:
    mse = masked_mse(gt, pred, mask)
    if not np.isfinite(mse):
        return float("nan")
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10((data_range ** 2) / mse))


def masked_ssim(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if skimage_ssim is None:
        return float("nan")

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return float("nan")

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    gt_crop = gt[y0:y1, x0:x1].copy()
    pred_crop = pred[y0:y1, x0:x1].copy()
    mask_crop = mask[y0:y1, x0:x1]

    invalid = mask_crop == 0
    pred_crop[invalid] = gt_crop[invalid]

    if gt_crop.shape[0] < 7 or gt_crop.shape[1] < 7:
        return float("nan")
    return float(skimage_ssim(gt_crop, pred_crop, data_range=1.0))


def per_bin_maes(bin_abs_sums: np.ndarray, bin_counts: np.ndarray) -> List[float]:
    return [safe_div(float(bin_abs_sums[idx]), int(bin_counts[idx])) for idx in range(5)]


def compute_bin_statistics(gt_valid: np.ndarray, pred_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff = np.abs(gt_valid - pred_valid)
    diff_sq = (gt_valid - pred_valid) ** 2
    bin_abs_sums = np.zeros(5, dtype=np.float64)
    bin_sq_sums = np.zeros(5, dtype=np.float64)
    bin_counts = np.zeros(5, dtype=np.int64)
    bin_ids = np.digitize(gt_valid, ROUGHNESS_BIN_EDGES[1:-1], right=False)
    for idx in range(5):
        mask = bin_ids == idx
        bin_counts[idx] = int(mask.sum())
        if bin_counts[idx] > 0:
            bin_abs_sums[idx] = float(diff[mask].sum())
            bin_sq_sums[idx] = float(diff_sq[mask].sum())
    return bin_abs_sums, bin_sq_sums, bin_counts


def compute_region_macro_errors(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, region_labels: Optional[np.ndarray]) -> Tuple[float, float, int, List[float], List[float]]:
    if region_labels is None:
        return float("nan"), float("nan"), 0, [], []

    labels = resize_label_to_size(region_labels, (gt.shape[1], gt.shape[0]))
    valid = mask > 0
    labels_valid = labels[valid]
    gt_valid = gt[valid]
    pred_valid = pred[valid]
    if labels_valid.size == 0:
        return float("nan"), float("nan"), 0, [], []

    region_maes: List[float] = []
    region_rmses: List[float] = []
    for label_value in np.unique(labels_valid):
        region_mask = labels_valid == label_value
        if not region_mask.any():
            continue
        region_diff = gt_valid[region_mask] - pred_valid[region_mask]
        region_maes.append(float(np.mean(np.abs(region_diff))))
        region_rmses.append(float(np.sqrt(np.mean(region_diff ** 2))))

    return mean_valid(region_maes), mean_valid(region_rmses), len(region_maes), region_maes, region_rmses


def resolve_object_segmentation_path(gt_path: Path) -> Optional[Path]:
    parts = list(gt_path.parts)
    try:
        roughness_idx = parts.index("Roughness")
    except ValueError:
        return None
    if roughness_idx + 2 >= len(parts):
        return None
    camera_dir = parts[roughness_idx + 1]
    camera_suffix = camera_dir.split("_")[-1]
    parts[roughness_idx] = "ObjectSegmentation"
    parts[-1] = f"ObjectSegmentation_0_0_0001_{camera_suffix}.png"
    candidate = Path(*parts)
    return candidate if candidate.exists() else None


def resolve_region_path(
    gt_path: Path,
    explicit_region_path: Optional[Path],
    region_source: str,
) -> Optional[Path]:
    if region_source == "none":
        return None
    if explicit_region_path is not None and explicit_region_path.exists():
        return explicit_region_path
    if region_source in {"auto", "object_segmentation"}:
        return resolve_object_segmentation_path(gt_path)
    return None


def make_baseline_prediction(
    gt_img: Image.Image,
    baseline: str,
    constant_value: Optional[float],
) -> Image.Image:
    if baseline == "none":
        raise ValueError("make_baseline_prediction should not be called when baseline=none")
    value = float(np.clip(constant_value if constant_value is not None else 0.0, 0.0, 1.0))
    arr = np.full((gt_img.size[1], gt_img.size[0]), int(round(value * 255.0)), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def compute_calibration_mean(
    gt_dir: Path,
    mask_dir: Optional[Path],
    args: argparse.Namespace,
) -> float:
    gt_files = list_images(gt_dir, args.recursive)
    mask_files = list_images(mask_dir, args.recursive) if mask_dir is not None else []
    mask_map = (
        {
            make_key(p, mask_dir, args.mask_suffix, args.mask_path_component_to_strip, args.pair_mode): p
            for p in mask_files
        }
        if mask_files
        else {}
    )
    total_sum = 0.0
    total_count = 0
    for gt_path in gt_files:
        gt_img = load_gray(gt_path)
        gt = image_to_float_np(gt_img)
        mask_path = resolve_aux_path(
            gt_path=gt_path,
            aux_dir=mask_dir,
            aux_map=mask_map if mask_map else None,
            pair_mode=args.pair_mode,
            gt_dir=gt_dir,
            pair_key=make_key(gt_path, gt_dir, args.gt_suffix_to_strip, args.gt_path_component_to_strip, args.pair_mode),
        )
        if mask_path is not None and mask_path.exists():
            mask = load_mask(mask_path, gt_img.size, args.mask_threshold)
        else:
            mask = np.ones_like(gt, dtype=np.uint8)
        gt_valid = gt[mask > 0]
        total_sum += float(gt_valid.sum())
        total_count += int(gt_valid.size)
    if total_count <= 0:
        raise RuntimeError(f"Could not compute calibration mean from {gt_dir}")
    mean_value = float(total_sum / total_count)
    print(f"Calibration-set roughness mean = {mean_value:.6f} over {total_count} valid pixels.", file=sys.stderr)
    return mean_value


def evaluate_pair(
    gt_path: Path,
    pred_path: Optional[Path],
    mask_path: Optional[Path],
    region_path: Optional[Path],
    args: argparse.Namespace,
    lpips_eval: Optional[LPIPSEvaluator],
    resized_pred_dir: Optional[Path],
    name: str,
    baseline_value: Optional[float],
) -> Tuple[ImageMetrics, PairDiagnostics]:
    mask_path = resolve_required_mask(gt_path, mask_path, disabled=getattr(args, "disable_mask", False))
    gt_img = load_gray(gt_path)
    if args.baseline == "none":
        assert pred_path is not None
        pred_img = load_gray(pred_path)
        pred_label = str(pred_path)
    else:
        pred_img = make_baseline_prediction(gt_img, args.baseline, baseline_value)
        pred_label = f"baseline://{args.baseline}/{baseline_value:.6f}" if baseline_value is not None else f"baseline://{args.baseline}"
    if pred_img.size != gt_img.size:
        pred_img = pred_img.resize(gt_img.size, pil_interp_mode(args.resize_interp))

    if resized_pred_dir is not None:
        resized_pred_dir.mkdir(parents=True, exist_ok=True)
        target_name = (pred_path.name if pred_path is not None else Path(name).name) if name else "baseline.png"
        pred_img.save(resized_pred_dir / target_name)

    gt = image_to_float_np(gt_img)
    pred = image_to_float_np(pred_img)
    if mask_path is not None and mask_path.exists():
        mask = load_mask(mask_path, gt_img.size, args.mask_threshold)
    else:
        mask = np.ones_like(gt, dtype=np.uint8)

    resolved_region_path = resolve_region_path(gt_path, region_path, args.region_source)
    region_labels = load_label(resolved_region_path) if resolved_region_path is not None and resolved_region_path.exists() else None

    gt_valid, pred_valid = masked_values(gt, pred, mask)
    valid_pixels = int(gt_valid.size)
    abs_sum = float(np.abs(gt_valid - pred_valid).sum()) if valid_pixels > 0 else 0.0
    sq_sum = float(((gt_valid - pred_valid) ** 2).sum()) if valid_pixels > 0 else 0.0

    bin_abs_sums, bin_sq_sums, bin_counts = compute_bin_statistics(gt_valid, pred_valid)
    bin_maes = per_bin_maes(bin_abs_sums, bin_counts)
    region_macro_mae, region_macro_rmse, region_count, region_maes, region_rmses = compute_region_macro_errors(
        gt=gt,
        pred=pred,
        mask=mask,
        region_labels=region_labels,
    )
    ssim = masked_ssim(gt, pred, mask)
    lpips_value = lpips_eval(gt, pred, mask) if lpips_eval is not None else None

    metrics = ImageMetrics(
        name=name,
        split=split_from_name(name),
        scene_id=scene_id_from_name(name),
        width=gt_img.size[0],
        height=gt_img.size[1],
        valid_pixels=valid_pixels,
        mae=masked_mae(gt, pred, mask),
        rmse=masked_rmse(gt, pred, mask),
        psnr=masked_psnr(gt, pred, mask),
        ssim=ssim,
        lpips=lpips_value,
        bin_bmae=mean_valid(bin_maes),
        region_macro_mae=region_macro_mae,
        region_macro_rmse=region_macro_rmse,
        region_count=region_count,
        bin_0_02_mae=bin_maes[0],
        bin_02_04_mae=bin_maes[1],
        bin_04_06_mae=bin_maes[2],
        bin_06_08_mae=bin_maes[3],
        bin_08_10_mae=bin_maes[4],
        bin_0_02_pixels=int(bin_counts[0]),
        bin_02_04_pixels=int(bin_counts[1]),
        bin_04_06_pixels=int(bin_counts[2]),
        bin_06_08_pixels=int(bin_counts[3]),
        bin_08_10_pixels=int(bin_counts[4]),
    )
    return metrics, PairDiagnostics(
        valid_pixels=valid_pixels,
        abs_sum=abs_sum,
        sq_sum=sq_sum,
        bin_abs_sums=bin_abs_sums,
        bin_sq_sums=bin_sq_sums,
        bin_counts=bin_counts,
        region_maes=region_maes,
        region_rmses=region_rmses,
        ssim=ssim,
        lpips=lpips_value,
    )


def summarize_image_metrics(metrics: Sequence[ImageMetrics]) -> Dict[str, float]:
    return {
        "count": len(metrics),
        "mae_mean": mean_valid([m.mae for m in metrics]),
        "rmse_mean": mean_valid([m.rmse for m in metrics]),
        "psnr_mean": mean_valid([m.psnr for m in metrics]),
        "ssim_mean": mean_valid([m.ssim for m in metrics]),
        "lpips_mean": mean_valid([m.lpips for m in metrics]),
        "mae_median": median_valid([m.mae for m in metrics]),
        "rmse_median": median_valid([m.rmse for m in metrics]),
        "psnr_median": median_valid([m.psnr for m in metrics]),
        "ssim_median": median_valid([m.ssim for m in metrics]),
        "lpips_median": median_valid([m.lpips for m in metrics]),
        "mae_std": std_valid([m.mae for m in metrics]),
        "rmse_std": std_valid([m.rmse for m in metrics]),
        "psnr_std": std_valid([m.psnr for m in metrics]),
        "ssim_std": std_valid([m.ssim for m in metrics]),
        "lpips_std": std_valid([m.lpips for m in metrics]),
        "bin_bmae_image_mean": mean_valid([m.bin_bmae for m in metrics]),
        "region_macro_mae_image_mean": mean_valid([m.region_macro_mae for m in metrics]),
        "region_macro_rmse_image_mean": mean_valid([m.region_macro_rmse for m in metrics]),
    }




def aggregate_bin_summary(diags: Sequence[PairDiagnostics], scene_metrics: Sequence[SceneMetrics]) -> Dict[str, object]:
    total_abs = np.zeros(5, dtype=np.float64)
    total_counts = np.zeros(5, dtype=np.int64)
    for diag in diags:
        total_abs += diag.bin_abs_sums
        total_counts += diag.bin_counts
    per_bin = per_bin_maes(total_abs, total_counts)
    scene_support = {
        ROUGHNESS_BIN_NAMES[idx]: int(sum(1 for s in scene_metrics if getattr(s, f"{ROUGHNESS_BIN_NAMES[idx]}_pixels") > 0))
        for idx in range(5)
    }
    result: Dict[str, object] = {
        "bin_bmae_global": mean_valid(per_bin),
        "scene_support": scene_support,
        "pixel_support": {ROUGHNESS_BIN_NAMES[idx]: int(total_counts[idx]) for idx in range(5)},
    }
    for idx, name in enumerate(ROUGHNESS_BIN_NAMES):
        result[f"{name}_mae"] = per_bin[idx]
    return result


def aggregate_region_summary(diags: Sequence[PairDiagnostics]) -> Dict[str, object]:
    region_maes: List[float] = []
    region_rmses: List[float] = []
    for diag in diags:
        region_maes.extend(diag.region_maes)
        region_rmses.extend(diag.region_rmses)
    return {
        "region_count_total": len(region_maes),
        "region_macro_mae_global": mean_valid(region_maes),
        "region_macro_rmse_global": mean_valid(region_rmses),
    }


def summarize_by_split(
    image_metrics: Sequence[ImageMetrics],
    scene_metrics: Sequence[SceneMetrics],
    scene_accs: Dict[str, SceneAccumulator],
) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    for split_name in ("mainaxis", "stresstest"):
        split_images = [m for m in image_metrics if m.split == split_name]
        if not split_images:
            continue
        split_scene_metrics = [m for m in scene_metrics if m.split == split_name]
        split_accs = [acc for acc in scene_accs.values() if acc.split == split_name]
        split_diags = [
            PairDiagnostics(
                valid_pixels=acc.valid_pixels,
                abs_sum=acc.abs_sum,
                sq_sum=acc.sq_sum,
                bin_abs_sums=acc.bin_abs_sums,
                bin_sq_sums=acc.bin_sq_sums,
                bin_counts=acc.bin_counts,
                region_maes=acc.region_maes,
                region_rmses=acc.region_rmses,
                ssim=float("nan"),
                lpips=None,
            )
            for acc in split_accs
        ]
        result[split_name] = {
            "image_level": summarize_image_metrics(split_images),
            "bin_summary": aggregate_bin_summary(split_diags, split_scene_metrics),
            "region_summary": aggregate_region_summary(split_diags),
        }
    return result


def bootstrap_summary(scene_accs: Sequence[SceneAccumulator], iters: int, seed: int) -> Dict[str, object]:
    if iters <= 0 or not scene_accs:
        return {}
    rng = np.random.default_rng(seed)
    metric_names = [
        "rmse_image_macro",
        "ssim_image_macro",
        "bin_bmae_global",
        "region_macro_mae_global",
    ]
    samples: Dict[str, List[float]] = {name: [] for name in metric_names}
    n = len(scene_accs)
    for _ in range(iters):
        indices = rng.integers(0, n, size=n)
        sampled_accs = [scene_accs[idx] for idx in indices]
        sampled_scene_metrics = [acc.finalize() for acc in sampled_accs]
        sampled_diags = [
            PairDiagnostics(
                valid_pixels=acc.valid_pixels,
                abs_sum=acc.abs_sum,
                sq_sum=acc.sq_sum,
                bin_abs_sums=acc.bin_abs_sums,
                bin_sq_sums=acc.bin_sq_sums,
                bin_counts=acc.bin_counts,
                region_maes=acc.region_maes,
                region_rmses=acc.region_rmses,
                ssim=float("nan"),
                lpips=None,
            )
            for acc in sampled_accs
        ]
        bin_summary = aggregate_bin_summary(sampled_diags, sampled_scene_metrics)
        region_summary = aggregate_region_summary(sampled_diags)
        values = {
            "rmse_image_macro": mean_valid([value for acc in sampled_accs for value in acc.image_rmse_values]),
            "ssim_image_macro": mean_valid([value for acc in sampled_accs for value in acc.ssim_values]),
            "bin_bmae_global": bin_summary["bin_bmae_global"],
            "region_macro_mae_global": region_summary["region_macro_mae_global"],
        }
        for key, value in values.items():
            if np.isfinite(value):
                samples[key].append(float(value))

    result: Dict[str, object] = {}
    for key, vals in samples.items():
        if not vals:
            result[key] = {"mean": None, "ci95_low": None, "ci95_high": None}
            continue
        arr = np.asarray(vals, dtype=np.float64)
        result[key] = {
            "mean": float(arr.mean()),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    return result


def sanitize_json_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [sanitize_json_value(float(v)) for v in value.tolist()]
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    return value


def save_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_results(
    output_dir: Path,
    image_metrics: Sequence[ImageMetrics],
    scene_metrics: Sequence[SceneMetrics],
    summary: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "roughness_metrics.csv", [asdict(m) for m in image_metrics])
    save_csv(output_dir / "roughness_scene_metrics.csv", [asdict(m) for m in scene_metrics])
    with open(output_dir / "roughness_summary.json", "w", encoding="utf-8") as f:
        json.dump(sanitize_json_value(summary), f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    validate_inputs(args)

    single_mode = bool(args.gt_path and (args.pred_path or args.baseline != "none"))
    output_dir = Path(args.output_dir)
    resized_pred_dir = output_dir / "resized_predictions" if args.save_resized_preds else None

    if skimage_ssim is None:
        print("[warn] SSIM disabled: missing optional dependency skimage.", file=sys.stderr)
    lpips_eval = None
    if args.compute_lpips:
        lpips_eval = LPIPSEvaluator(device=args.device, net=args.lpips_net)
        if not lpips_eval.available:
            print(f"[warn] LPIPS disabled: {lpips_eval.unavailable_reason}", file=sys.stderr)

    baseline_value: Optional[float] = args.constant_value
    if args.baseline == "calibration_mean":
        baseline_value = compute_calibration_mean(
            gt_dir=Path(args.calibration_gt_dir),
            mask_dir=Path(args.calibration_mask_dir) if args.calibration_mask_dir else None,
            args=args,
        )

    if single_mode:
        gt_dir = None
        pairs = [(
            Path(args.gt_path),
            Path(args.pred_path) if args.baseline == "none" else None,
            Path(args.mask_path) if args.mask_path else None,
            Path(args.region_path) if args.region_path else None,
        )]
    else:
        gt_dir = Path(args.gt_dir)
        pred_dir = Path(args.pred_dir) if args.pred_dir else None
        mask_dir = Path(args.mask_dir) if args.mask_dir else None
        region_dir = Path(args.region_dir) if args.region_dir else None
        pairs = build_pairs_from_dirs(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            mask_dir=mask_dir,
            region_dir=region_dir,
            pair_mode=args.pair_mode,
            pred_suffix_to_strip=args.pred_suffix_to_strip,
            gt_suffix_to_strip=args.gt_suffix_to_strip,
            mask_suffix=args.mask_suffix,
            region_suffix=args.region_suffix,
            pred_path_component_to_strip=args.pred_path_component_to_strip,
            gt_path_component_to_strip=args.gt_path_component_to_strip,
            mask_path_component_to_strip=args.mask_path_component_to_strip,
            region_path_component_to_strip=args.region_path_component_to_strip,
            recursive=args.recursive,
            synthetic_baseline=args.baseline != "none",
        )
        if not pairs:
            raise RuntimeError("No matched GT/prediction pairs found.")

    image_metrics: List[ImageMetrics] = []
    pair_diags: List[PairDiagnostics] = []
    scene_accs: Dict[str, SceneAccumulator] = {}

    for gt_path, pred_path, mask_path, region_path in pairs:
        name = gt_path.stem if single_mode else gt_path.relative_to(gt_dir).as_posix()
        metric, diag = evaluate_pair(
            gt_path=gt_path,
            pred_path=pred_path,
            mask_path=mask_path,
            region_path=region_path,
            args=args,
            lpips_eval=lpips_eval,
            resized_pred_dir=resized_pred_dir,
            name=name,
            baseline_value=baseline_value,
        )
        image_metrics.append(metric)
        pair_diags.append(diag)
        acc = scene_accs.get(metric.scene_id)
        if acc is None:
            acc = SceneAccumulator(scene_id=metric.scene_id, split=metric.split)
            scene_accs[metric.scene_id] = acc
        acc.update(diag)
        print(
            f"[{metric.name}] "
            f"RMSE={metric.rmse:.6f} | MAE={metric.mae:.6f} | "
            f"Bin-BMAE={metric.bin_bmae:.6f} | "
            f"Region-MAE={metric.region_macro_mae:.6f} | "
            f"SSIM={metric.ssim:.6f} | "
            f"LPIPS={metric.lpips if metric.lpips is not None else 'N/A'}"
        )

    scene_metrics = [scene_accs[key].finalize() for key in sorted(scene_accs.keys())]

    summary: Dict[str, object] = {
        "config": {
            "baseline": args.baseline,
            "baseline_value": baseline_value,
            "region_source": args.region_source,
            "resize_interp": args.resize_interp,
            "mask_enabled": not args.disable_mask,
        },
        "image_level": summarize_image_metrics(image_metrics),
        "bin_summary": aggregate_bin_summary(pair_diags, scene_metrics),
        "region_summary": aggregate_region_summary(pair_diags),
        "main_table_recommendation": {
            "rmse": summarize_image_metrics(image_metrics)["rmse_mean"],
            "bin_bmae": aggregate_bin_summary(pair_diags, scene_metrics)["bin_bmae_global"],
            "region_macro_mae": aggregate_region_summary(pair_diags)["region_macro_mae_global"],
            "ssim": summarize_image_metrics(image_metrics)["ssim_mean"],
        },
        "by_split": summarize_by_split(image_metrics, scene_metrics, scene_accs),
        "bootstrap_ci": bootstrap_summary(list(scene_accs.values()), args.bootstrap_iters, args.bootstrap_seed),
        "metric_groups": {
            "main_table": ["rmse", "bin_bmae", "region_macro_mae", "ssim"],
            "appendix": [
                "mae",
                "psnr",
                "lpips",
                "per_bin_mae",
                "region_macro_rmse",
            ],
        },
        "notes": {
            "bin_bmae": (
                "Bin-BMAE is computed by partitioning GT roughness into five fixed bins over [0,1], "
                "computing MAE inside each bin, then averaging the bin MAEs with equal weight."
            ),
            "region_macro": (
                "Region-macro metrics treat each resolved region label equally. "
                "When OpenRooms-style ObjectSegmentation is used, this should be interpreted as instance/region diagnostics, "
                "not necessarily exact material-instance scoring."
            ),
            "baseline": (
                "Calibration-mean baseline uses a frozen global GT roughness mean estimated from independent calibration scenes."
            ),
        },
    }

    save_results(output_dir, image_metrics, scene_metrics, summary)

    print("\n=== Summary ===")
    print(json.dumps(sanitize_json_value(summary["main_table_recommendation"]), indent=2, ensure_ascii=False))
    if summary["bootstrap_ci"]:
        print("\n=== Bootstrap CI ===")
        print(json.dumps(sanitize_json_value(summary["bootstrap_ci"]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
