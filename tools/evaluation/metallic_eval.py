#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as skimage_ssim


KNOWN_SPLITS = {"mainaxis", "stresstest"}
DEFAULT_SENSITIVITY_GT_THRESHOLDS = (0.1, 0.3, 0.5)
STRESSTEST_LIGHTING_GROUPS = [
    ("L0", "normal_light"),
    ("L1", "Low-light"),
    ("L2", "HDR"),
    ("L3", "dark heavy"),
    ("L4", "Highlight-heavy"),
    ("L5", "Mix-temperature lighting"),
]


@dataclass
class ImageMetrics:
    name: str
    split: str
    scene_id: str
    lighting: str
    gt_path: str
    pred_path: str
    mask_path: str
    width: int
    height: int
    total_pixels: int
    valid_pixels: int
    mask_coverage: float
    gt_positive_pixels: int
    gt_remaining_pixels: int
    gt_positive_ratio: float
    pred_positive_pixels: int
    pred_positive_ratio: float
    mae: float
    rmse: float
    psnr: float
    ssim: float
    positive_mae: float
    remaining_mae: float
    rb_mae: float
    metal_weighted_mae: float
    nonmetal_weighted_mae: float
    soft_rb_mae: float
    precision_at_05: float
    recall_at_05: float
    f1_at_05: float
    iou_at_05: float
    fpr_at_05: float
    ap: float


@dataclass
class SceneMetrics:
    scene_id: str
    split: str
    lighting: str
    num_images: int
    valid_pixels: int
    gt_positive_pixels: int
    gt_remaining_pixels: int
    gt_positive_ratio: float
    pred_positive_pixels: int
    pred_positive_ratio: float
    is_positive_scene: bool
    mae: float
    rmse: float
    positive_mae: float
    remaining_mae: float
    rb_mae: float
    metal_weighted_mae: float
    nonmetal_weighted_mae: float
    soft_rb_mae: float
    precision_at_05: float
    recall_at_05: float
    f1_at_05: float
    iou_at_05: float
    fpr_at_05: float
    ap: float


@dataclass
class PairCacheEntry:
    name: str
    split: str
    scene_id: str
    lighting: str
    valid_pixels: int
    total_pixels: int
    gt_valid_path: str
    pred_valid_path: str


def safe_cache_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "item"


def load_cached_pair_arrays(pair: PairCacheEntry) -> Tuple[np.ndarray, np.ndarray]:
    gt_valid = np.load(pair.gt_valid_path, mmap_mode="r")
    pred_valid = np.load(pair.pred_valid_path, mmap_mode="r")
    return gt_valid, pred_valid


class SceneAccumulator:
    def __init__(self, scene_id: str, split: str, lighting: str, cache_dir: Optional[Path] = None) -> None:
        self.scene_id = scene_id
        self.split = split
        self.lighting = lighting
        self.cache_dir = cache_dir
        self.chunk_index = 0
        self.score_chunk_paths: List[str] = []
        self.label_chunk_paths: List[str] = []
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_images = 0
        self.valid_pixels = 0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.pos_abs_sum = 0.0
        self.neg_abs_sum = 0.0
        self.metal_weighted_abs_sum = 0.0
        self.nonmetal_weighted_abs_sum = 0.0
        self.metal_weight_sum = 0.0
        self.nonmetal_weight_sum = 0.0
        self.pos_count = 0
        self.neg_count = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.gt_positive_pixels = 0
        self.pred_positive_pixels = 0
        self.score_chunks: List[np.ndarray] = []
        self.label_chunks: List[np.ndarray] = []

    def update(self, gt_valid: np.ndarray, pred_valid: np.ndarray, gt_binary: np.ndarray, pred_binary: np.ndarray) -> None:
        diff = np.abs(gt_valid - pred_valid)
        self.num_images += 1
        self.valid_pixels += int(gt_valid.size)
        self.abs_sum += float(diff.sum())
        self.sq_sum += float(((gt_valid - pred_valid) ** 2).sum())

        pos_mask = gt_binary
        neg_mask = ~gt_binary
        self.pos_count += int(pos_mask.sum())
        self.neg_count += int(neg_mask.sum())
        self.pos_abs_sum += float(diff[pos_mask].sum()) if pos_mask.any() else 0.0
        self.neg_abs_sum += float(diff[neg_mask].sum()) if neg_mask.any() else 0.0
        self.metal_weighted_abs_sum += float((gt_valid * diff).sum())
        self.nonmetal_weighted_abs_sum += float(((1.0 - gt_valid) * diff).sum())
        self.metal_weight_sum += float(gt_valid.sum())
        self.nonmetal_weight_sum += float((1.0 - gt_valid).sum())

        self.gt_positive_pixels += int(gt_binary.sum())
        self.pred_positive_pixels += int(pred_binary.sum())
        self.tp += int(np.logical_and(gt_binary, pred_binary).sum())
        self.fp += int(np.logical_and(~gt_binary, pred_binary).sum())
        self.fn += int(np.logical_and(gt_binary, ~pred_binary).sum())
        self.tn += int(np.logical_and(~gt_binary, ~pred_binary).sum())

        score_chunk = pred_valid.astype(np.float32, copy=False)
        label_chunk = gt_binary.astype(bool, copy=False)
        if self.cache_dir is None:
            self.score_chunks.append(score_chunk)
            self.label_chunks.append(label_chunk)
        else:
            prefix = f"{safe_cache_name(self.scene_id)}_{self.chunk_index:06d}"
            score_path = self.cache_dir / f"{prefix}_score.npy"
            label_path = self.cache_dir / f"{prefix}_label.npy"
            np.save(score_path, score_chunk)
            np.save(label_path, label_chunk)
            self.score_chunk_paths.append(str(score_path))
            self.label_chunk_paths.append(str(label_path))
            self.chunk_index += 1

    def finalize(self) -> SceneMetrics:
        if self.score_chunk_paths:
            labels = np.concatenate([np.load(path, mmap_mode="r") for path in self.label_chunk_paths])
            scores = np.concatenate([np.load(path, mmap_mode="r") for path in self.score_chunk_paths])
        else:
            labels = np.concatenate(self.label_chunks) if self.label_chunks else np.zeros((0,), dtype=bool)
            scores = np.concatenate(self.score_chunks) if self.score_chunks else np.zeros((0,), dtype=np.float32)

        mae = safe_div(self.abs_sum, self.valid_pixels)
        mse = safe_div(self.sq_sum, self.valid_pixels)
        rmse = float(math.sqrt(mse)) if np.isfinite(mse) else float("nan")
        positive_mae = safe_div(self.pos_abs_sum, self.pos_count)
        remaining_mae = safe_div(self.neg_abs_sum, self.neg_count)
        rb_mae = balanced_mean(positive_mae, remaining_mae)
        metal_weighted_mae = safe_div(self.metal_weighted_abs_sum, self.metal_weight_sum)
        nonmetal_weighted_mae = safe_div(self.nonmetal_weighted_abs_sum, self.nonmetal_weight_sum)
        soft_rb_mae = balanced_mean(metal_weighted_mae, nonmetal_weighted_mae)
        precision, recall, f1, iou, fpr = threshold_metrics_from_counts(
            tp=self.tp,
            fp=self.fp,
            fn=self.fn,
            tn=self.tn,
            positive_required_for_iou=True,
        )
        ap = average_precision_from_scores(labels, scores)

        return SceneMetrics(
            scene_id=self.scene_id,
            split=self.split,
            lighting=self.lighting,
            num_images=self.num_images,
            valid_pixels=self.valid_pixels,
            gt_positive_pixels=self.gt_positive_pixels,
            gt_remaining_pixels=self.neg_count,
            gt_positive_ratio=safe_div(self.gt_positive_pixels, self.valid_pixels),
            pred_positive_pixels=self.pred_positive_pixels,
            pred_positive_ratio=safe_div(self.pred_positive_pixels, self.valid_pixels),
            is_positive_scene=self.gt_positive_pixels > 0,
            mae=mae,
            rmse=rmse,
            positive_mae=positive_mae,
            remaining_mae=remaining_mae,
            rb_mae=rb_mae,
            metal_weighted_mae=metal_weighted_mae,
            nonmetal_weighted_mae=nonmetal_weighted_mae,
            soft_rb_mae=soft_rb_mae,
            precision_at_05=precision,
            recall_at_05=recall,
            f1_at_05=f1,
            iou_at_05=iou,
            fpr_at_05=fpr,
            ap=ap,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate metallic predictions with full-map MAE plus region-balanced "
            "and localization-oriented supplementary metrics."
        )
    )
    parser.add_argument("--gt_root", type=Path, required=True, help="GT root directory.")
    parser.add_argument(
        "--pred_root",
        type=Path,
        required=True,
        help="Prediction root directory. Also defines the evaluation subset for synthetic baselines.",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save CSV/JSON results.")
    parser.add_argument(
        "--resize_interp",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        help="Interpolation used when resizing predictions to GT resolution.",
    )
    parser.add_argument(
        "--gt_threshold",
        type=float,
        default=0.5,
        help="GT threshold used to define the high-metallic region for RB-MAE and localization metrics.",
    )
    parser.add_argument(
        "--pred_threshold",
        type=float,
        default=0.5,
        help="Prediction threshold used for IoU / precision / recall / F1 / FPR.",
    )
    parser.add_argument(
        "--sensitivity_gt_thresholds",
        type=float,
        nargs="*",
        default=list(DEFAULT_SENSITIVITY_GT_THRESHOLDS),
        help=(
            "GT thresholds for sensitivity analysis. Defaults to 0.1 0.3 0.5. "
            "Pass an empty list explicitly only if you want to suppress the sweep in code changes."
        ),
    )
    parser.add_argument(
        "--sensitivity_pred_thresholds",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional prediction thresholds paired with --sensitivity_gt_thresholds. "
            "If omitted, prediction thresholds are set equal to the GT thresholds."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="none",
        choices=["none", "all_black", "constant", "calibration_mean"],
        help="Optional synthetic baseline. When enabled, pred_root is used only to define the evaluation subset.",
    )
    parser.add_argument(
        "--constant_value",
        type=float,
        default=None,
        help="Constant prediction value for --baseline constant.",
    )
    parser.add_argument(
        "--calibration_gt_root",
        type=Path,
        default=None,
        help="GT root used to estimate the calibration-set mean for --baseline calibration_mean.",
    )
    parser.add_argument(
        "--skip_lpips",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--save_resized_preds",
        action="store_true",
        default=False,
        help="Save resized predictions next to the outputs for inspection.",
    )
    parser.add_argument(
        "--mask_fallback",
        type=str,
        default="none",
        choices=["none", "object_segmentation", "full_image", "auto"],
        help=(
            "Fallback when MetallicEvalMask is absent. "
            "'auto' tries MetallicEvalMask first, then ObjectSegmentation. "
            "'full_image' uses all pixels inside the image."
        ),
    )
    parser.add_argument(
        "--bootstrap_iters",
        type=int,
        default=0,
        help="Optional number of scene-cluster bootstrap iterations for 95%% CIs.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=0,
        help="Random seed for scene-cluster bootstrap.",
    )
    parser.add_argument(
        "--pixel_cache_dir",
        type=Path,
        default=None,
        help=(
            "Directory for temporary per-pixel caches used by AP and threshold sensitivity. "
            "Defaults to output_dir/_pixel_cache. Keeps RAM bounded without dropping metrics."
        ),
    )
    parser.add_argument(
        "--keep_pixel_cache",
        action="store_true",
        help="Keep temporary per-pixel cache files after a successful run.",
    )
    return parser.parse_args()


def pil_interp_mode(name: str) -> int:
    if name == "nearest":
        return Image.Resampling.NEAREST
    if name == "bilinear":
        return Image.Resampling.BILINEAR
    if name == "bicubic":
        return Image.Resampling.BICUBIC
    if name == "lanczos":
        return Image.Resampling.LANCZOS
    raise ValueError(f"Unknown interpolation: {name}")


def load_gray(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("L")


def load_gt_gray16(path: Path) -> Image.Image:
    """Read a 16-bit grayscale PNG without converting through 8-bit L mode."""
    with path.open("rb") as handle:
        header = handle.read(29)
    if (len(header) < 29 or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[12:16] != b"IHDR" or header[24] != 16 or header[25] != 0):
        raise ValueError(f"Expected a 16-bit grayscale PNG GT: {path}")
    with Image.open(path) as img:
        values = np.asarray(img, dtype=np.float32) / 65535.0
    return Image.fromarray(values, mode="F")


def image_to_float_np(img: Image.Image) -> np.ndarray:
    values = np.asarray(img, dtype=np.float32)
    # F images in this evaluator already contain normalized floating-point values.
    return values if img.mode == "F" else values / 255.0


def safe_div(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num / den)


def balanced_mean(a: float, b: float) -> float:
    vals = [v for v in (a, b) if np.isfinite(v)]
    if len(vals) != 2:
        return float("nan")
    return float(0.5 * (vals[0] + vals[1]))


def psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if not np.isfinite(mse):
        return float("nan")
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10((data_range ** 2) / mse))


def metric_values(values: Sequence[Optional[float]]) -> List[float]:
    return [float(v) for v in values if v is not None and np.isfinite(v)]


def metric_mean(values: Sequence[Optional[float]]) -> float:
    vals = metric_values(values)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def metric_median(values: Sequence[Optional[float]]) -> float:
    vals = metric_values(values)
    if not vals:
        return float("nan")
    return float(np.median(vals))


def metric_std(values: Sequence[Optional[float]]) -> float:
    vals = metric_values(values)
    if not vals:
        return float("nan")
    return float(np.std(vals))


def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    h, w = gt.shape
    if min(h, w) < 7:
        return float("nan")
    return float(skimage_ssim(gt, pred, data_range=1.0))


def average_precision_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute non-interpolated AP at the end of each distinct score group."""
    if labels.size == 0:
        return float("nan")
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")

    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.float64)
    cumulative_tp = np.cumsum(sorted_labels)

    # Evaluate precision/recall only after every pixel with the same prediction
    # score has been included. Exact score equality is intentional: scores are
    # neither quantized nor merged using a tolerance.
    group_end_indices = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    )
    group_end_tp = cumulative_tp[group_end_indices]
    group_positive_counts = np.diff(np.r_[0.0, group_end_tp])
    group_end_precision = group_end_tp / (group_end_indices.astype(np.float64) + 1.0)

    return float(np.sum((group_positive_counts / positives) * group_end_precision))


def threshold_metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    positive_required_for_iou: bool,
) -> Tuple[float, float, float, float, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = float("nan")
    if positive_required_for_iou and (tp + fn) == 0:
        iou = float("nan")
    else:
        iou = safe_div(tp, tp + fp + fn)
    fpr = safe_div(fp, fp + tn)
    return precision, recall, f1, iou, fpr


def normalize_split_name(part: str) -> str:
    if part.startswith("companion_"):
        return part[len("companion_") :]
    return part


def build_pred_to_gt_path(pred_path: Path, pred_root: Path, gt_root: Path) -> Path:
    rel = pred_path.relative_to(pred_root)
    parts = list(rel.parts)

    if len(parts) >= 7 and parts[-2] == "metallic":
        camera_dir = parts[-3]
        scene_prefix = parts[:-4]
    elif len(parts) >= 7 and parts[-3] == "Image":
        camera_dir = parts[-2]
        scene_prefix = parts[:-3]
    else:
        raise ValueError(f"Unexpected prediction path layout: {pred_path}")

    if scene_prefix:
        scene_prefix[0] = normalize_split_name(scene_prefix[0])

    return gt_root.joinpath(*scene_prefix, "Metallic", camera_dir, "Metallic_0001.png")


def build_gt_to_mask_path(gt_path: Path) -> Path:
    if gt_path.parent.parent.name != "Metallic":
        raise ValueError(f"Unexpected GT path layout: {gt_path}")
    view_dir = gt_path.parent.parent.parent
    camera_dir = gt_path.parent.name
    return view_dir / "MetallicEvalMask" / camera_dir / "MetallicEvalMask_0001.png"


def build_gt_to_object_seg_path(gt_path: Path) -> Path:
    if gt_path.parent.parent.name != "Metallic":
        raise ValueError(f"Unexpected GT path layout: {gt_path}")
    view_dir = gt_path.parent.parent.parent
    camera_dir = gt_path.parent.name
    camera_suffix = camera_dir.split("_")[-1]
    return view_dir / "ObjectSegmentation" / camera_dir / f"ObjectSegmentation_0_0_0001_{camera_suffix}.png"


def scene_id_from_gt_path(gt_path: Path, gt_root: Path) -> str:
    rel_parts = list(gt_path.relative_to(gt_root).parts)
    if rel_parts:
        rel_parts[0] = normalize_split_name(rel_parts[0])
    if len(rel_parts) < 3:
        raise ValueError(f"Unexpected GT path layout for scene id: {gt_path}")
    # In stresstest, V*_P*_L* is the lighting/view condition. Keep it in
    # the scene key so L0-L5 are not merged into the first (usually L0) scene.
    scene_parts = 4 if rel_parts[0] == "stresstest" and len(rel_parts) >= 4 else 3
    return "/".join(rel_parts[:scene_parts])


def lighting_from_name(name: str, split: str) -> str:
    if split != "stresstest":
        return ""
    for part in Path(name).parts:
        match = re.search(r"(?:^|_)L([0-5])(?:_|$)", part)
        if match:
            return f"L{match.group(1)}"
    return "unknown"


def is_prediction_metallic_path(path: Path) -> bool:
    if path.suffix.lower() != ".png":
        return False
    stem = path.stem
    if path.parent.name == "metallic":
        return stem.startswith("Image_") and stem.endswith("_metallic")
    if path.parent.parent.name != "Image":
        return False
    if stem.endswith("_metallic"):
        return True
    non_metallic_suffixes = ("_albedo", "_depth", "_normal", "_roughness")
    return stem.startswith("Image_") and not stem.endswith(non_metallic_suffixes)


def resolve_mask_path(gt_path: Path, mask_fallback: str) -> Optional[Path]:
    metallic_eval_mask = build_gt_to_mask_path(gt_path)
    if metallic_eval_mask.exists():
        return metallic_eval_mask

    object_seg_path = build_gt_to_object_seg_path(gt_path)
    if mask_fallback in {"auto", "object_segmentation"} and object_seg_path.exists():
        return object_seg_path

    if mask_fallback == "full_image":
        return Path("__FULL_IMAGE__")

    return None


def collect_pairs(pred_root: Path, gt_root: Path, mask_fallback: str) -> List[Tuple[Path, Path, Path, str, str, str]]:
    pred_files = sorted(path for path in pred_root.rglob("*.png") if is_prediction_metallic_path(path))
    pairs: List[Tuple[Path, Path, Path, str, str, str]] = []
    missing_gt: List[Path] = []
    missing_mask: List[Path] = []

    for pred_path in pred_files:
        gt_path = build_pred_to_gt_path(pred_path, pred_root, gt_root)
        if not gt_path.exists():
            missing_gt.append(gt_path)
            continue
        mask_path = resolve_mask_path(gt_path, mask_fallback=mask_fallback)
        if mask_path is None:
            missing_mask.append(build_gt_to_mask_path(gt_path))
            continue

        rel = gt_path.relative_to(gt_root)
        name = rel.as_posix()
        split = normalize_split_name(rel.parts[0])
        scene_id = scene_id_from_gt_path(gt_path, gt_root)
        pairs.append((gt_path, pred_path, mask_path, name, split, scene_id))

    if missing_gt:
        print(f"Warning: {len(missing_gt)} prediction files did not find GT. First missing GT: {missing_gt[0]}")
    if missing_mask:
        print(f"Warning: {len(missing_mask)} prediction files did not find mask. First missing mask: {missing_mask[0]}")
    return pairs


def collect_calibration_mean(calibration_gt_root: Path, gt_threshold: float) -> float:
    total_sum = 0.0
    total_count = 0
    gt_files = sorted(calibration_gt_root.rglob("Metallic_0001.png"))
    for gt_path in gt_files:
        if gt_path.parent.parent.name != "Metallic":
            continue
        mask_path = build_gt_to_mask_path(gt_path)
        if not mask_path.exists():
            continue
        gt_img = load_gt_gray16(gt_path)
        gt = image_to_float_np(gt_img)
        mask = load_eval_mask(mask_path, gt_img.size)
        gt_valid = gt[mask]
        total_sum += float(gt_valid.sum())
        total_count += int(gt_valid.size)
    if total_count <= 0:
        raise RuntimeError(f"Could not compute calibration mean from {calibration_gt_root}")
    mean_value = float(total_sum / total_count)
    print(
        f"Calibration-set mean baseline value = {mean_value:.6f} "
        f"(computed over {total_count} valid pixels, gt_threshold={gt_threshold})."
    )
    return mean_value


def load_eval_mask(mask_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    if str(mask_path) == "__FULL_IMAGE__":
        return np.ones((target_size[1], target_size[0]), dtype=bool)
    mask_img = load_gray(mask_path)
    if mask_img.size != target_size:
        mask_img = mask_img.resize(target_size, Image.Resampling.NEAREST)
    return image_to_float_np(mask_img) > 0.5


def make_prediction_array(
    pred_path: Path,
    gt_img: Image.Image,
    resize_interp: str,
    baseline: str,
    constant_value: Optional[float],
) -> Tuple[Image.Image, str]:
    if baseline == "all_black":
        pred_arr = np.zeros((gt_img.size[1], gt_img.size[0]), dtype=np.uint8)
        return Image.fromarray(pred_arr, mode="L"), "baseline://all_black"
    if baseline in {"constant", "calibration_mean"}:
        if constant_value is None:
            raise ValueError(f"baseline {baseline} requires a numeric constant value")
        value = int(round(float(np.clip(constant_value, 0.0, 1.0)) * 255.0))
        pred_arr = np.full((gt_img.size[1], gt_img.size[0]), value, dtype=np.uint8)
        return Image.fromarray(pred_arr, mode="L"), f"baseline://constant/{float(np.clip(constant_value, 0.0, 1.0)):.6f}"

    pred_values = image_to_float_np(load_gray(pred_path))
    pred_img = Image.fromarray(pred_values, mode="F")
    if pred_img.size != gt_img.size:
        pred_img = pred_img.resize(gt_img.size, pil_interp_mode(resize_interp))
    return pred_img, str(pred_path)


def evaluate_pair(
    gt_path: Path,
    pred_path: Path,
    mask_path: Path,
    name: str,
    split: str,
    scene_id: str,
    pred_root: Path,
    resize_interp: str,
    gt_threshold: float,
    pred_threshold: float,
    baseline: str,
    baseline_value: Optional[float],
    resized_pred_dir: Optional[Path],
) -> Tuple[ImageMetrics, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt_img = load_gt_gray16(gt_path)
    pred_img, pred_label = make_prediction_array(
        pred_path=pred_path,
        gt_img=gt_img,
        resize_interp=resize_interp,
        baseline=baseline,
        constant_value=baseline_value,
    )

    if resized_pred_dir is not None:
        resized_target = resized_pred_dir / pred_path.relative_to(pred_root)
        resized_target.parent.mkdir(parents=True, exist_ok=True)
        # Quantize only the inspection PNG; metrics keep floating-point predictions.
        preview = np.clip(np.rint(image_to_float_np(pred_img) * 255.0), 0, 255).astype(np.uint8)
        Image.fromarray(preview, mode="L").save(resized_target)

    gt = image_to_float_np(gt_img)
    pred = image_to_float_np(pred_img)
    mask = load_eval_mask(mask_path, gt_img.size)

    total_pixels = int(gt.size)
    valid_pixels = int(mask.sum())
    if valid_pixels <= 0:
        raise ValueError(f"Mask contains no valid pixels: {mask_path}")

    gt_valid = gt[mask]
    pred_valid = pred[mask]

    diff = np.abs(gt_valid - pred_valid)
    diff_sq = (gt_valid - pred_valid) ** 2
    mae = safe_div(float(diff.sum()), valid_pixels)
    mse = safe_div(float(diff_sq.sum()), valid_pixels)
    rmse = float(math.sqrt(mse)) if np.isfinite(mse) else float("nan")
    psnr = psnr_from_mse(mse)

    gt_masked = np.where(mask, gt, 0.0)
    pred_masked = np.where(mask, pred, 0.0)
    ssim = compute_ssim(gt_masked, pred_masked)

    gt_binary = gt_valid >= gt_threshold
    pred_binary = pred_valid >= pred_threshold

    pos_mask = gt_binary
    neg_mask = ~gt_binary
    positive_pixels = int(pos_mask.sum())
    remaining_pixels = int(neg_mask.sum())
    positive_mae = safe_div(float(diff[pos_mask].sum()), positive_pixels)
    remaining_mae = safe_div(float(diff[neg_mask].sum()), remaining_pixels)
    rb_mae = balanced_mean(positive_mae, remaining_mae)
    metal_weighted_mae = safe_div(float((gt_valid * diff).sum()), float(gt_valid.sum()))
    nonmetal_weighted_mae = safe_div(float(((1.0 - gt_valid) * diff).sum()), float((1.0 - gt_valid).sum()))
    soft_rb_mae = balanced_mean(metal_weighted_mae, nonmetal_weighted_mae)

    tp = int(np.logical_and(gt_binary, pred_binary).sum())
    fp = int(np.logical_and(~gt_binary, pred_binary).sum())
    fn = int(np.logical_and(gt_binary, ~pred_binary).sum())
    tn = int(np.logical_and(~gt_binary, ~pred_binary).sum())
    precision, recall, f1, iou, fpr = threshold_metrics_from_counts(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        positive_required_for_iou=True,
    )
    ap = average_precision_from_scores(gt_binary, pred_valid)

    metrics = ImageMetrics(
        name=name,
        split=split,
        scene_id=scene_id,
        lighting=lighting_from_name(name, split),
        gt_path=name,
        pred_path=pred_label if baseline != "none" else pred_path.relative_to(pred_root).as_posix(),
        mask_path=(Path(name).parent / mask_path.name).as_posix()
        if str(mask_path) != "__FULL_IMAGE__"
        else "__FULL_IMAGE__",
        width=gt_img.size[0],
        height=gt_img.size[1],
        total_pixels=total_pixels,
        valid_pixels=valid_pixels,
        mask_coverage=safe_div(valid_pixels, total_pixels),
        gt_positive_pixels=positive_pixels,
        gt_remaining_pixels=remaining_pixels,
        gt_positive_ratio=safe_div(positive_pixels, valid_pixels),
        pred_positive_pixels=int(pred_binary.sum()),
        pred_positive_ratio=safe_div(int(pred_binary.sum()), valid_pixels),
        mae=mae,
        rmse=rmse,
        psnr=psnr,
        ssim=ssim,
        positive_mae=positive_mae,
        remaining_mae=remaining_mae,
        rb_mae=rb_mae,
        metal_weighted_mae=metal_weighted_mae,
        nonmetal_weighted_mae=nonmetal_weighted_mae,
        soft_rb_mae=soft_rb_mae,
        precision_at_05=precision,
        recall_at_05=recall,
        f1_at_05=f1,
        iou_at_05=iou,
        fpr_at_05=fpr,
        ap=ap,
    )
    return metrics, gt_valid, pred_valid, gt_binary, pred_binary


def summarize_images(metrics: Sequence[ImageMetrics]) -> dict:
    return {
        "count": len(metrics),
        "mae_mean": metric_mean([m.mae for m in metrics]),
        "mae_median": metric_median([m.mae for m in metrics]),
        "mae_std": metric_std([m.mae for m in metrics]),
        "rb_mae_mean": metric_mean([m.rb_mae for m in metrics]),
        "soft_rb_mae_mean": metric_mean([m.soft_rb_mae for m in metrics]),
        "metal_weighted_mae_mean": metric_mean([m.metal_weighted_mae for m in metrics]),
        "nonmetal_weighted_mae_mean": metric_mean([m.nonmetal_weighted_mae for m in metrics]),
        "positive_mae_mean": metric_mean([m.positive_mae for m in metrics]),
        "remaining_mae_mean": metric_mean([m.remaining_mae for m in metrics]),
        "iou_mean_positive_images_only": metric_mean([m.iou_at_05 for m in metrics if m.gt_positive_pixels > 0]),
        "ap_mean_positive_images_only": metric_mean([m.ap for m in metrics if m.gt_positive_pixels > 0]),
        "precision_mean": metric_mean([m.precision_at_05 for m in metrics]),
        "recall_mean_positive_images_only": metric_mean([m.recall_at_05 for m in metrics if m.gt_positive_pixels > 0]),
        "f1_mean_positive_images_only": metric_mean([m.f1_at_05 for m in metrics if m.gt_positive_pixels > 0]),
        "negative_image_fpr_mean": metric_mean([m.fpr_at_05 for m in metrics if m.gt_positive_pixels == 0]),
        "gt_positive_ratio_mean": metric_mean([m.gt_positive_ratio for m in metrics]),
        "pred_positive_ratio_mean": metric_mean([m.pred_positive_ratio for m in metrics]),
        "psnr_mean": metric_mean([m.psnr for m in metrics]),
        "ssim_mean": metric_mean([m.ssim for m in metrics]),
    }


def summarize_scenes(scene_metrics: Sequence[SceneMetrics]) -> dict:
    positive_scenes = [m for m in scene_metrics if m.is_positive_scene]
    negative_scenes = [m for m in scene_metrics if not m.is_positive_scene]
    return {
        "count": len(scene_metrics),
        "positive_scene_count": len(positive_scenes),
        "negative_scene_count": len(negative_scenes),
        "mae_scene_macro": metric_mean([m.mae for m in scene_metrics]),
        "rb_mae_scene_macro": metric_mean([m.rb_mae for m in scene_metrics]),
        "soft_rb_mae_scene_macro": metric_mean([m.soft_rb_mae for m in scene_metrics]),
        "metal_weighted_mae_scene_macro": metric_mean([m.metal_weighted_mae for m in scene_metrics]),
        "nonmetal_weighted_mae_scene_macro": metric_mean([m.nonmetal_weighted_mae for m in scene_metrics]),
        "positive_mae_scene_macro": metric_mean([m.positive_mae for m in scene_metrics]),
        "remaining_mae_scene_macro": metric_mean([m.remaining_mae for m in scene_metrics]),
        "iou_positive_scene_macro": metric_mean([m.iou_at_05 for m in positive_scenes]),
        "ap_positive_scene_macro": metric_mean([m.ap for m in positive_scenes]),
        "precision_positive_scene_macro": metric_mean([m.precision_at_05 for m in positive_scenes]),
        "recall_positive_scene_macro": metric_mean([m.recall_at_05 for m in positive_scenes]),
        "f1_positive_scene_macro": metric_mean([m.f1_at_05 for m in positive_scenes]),
        "negative_scene_fpr_macro": metric_mean([m.fpr_at_05 for m in negative_scenes]),
        "gt_positive_ratio_scene_macro": metric_mean([m.gt_positive_ratio for m in scene_metrics]),
        "pred_positive_ratio_scene_macro": metric_mean([m.pred_positive_ratio for m in scene_metrics]),
    }


def summarize_by_split(scene_metrics: Sequence[SceneMetrics]) -> dict:
    grouped: Dict[str, dict] = {}
    for split in sorted({m.split for m in scene_metrics}):
        split_metrics = [m for m in scene_metrics if m.split == split]
        grouped[split] = {"overall": summarize_scenes(split_metrics)}
        if split == "stresstest":
            grouped[split]["by_lighting"] = {
                f"{code}_{label}": summarize_scenes([m for m in split_metrics if m.lighting == code])
                for code, label in STRESSTEST_LIGHTING_GROUPS
            }
            unknown_lighting = [m for m in split_metrics if m.lighting == "unknown"]
            if unknown_lighting:
                grouped[split]["by_lighting"]["unknown"] = summarize_scenes(unknown_lighting)
    return grouped


def threshold_key(gt_threshold: float, pred_threshold: float) -> str:
    return f"gt_{gt_threshold:.2f}_pred_{pred_threshold:.2f}"


def summarize_threshold_metrics(
    cached_pairs: Sequence[PairCacheEntry],
    gt_threshold: float,
    pred_threshold: float,
    cache_dir: Optional[Path] = None,
) -> dict:
    image_rows = []
    scene_accumulators: Dict[str, SceneAccumulator] = {}

    scene_cache_dir = cache_dir / "scene_chunks" if cache_dir is not None else None

    for pair in cached_pairs:
        gt_valid, pred_valid = load_cached_pair_arrays(pair)
        gt_binary = gt_valid >= gt_threshold
        pred_binary = pred_valid >= pred_threshold
        diff = np.abs(gt_valid - pred_valid)

        positive_pixels = int(gt_binary.sum())
        remaining_pixels = int((~gt_binary).sum())
        positive_mae = safe_div(float(diff[gt_binary].sum()), positive_pixels)
        remaining_mae = safe_div(float(diff[~gt_binary].sum()), remaining_pixels)
        rb_mae = balanced_mean(positive_mae, remaining_mae)
        metal_weighted_mae = safe_div(float((gt_valid * diff).sum()), float(gt_valid.sum()))
        nonmetal_weighted_mae = safe_div(
            float(((1.0 - gt_valid) * diff).sum()),
            float((1.0 - gt_valid).sum()),
        )
        soft_rb_mae = balanced_mean(metal_weighted_mae, nonmetal_weighted_mae)

        tp = int(np.logical_and(gt_binary, pred_binary).sum())
        fp = int(np.logical_and(~gt_binary, pred_binary).sum())
        fn = int(np.logical_and(gt_binary, ~pred_binary).sum())
        tn = int(np.logical_and(~gt_binary, ~pred_binary).sum())
        precision, recall, f1, iou, fpr = threshold_metrics_from_counts(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            positive_required_for_iou=True,
        )
        ap = average_precision_from_scores(gt_binary, pred_valid)

        image_rows.append(
            {
                "rb_mae": rb_mae,
                "soft_rb_mae": soft_rb_mae,
                "metal_weighted_mae": metal_weighted_mae,
                "nonmetal_weighted_mae": nonmetal_weighted_mae,
                "positive_mae": positive_mae,
                "remaining_mae": remaining_mae,
                "iou": iou,
                "ap": ap,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
                "gt_positive_ratio": safe_div(positive_pixels, pair.valid_pixels),
                "pred_positive_ratio": safe_div(int(pred_binary.sum()), pair.valid_pixels),
                "is_positive": positive_pixels > 0,
            }
        )

        acc = scene_accumulators.get(pair.scene_id)
        if acc is None:
            acc = SceneAccumulator(
                scene_id=pair.scene_id,
                split=pair.split,
                lighting=pair.lighting,
                cache_dir=scene_cache_dir,
            )
            scene_accumulators[pair.scene_id] = acc
        acc.update(
            gt_valid=gt_valid,
            pred_valid=pred_valid,
            gt_binary=gt_binary,
            pred_binary=pred_binary,
        )

    scene_metrics = [scene_accumulators[key].finalize() for key in sorted(scene_accumulators)]
    positive_images = [row for row in image_rows if row["is_positive"]]
    negative_images = [row for row in image_rows if not row["is_positive"]]

    image_level = {
        "count": len(image_rows),
        "rb_mae_mean": metric_mean([row["rb_mae"] for row in image_rows]),
        "soft_rb_mae_mean": metric_mean([row["soft_rb_mae"] for row in image_rows]),
        "metal_weighted_mae_mean": metric_mean([row["metal_weighted_mae"] for row in image_rows]),
        "nonmetal_weighted_mae_mean": metric_mean([row["nonmetal_weighted_mae"] for row in image_rows]),
        "positive_mae_mean": metric_mean([row["positive_mae"] for row in image_rows]),
        "remaining_mae_mean": metric_mean([row["remaining_mae"] for row in image_rows]),
        "iou_mean_positive_images_only": metric_mean([row["iou"] for row in positive_images]),
        "ap_mean_positive_images_only": metric_mean([row["ap"] for row in positive_images]),
        "precision_mean": metric_mean([row["precision"] for row in image_rows]),
        "recall_mean_positive_images_only": metric_mean([row["recall"] for row in positive_images]),
        "f1_mean_positive_images_only": metric_mean([row["f1"] for row in positive_images]),
        "negative_image_fpr_mean": metric_mean([row["fpr"] for row in negative_images]),
        "gt_positive_ratio_mean": metric_mean([row["gt_positive_ratio"] for row in image_rows]),
        "pred_positive_ratio_mean": metric_mean([row["pred_positive_ratio"] for row in image_rows]),
    }

    return {
        "config": {
            "gt_threshold": gt_threshold,
            "pred_threshold": pred_threshold,
        },
        "scene_level": summarize_scenes(scene_metrics),
        "image_level": image_level,
        "by_split": summarize_by_split(scene_metrics),
    }


def build_threshold_sensitivity(
    cached_pairs: Sequence[PairCacheEntry],
    gt_thresholds: Sequence[float],
    pred_thresholds: Optional[Sequence[float]],
    cache_dir: Optional[Path] = None,
) -> dict:
    if not gt_thresholds:
        return {}

    if pred_thresholds is None or len(pred_thresholds) == 0:
        paired_pred_thresholds = list(gt_thresholds)
    else:
        if len(pred_thresholds) != len(gt_thresholds):
            raise ValueError(
                "--sensitivity_pred_thresholds must either be omitted or have the same "
                "length as --sensitivity_gt_thresholds."
            )
        paired_pred_thresholds = list(pred_thresholds)

    results = {}
    for gt_threshold, pred_threshold in zip(gt_thresholds, paired_pred_thresholds):
        key = threshold_key(gt_threshold, pred_threshold)
        threshold_cache_dir = cache_dir / key if cache_dir is not None else None
        results[key] = summarize_threshold_metrics(
            cached_pairs=cached_pairs,
            gt_threshold=gt_threshold,
            pred_threshold=pred_threshold,
            cache_dir=threshold_cache_dir,
        )
    return results


def save_threshold_sensitivity_csv(path: Path, threshold_sensitivity: dict) -> None:
    rows = []
    for key, payload in threshold_sensitivity.items():
        config = payload["config"]
        scene_level = payload["scene_level"]
        rows.append(
            {
                "scope": "overall",
                "split": "all",
                "threshold_key": key,
                "gt_threshold": config["gt_threshold"],
                "pred_threshold": config["pred_threshold"],
                **scene_level,
            }
        )
        for split, split_payload in payload.get("by_split", {}).items():
            rows.append(
                {
                    "scope": "split",
                    "split": split,
                    "threshold_key": key,
                    "gt_threshold": config["gt_threshold"],
                    "pred_threshold": config["pred_threshold"],
                    **split_payload["overall"],
                }
            )
    save_csv(path, rows)


def bootstrap_scene_summary(scene_metrics: Sequence[SceneMetrics], iters: int, seed: int) -> dict:
    if iters <= 0 or not scene_metrics:
        return {}

    rng = np.random.default_rng(seed)
    n = len(scene_metrics)
    metric_names = [
        "mae_scene_macro",
        "rb_mae_scene_macro",
        "iou_positive_scene_macro",
        "ap_positive_scene_macro",
        "precision_positive_scene_macro",
        "recall_positive_scene_macro",
        "f1_positive_scene_macro",
        "negative_scene_fpr_macro",
    ]
    values: Dict[str, List[float]] = {name: [] for name in metric_names}

    for _ in range(iters):
        sample_indices = rng.integers(0, n, size=n)
        sampled = [scene_metrics[idx] for idx in sample_indices]
        summary = summarize_scenes(sampled)
        for name in metric_names:
            value = summary.get(name)
            if value is not None and np.isfinite(value):
                values[name].append(float(value))

    ci = {}
    for name, vals in values.items():
        if not vals:
            ci[name] = {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
            continue
        arr = np.asarray(vals, dtype=np.float64)
        ci[name] = {
            "mean": float(arr.mean()),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    return ci


def save_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir: Path, image_metrics: Sequence[ImageMetrics], scene_metrics: Sequence[SceneMetrics], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "metallic_image_metrics.csv", [asdict(m) for m in image_metrics])
    save_csv(output_dir / "metallic_scene_metrics.csv", [asdict(m) for m in scene_metrics])
    if summary.get("threshold_sensitivity"):
        save_threshold_sensitivity_csv(output_dir / "metallic_threshold_sensitivity.csv", summary["threshold_sensitivity"])
    with open(output_dir / "metallic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    baseline_value = args.constant_value
    if args.baseline == "constant" and baseline_value is None:
        raise ValueError("--baseline constant requires --constant_value")
    if args.baseline == "calibration_mean":
        calibration_root = args.calibration_gt_root or args.gt_root
        baseline_value = collect_calibration_mean(calibration_root, args.gt_threshold)

    pairs = collect_pairs(pred_root=args.pred_root, gt_root=args.gt_root, mask_fallback=args.mask_fallback)
    if not pairs:
        raise RuntimeError("No matched GT/prediction pairs found.")

    resized_pred_dir = args.output_dir / "resized_predictions" if args.save_resized_preds else None
    pixel_cache_dir = args.pixel_cache_dir or (args.output_dir / "_pixel_cache")
    pair_cache_dir = pixel_cache_dir / "pairs"
    main_scene_cache_dir = pixel_cache_dir / "scene_main"
    sensitivity_cache_dir = pixel_cache_dir / "threshold_sensitivity"
    pair_cache_dir.mkdir(parents=True, exist_ok=True)

    image_metrics: List[ImageMetrics] = []
    cached_pairs: List[PairCacheEntry] = []
    scene_accumulators: Dict[str, SceneAccumulator] = {}

    for pair_index, (gt_path, pred_path, mask_path, name, split, scene_id) in enumerate(pairs):
        item, gt_valid, pred_valid, gt_binary, pred_binary = evaluate_pair(
            gt_path=gt_path,
            pred_path=pred_path,
            mask_path=mask_path,
            name=name,
            split=split,
            scene_id=scene_id,
            pred_root=args.pred_root,
            resize_interp=args.resize_interp,
            gt_threshold=args.gt_threshold,
            pred_threshold=args.pred_threshold,
            baseline=args.baseline,
            baseline_value=baseline_value,
            resized_pred_dir=resized_pred_dir,
        )
        image_metrics.append(item)
        gt_cache_path = pair_cache_dir / f"{pair_index:08d}_gt.npy"
        pred_cache_path = pair_cache_dir / f"{pair_index:08d}_pred.npy"
        np.save(gt_cache_path, gt_valid.astype(np.float32, copy=False))
        np.save(pred_cache_path, pred_valid.astype(np.float32, copy=False))
        cached_pairs.append(
            PairCacheEntry(
                name=item.name,
                split=item.split,
                scene_id=item.scene_id,
                lighting=item.lighting,
                valid_pixels=item.valid_pixels,
                total_pixels=item.total_pixels,
                gt_valid_path=str(gt_cache_path),
                pred_valid_path=str(pred_cache_path),
            )
        )

        acc = scene_accumulators.get(scene_id)
        if acc is None:
            acc = SceneAccumulator(
                scene_id=scene_id,
                split=split,
                lighting=item.lighting,
                cache_dir=main_scene_cache_dir,
            )
            scene_accumulators[scene_id] = acc
        acc.update(gt_valid=gt_valid, pred_valid=pred_valid, gt_binary=gt_binary, pred_binary=pred_binary)

        print(
            f"[{item.name}] "
            f"MAE={item.mae:.6f} | RB-MAE={item.rb_mae:.6f} | "
            f"PosMAE={item.positive_mae:.6f} | RemMAE={item.remaining_mae:.6f} | "
            f"IoU@{args.pred_threshold:.2f}={item.iou_at_05:.6f} | "
            f"AP={item.ap:.6f} | Recall={item.recall_at_05:.6f} | "
            f"FPR={item.fpr_at_05:.6f}"
        )

    scene_metrics = [scene_accumulators[key].finalize() for key in sorted(scene_accumulators)]
    sensitivity = build_threshold_sensitivity(
        cached_pairs=cached_pairs,
        gt_thresholds=args.sensitivity_gt_thresholds,
        pred_thresholds=args.sensitivity_pred_thresholds,
        cache_dir=sensitivity_cache_dir,
    )

    summary = {
        "config": {
            "baseline": args.baseline,
            "baseline_value": baseline_value,
            "gt_threshold": args.gt_threshold,
            "pred_threshold": args.pred_threshold,
            "resize_interp": args.resize_interp,
            "mask_fallback": args.mask_fallback,
        },
        "main_table_recommendation": summarize_scenes(scene_metrics),
        "image_level": summarize_images(image_metrics),
        "scene_level": summarize_scenes(scene_metrics),
        "by_split": summarize_by_split(scene_metrics),
        "bootstrap_ci": bootstrap_scene_summary(
            scene_metrics=scene_metrics,
            iters=args.bootstrap_iters,
            seed=args.bootstrap_seed,
        ),
        "metric_groups": {
            "main_table": ["mae_scene_macro", "rb_mae_scene_macro", "iou_positive_scene_macro", "ap_positive_scene_macro"],
            "appendix_localization": [
                "soft_rb_mae_scene_macro",
                "metal_weighted_mae_scene_macro",
                "nonmetal_weighted_mae_scene_macro",
                "positive_mae_scene_macro",
                "remaining_mae_scene_macro",
                "precision_positive_scene_macro",
                "recall_positive_scene_macro",
                "f1_positive_scene_macro",
                "negative_scene_fpr_macro",
            ],
            "auxiliary_visual": ["psnr_mean", "ssim_mean"],
        },
        "threshold_sensitivity": sensitivity,
        "notes": {
            "rb_mae": "RB-MAE = 0.5 * (positive-region MAE + remaining-region MAE), where GT positivity is defined by gt >= gt_threshold.",
            "soft_rb_mae": (
                "Soft-RB-MAE = 0.5 * (Metal-weighted MAE + Nonmetal-weighted MAE), "
                "where metal weights use GT metallic y and nonmetal weights use (1 - y), "
                "avoiding a hard GT threshold."
            ),
            "iou_ap_aggregation": "IoU and AP are reported as macro-averages over positive scenes after concatenating all retained views within each scene.",
            "negative_scenes": "Negative scenes are not given IoU=1 when both GT and prediction are empty; they contribute via negative-scene FPR instead.",
            "ap": "AP uses continuous prediction values as ranking scores and GT >= gt_threshold as positives.",
            "threshold_sensitivity": (
                "If sensitivity_gt_thresholds is provided, threshold_sensitivity reports the same "
                "region-balanced and localization metrics under multiple disclosed operational thresholds."
            ),
            "mask": (
                "Metrics are computed over the resolved valid-pixel mask. "
                "By default the script uses MetallicEvalMask when present and falls back to ObjectSegmentation."
            ),
        },
    }

    save_results(args.output_dir, image_metrics, scene_metrics, summary)

    if not args.keep_pixel_cache:
        shutil.rmtree(pixel_cache_dir, ignore_errors=True)

    print("\n=== Main Table Recommendation ===")
    print(json.dumps(summary["main_table_recommendation"], indent=2, ensure_ascii=False))
    if summary["threshold_sensitivity"]:
        print("\n=== Threshold Sensitivity ===")
        print(json.dumps(summary["threshold_sensitivity"], indent=2, ensure_ascii=False))
    if summary["bootstrap_ci"]:
        print("\n=== Scene Bootstrap 95% CI ===")
        print(json.dumps(summary["bootstrap_ci"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
