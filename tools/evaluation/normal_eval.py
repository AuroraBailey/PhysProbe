#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate predicted normal maps against GT normal maps.

Metrics:
- Mean Angular Error (degree)
- Median Angular Error (degree)
- RMSE Angular Error (degree)
- Acc@5 / Acc@7.5 / Acc@11.25 / Acc@22.5 / Acc@30

Predictions are resized to GT resolution before evaluation.
Normal RGB is decoded as n = 2 * RGB - 1, then L2-normalized.

This version reports only the fixed variant for each pair.

The fixed prediction x-axis flip is selected by model name from the prediction
path: Lotus and StableNormal are flipped on x, while other models are not
flipped. Fixed affine fitting is shared within batches of pairs instead of
being re-fit independently for every single pair. Use --batch_size to control
the batch size.
"""

from __future__ import annotations
try:
    from .mask_policy import resolve_required_mask
except ImportError:
    from mask_policy import resolve_required_mask


import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PRED_X_FLIP_MODELS = {"Lotus", "StableNormal"}


@dataclass
class NormalMetrics:
    mean_angular: float
    median_angular: float
    rmse_angular: float
    acc_5: float
    acc_7_5: float
    acc_11_25: float
    acc_22_5: float
    acc_30: float
    valid_pixels: int


@dataclass
class EvalResult:
    name: str
    strategy: str
    width: int
    height: int
    total_pixels: int
    valid_pixels: int
    valid_ratio: float
    pred_flip_x: bool
    pred_flip_y: bool
    pred_flip_z: bool
    affine_mode: str
    affine_global_scale: float
    affine_global_shift: float
    affine_x_scale: float
    affine_x_shift: float
    affine_y_scale: float
    affine_y_shift: float
    affine_z_scale: float
    affine_z_shift: float
    rotation_r00: float
    rotation_r01: float
    rotation_r02: float
    rotation_r10: float
    rotation_r11: float
    rotation_r12: float
    rotation_r20: float
    rotation_r21: float
    rotation_r22: float
    gt_black_ratio: float
    pred_black_ratio: float
    gt_norm_mean: float
    gt_norm_std: float
    pred_norm_mean_pre: float
    pred_norm_std_pre: float
    pred_norm_mean_post: float
    pred_norm_std_post: float
    mean_angular: float
    median_angular: float
    rmse_angular: float
    acc_5: float
    acc_7_5: float
    acc_11_25: float
    acc_22_5: float
    acc_30: float
    sum_angular: float
    sum_sq_angular: float
    count_acc_5: int
    count_acc_7_5: int
    count_acc_11_25: int
    count_acc_22_5: int
    count_acc_30: int


@dataclass
class PreparedPair:
    name: str
    width: int
    height: int
    gt_unit: np.ndarray
    pred_raw: np.ndarray
    valid_base: np.ndarray
    gt_black_ratio: float
    pred_black_ratio: float
    gt_norm_stats: Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted normal maps against GT normal maps."
    )

    parser.add_argument("--gt_path", type=str, help="Single GT normal path.")
    parser.add_argument("--pred_path", type=str, help="Single predicted normal path.")
    parser.add_argument("--mask_path", type=str, help="Optional single valid-mask path.")

    parser.add_argument("--gt_dir", type=str, help="GT normal directory.")
    parser.add_argument("--pred_dir", type=str, help="Prediction directory.")
    parser.add_argument("--mask_dir", type=str, help="Optional valid-mask directory.")

    parser.add_argument("--pair_mode", type=str, default="stem", choices=["stem", "name"])
    parser.add_argument("--gt_suffix_to_strip", type=str, default="")
    parser.add_argument("--pred_suffix_to_strip", type=str, default="")
    parser.add_argument("--mask_suffix_to_strip", type=str, default="")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_resized_preds", action="store_true")

    parser.add_argument("--recursive", dest="recursive", action="store_true")
    parser.add_argument("--no_recursive", dest="recursive", action="store_false")
    parser.set_defaults(recursive=True)

    parser.add_argument(
        "--resize_interp",
        type=str,
        default="nearest",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
    )
    parser.add_argument("--mask_threshold", type=float, default=0.5)

    parser.add_argument(
        "--normal_encoding",
        type=str,
        default="rgb01",
        choices=["rgb01"],
        help="Decode RGB [0,1] as normal = 2*RGB - 1.",
    )
    parser.add_argument("--gt_flip_x", action="store_true")
    parser.add_argument("--gt_flip_y", action="store_true")
    parser.add_argument("--gt_flip_z", action="store_true")

    parser.add_argument("--min_norm", type=float, default=1e-6)
    parser.add_argument("--ignore_back_facing_gt", action="store_true")

    parser.add_argument(
        "--black_threshold",
        type=int,
        default=4,
        help="Treat RGB <= threshold on all channels as invalid black background.",
    )
    parser.add_argument("--ignore_black_gt", dest="ignore_black_gt", action="store_true")
    parser.add_argument("--no_ignore_black_gt", dest="ignore_black_gt", action="store_false")
    parser.add_argument("--ignore_black_pred", dest="ignore_black_pred", action="store_true")
    parser.add_argument("--no_ignore_black_pred", dest="ignore_black_pred", action="store_false")
    parser.set_defaults(ignore_black_gt=True, ignore_black_pred=True)
    parser.add_argument(
        "--erode_valid_pixels",
        type=int,
        default=0,
        help="Erode the valid evaluation region by N pixels to reduce boundary noise.",
    )

    parser.add_argument(
        "--fixed_affine_mode",
        type=str,
        default="none",
        choices=["none", "global", "per_axis"],
        help="Affine mode for fixed-flip evaluation. Fitting is shared within each batch.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="Run fixed shared affine fitting once per N pairs.",
    )
    parser.add_argument("--best_search_batch_size", type=int, dest="batch_size", help=argparse.SUPPRESS)

    parser.add_argument("--disable_mask", action="store_true", help="Explicitly evaluate without a valid-pixel mask.")
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


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def image_to_float_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(path: Path, size: Tuple[int, int], threshold: float) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != size:
            img = img.resize(size, Image.Resampling.NEAREST)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr >= threshold


def rgb_to_normal_raw(img_np: np.ndarray) -> np.ndarray:
    return img_np * 2.0 - 1.0


def flip_normals(raw_n: np.ndarray, flips: Tuple[bool, bool, bool]) -> np.ndarray:
    out = raw_n.copy()
    if flips[0]:
        out[..., 0] *= -1.0
    if flips[1]:
        out[..., 1] *= -1.0
    if flips[2]:
        out[..., 2] *= -1.0
    return out


def unitize_normals(raw_n: np.ndarray, min_norm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(raw_n).all(axis=2)
    norm = np.linalg.norm(raw_n, axis=2)
    legal = finite & (norm > min_norm)

    unit_n = np.zeros_like(raw_n, dtype=np.float32)
    unit_n[legal] = raw_n[legal] / norm[legal, None]
    return unit_n, legal, norm.astype(np.float32)


def black_mask(img_np: np.ndarray, threshold: int) -> np.ndarray:
    thr = float(threshold) / 255.0
    return np.all(img_np <= thr, axis=2)


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask

    out = mask.copy()
    for _ in range(pixels):
        nxt = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.zeros_like(out)
                ys = slice(max(0, dy), out.shape[0] + min(0, dy))
                xs = slice(max(0, dx), out.shape[1] + min(0, dx))
                y2 = slice(max(0, -dy), out.shape[0] - max(0, dy))
                x2 = slice(max(0, -dx), out.shape[1] - max(0, dx))
                shifted[ys, xs] = out[y2, x2]
                nxt &= shifted
        out = nxt
    return out


def angular_values_deg(gt_n: np.ndarray, pred_n: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if valid_mask.sum() == 0:
        return np.asarray([], dtype=np.float32)
    dot = np.sum(gt_n * pred_n, axis=2)
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.degrees(np.arccos(dot)).astype(np.float32)
    return ang[valid_mask]


def compute_normal_metrics(
    gt_n: np.ndarray,
    pred_n: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[NormalMetrics, float, float, Dict[str, int]]:
    angles = angular_values_deg(gt_n, pred_n, valid_mask)
    valid_pixels = int(angles.size)

    if valid_pixels == 0:
        nan = float("nan")
        return (
            NormalMetrics(nan, nan, nan, nan, nan, nan, nan, nan, 0),
            0.0,
            0.0,
            {"acc_5": 0, "acc_7_5": 0, "acc_11_25": 0, "acc_22_5": 0, "acc_30": 0},
        )

    counts = {
        "acc_5": int(np.sum(angles < 5.0)),
        "acc_7_5": int(np.sum(angles < 7.5)),
        "acc_11_25": int(np.sum(angles < 11.25)),
        "acc_22_5": int(np.sum(angles < 22.5)),
        "acc_30": int(np.sum(angles < 30.0)),
    }

    metrics = NormalMetrics(
        mean_angular=float(np.mean(angles)),
        median_angular=float(np.median(angles)),
        rmse_angular=float(np.sqrt(np.mean(angles ** 2))),
        acc_5=float(counts["acc_5"] / valid_pixels),
        acc_7_5=float(counts["acc_7_5"] / valid_pixels),
        acc_11_25=float(counts["acc_11_25"] / valid_pixels),
        acc_22_5=float(counts["acc_22_5"] / valid_pixels),
        acc_30=float(counts["acc_30"] / valid_pixels),
        valid_pixels=valid_pixels,
    )
    return metrics, float(np.sum(angles)), float(np.sum(angles ** 2)), counts


def normalize_match_key(name: str, suffix_to_strip: str) -> str:
    stem = Path(name).stem
    if suffix_to_strip and stem.endswith(suffix_to_strip):
        stem = stem[: -len(suffix_to_strip)]
    return stem


def list_images(directory: Path, recursive: bool) -> List[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted([p for p in iterator if p.is_file() and p.suffix.lower() in IMG_EXTS])


def build_pairs_from_dirs(
    gt_dir: Path,
    pred_dir: Path,
    mask_dir: Optional[Path],
    pair_mode: str,
    gt_suffix_to_strip: str,
    pred_suffix_to_strip: str,
    mask_suffix_to_strip: str,
    recursive: bool,
) -> Tuple[List[Tuple[Path, Path, Optional[Path]]], Dict[str, object]]:
    gt_files = list_images(gt_dir, recursive)
    pred_files = list_images(pred_dir, recursive)
    mask_files = list_images(mask_dir, recursive) if mask_dir is not None else []

    def make_key(path: Path, root: Path, suffix_to_strip: str) -> str:
        rel = path.relative_to(root)
        if pair_mode == "name":
            return rel.as_posix()
        stem = normalize_match_key(path.name, suffix_to_strip)
        parent_parts = [part for part in rel.parent.parts if part not in {"normal", "normal_vis"}]
        if not parent_parts:
            return stem
        return (Path(*parent_parts) / stem).as_posix()

    gt_map = {make_key(p, gt_dir, gt_suffix_to_strip): p for p in gt_files}
    pred_map = {make_key(p, pred_dir, pred_suffix_to_strip): p for p in pred_files}
    mask_map = {make_key(p, mask_dir, mask_suffix_to_strip): p for p in mask_files} if mask_files else {}

    gt_keys = set(gt_map.keys())
    pred_keys = set(pred_map.keys())
    mask_keys = set(mask_map.keys())
    common_keys = sorted(gt_keys & pred_keys)

    pairs = [(gt_map[k], pred_map[k], mask_map.get(k) if mask_map else None) for k in common_keys]
    pairing_info = {
        "gt_files": len(gt_files),
        "pred_files": len(pred_files),
        "mask_files": len(mask_files),
        "matched_pairs": len(pairs),
        "gt_only_count": len(gt_keys - pred_keys),
        "pred_only_count": len(pred_keys - gt_keys),
        "mask_only_count": len(mask_keys - gt_keys) if mask_map else 0,
        "gt_only_examples": sorted(list(gt_keys - pred_keys))[:10],
        "pred_only_examples": sorted(list(pred_keys - gt_keys))[:10],
    }
    return pairs, pairing_info


def apply_global_affine(raw_n: np.ndarray, scale: float, shift: float) -> np.ndarray:
    return raw_n * scale + shift


def apply_per_axis_affine(raw_n: np.ndarray, params: Sequence[Tuple[float, float]]) -> np.ndarray:
    out = raw_n.copy()
    for axis, (scale, shift) in enumerate(params):
        out[..., axis] = out[..., axis] * scale + shift
    return out


def apply_rotation(raw_n: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return raw_n @ rotation


def summarize_norms(norm_map: np.ndarray, valid_mask: np.ndarray) -> Tuple[float, float]:
    values = norm_map[valid_mask]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def make_batch_total() -> Dict[str, float]:
    return {"sum_angular": 0.0, "valid_pixels": 0}


def update_batch_total(total: Dict[str, float], sum_angular: float, valid_pixels: int) -> None:
    total["sum_angular"] += float(sum_angular)
    total["valid_pixels"] += int(valid_pixels)


def update_batch_total_from_result(total: Dict[str, float], result: EvalResult) -> None:
    update_batch_total(total, result.sum_angular, result.valid_pixels)


def choose_best_batch_index(totals: Sequence[Dict[str, float]]) -> int:
    return min(
        range(len(totals)),
        key=lambda idx: (
            float("inf")
            if totals[idx]["valid_pixels"] <= 0
            else float(totals[idx]["sum_angular"] / totals[idx]["valid_pixels"]),
            -int(totals[idx]["valid_pixels"]),
        ),
    )


def accumulate_global_affine_stats(stats: np.ndarray, pred_v: np.ndarray, gt_v: np.ndarray) -> None:
    x = pred_v.reshape(-1).astype(np.float64, copy=False)
    y = gt_v.reshape(-1).astype(np.float64, copy=False)
    stats[0] += float(np.dot(x, x))
    stats[1] += float(np.sum(x))
    stats[2] += float(x.size)
    stats[3] += float(np.dot(x, y))
    stats[4] += float(np.sum(y))


def solve_global_affine_from_stats(stats: np.ndarray) -> Tuple[float, float]:
    if stats[2] <= 0:
        return float("nan"), float("nan")

    design = np.asarray(
        [[stats[0], stats[1]], [stats[1], stats[2]]],
        dtype=np.float64,
    )
    target = np.asarray([stats[3], stats[4]], dtype=np.float64)
    scale, shift = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(scale), float(shift)


def accumulate_per_axis_affine_stats(stats: np.ndarray, pred_v: np.ndarray, gt_v: np.ndarray) -> None:
    for axis in range(3):
        x = pred_v[:, axis].astype(np.float64, copy=False)
        y = gt_v[:, axis].astype(np.float64, copy=False)
        stats[axis, 0] += float(np.dot(x, x))
        stats[axis, 1] += float(np.sum(x))
        stats[axis, 2] += float(x.size)
        stats[axis, 3] += float(np.dot(x, y))
        stats[axis, 4] += float(np.sum(y))


def solve_per_axis_affine_from_stats(stats: np.ndarray) -> List[Tuple[float, float]]:
    params: List[Tuple[float, float]] = []
    for axis in range(3):
        if stats[axis, 2] <= 0:
            params.append((float("nan"), float("nan")))
            continue

        design = np.asarray(
            [[stats[axis, 0], stats[axis, 1]], [stats[axis, 1], stats[axis, 2]]],
            dtype=np.float64,
        )
        target = np.asarray([stats[axis, 3], stats[axis, 4]], dtype=np.float64)
        scale, shift = np.linalg.lstsq(design, target, rcond=None)[0]
        params.append((float(scale), float(shift)))
    return params


def accumulate_rotation_stats(stats: np.ndarray, pred_v: np.ndarray, gt_v: np.ndarray) -> None:
    stats += pred_v.astype(np.float64, copy=False).T @ gt_v.astype(np.float64, copy=False)


def solve_rotation_from_stats(stats: np.ndarray) -> np.ndarray:
    if not np.any(stats):
        return np.eye(3, dtype=np.float64)

    u, _, vt = np.linalg.svd(stats, full_matrices=False)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    return u @ correction @ vt


def evaluate_candidate(
    *,
    name: str,
    strategy: str,
    width: int,
    height: int,
    gt_unit: np.ndarray,
    pred_raw_post: np.ndarray,
    valid_base: np.ndarray,
    flips: Tuple[bool, bool, bool],
    affine_mode: str,
    affine_global_params: Tuple[float, float],
    affine_axis_params: Sequence[Tuple[float, float]],
    rotation_matrix: Optional[np.ndarray],
    gt_black_ratio: float,
    pred_black_ratio: float,
    gt_norm_stats: Tuple[float, float],
    pred_pre_stats: Tuple[float, float],
    min_norm: float,
) -> EvalResult:
    pred_unit, pred_legal_post, pred_norm_post = unitize_normals(pred_raw_post, min_norm=min_norm)
    valid = valid_base & pred_legal_post
    metrics, sum_ang, sum_sq_ang, counts = compute_normal_metrics(gt_unit, pred_unit, valid)
    pred_post_stats = summarize_norms(pred_norm_post, valid)

    axis_params = list(affine_axis_params)
    while len(axis_params) < 3:
        axis_params.append((float("nan"), float("nan")))

    if rotation_matrix is None:
        rotation_values = [float("nan")] * 9
    else:
        rotation_values = [float(v) for v in rotation_matrix.reshape(-1)]

    return EvalResult(
        name=name,
        strategy=strategy,
        width=width,
        height=height,
        total_pixels=width * height,
        valid_pixels=metrics.valid_pixels,
        valid_ratio=float(metrics.valid_pixels / (width * height)),
        pred_flip_x=flips[0],
        pred_flip_y=flips[1],
        pred_flip_z=flips[2],
        affine_mode=affine_mode,
        affine_global_scale=float(affine_global_params[0]),
        affine_global_shift=float(affine_global_params[1]),
        affine_x_scale=float(axis_params[0][0]),
        affine_x_shift=float(axis_params[0][1]),
        affine_y_scale=float(axis_params[1][0]),
        affine_y_shift=float(axis_params[1][1]),
        affine_z_scale=float(axis_params[2][0]),
        affine_z_shift=float(axis_params[2][1]),
        rotation_r00=rotation_values[0],
        rotation_r01=rotation_values[1],
        rotation_r02=rotation_values[2],
        rotation_r10=rotation_values[3],
        rotation_r11=rotation_values[4],
        rotation_r12=rotation_values[5],
        rotation_r20=rotation_values[6],
        rotation_r21=rotation_values[7],
        rotation_r22=rotation_values[8],
        gt_black_ratio=float(gt_black_ratio),
        pred_black_ratio=float(pred_black_ratio),
        gt_norm_mean=float(gt_norm_stats[0]),
        gt_norm_std=float(gt_norm_stats[1]),
        pred_norm_mean_pre=float(pred_pre_stats[0]),
        pred_norm_std_pre=float(pred_pre_stats[1]),
        pred_norm_mean_post=float(pred_post_stats[0]),
        pred_norm_std_post=float(pred_post_stats[1]),
        mean_angular=metrics.mean_angular,
        median_angular=metrics.median_angular,
        rmse_angular=metrics.rmse_angular,
        acc_5=metrics.acc_5,
        acc_7_5=metrics.acc_7_5,
        acc_11_25=metrics.acc_11_25,
        acc_22_5=metrics.acc_22_5,
        acc_30=metrics.acc_30,
        sum_angular=float(sum_ang),
        sum_sq_angular=float(sum_sq_ang),
        count_acc_5=int(counts["acc_5"]),
        count_acc_7_5=int(counts["acc_7_5"]),
        count_acc_11_25=int(counts["acc_11_25"]),
        count_acc_22_5=int(counts["acc_22_5"]),
        count_acc_30=int(counts["acc_30"]),
    )


def prepare_pair(
    gt_path: Path,
    pred_path: Path,
    mask_path: Optional[Path],
    args: argparse.Namespace,
    resized_pred_dir: Optional[Path],
    name: str,
) -> PreparedPair:
    mask_path = resolve_required_mask(gt_path, mask_path, disabled=getattr(args, "disable_mask", False))
    gt_img = load_rgb(gt_path)
    pred_img = load_rgb(pred_path)

    if pred_img.size != gt_img.size:
        pred_img = pred_img.resize(gt_img.size, pil_interp_mode(args.resize_interp))

    if resized_pred_dir is not None:
        resized_pred_dir.mkdir(parents=True, exist_ok=True)
        pred_img.save(resized_pred_dir / pred_path.name)

    gt_np = image_to_float_np(gt_img)
    pred_np = image_to_float_np(pred_img)

    gt_raw = rgb_to_normal_raw(gt_np)
    pred_raw = rgb_to_normal_raw(pred_np)

    gt_raw = flip_normals(gt_raw, (args.gt_flip_x, args.gt_flip_y, args.gt_flip_z))
    gt_unit, gt_legal, gt_norm = unitize_normals(gt_raw, min_norm=args.min_norm)
    _, pred_legal, _ = unitize_normals(pred_raw, min_norm=args.min_norm)

    if mask_path is not None and mask_path.exists():
        mask = load_mask(mask_path, gt_img.size, args.mask_threshold)
    else:
        mask = np.ones((gt_np.shape[0], gt_np.shape[1]), dtype=bool)

    gt_black = black_mask(gt_np, args.black_threshold) if args.ignore_black_gt else np.zeros(mask.shape, dtype=bool)
    pred_black = black_mask(pred_np, args.black_threshold) if args.ignore_black_pred else np.zeros(mask.shape, dtype=bool)

    valid_base = mask & gt_legal & pred_legal & (~gt_black) & (~pred_black)
    if args.ignore_back_facing_gt:
        valid_base &= gt_unit[..., 2] > 0
    valid_base = erode_mask(valid_base, args.erode_valid_pixels)

    gt_norm_stats = summarize_norms(gt_norm, valid_base)
    gt_black_ratio = float(np.mean(gt_black))
    pred_black_ratio = float(np.mean(pred_black))

    return PreparedPair(
        name=name,
        width=gt_img.size[0],
        height=gt_img.size[1],
        gt_unit=gt_unit,
        pred_raw=pred_raw,
        valid_base=valid_base,
        gt_black_ratio=gt_black_ratio,
        pred_black_ratio=pred_black_ratio,
        gt_norm_stats=gt_norm_stats,
    )


def evaluate_prepared_candidate(
    prepared: PreparedPair,
    *,
    strategy: str,
    flips: Tuple[bool, bool, bool],
    affine_mode: str,
    affine_global_params: Tuple[float, float],
    affine_axis_params: Sequence[Tuple[float, float]],
    rotation_matrix: Optional[np.ndarray],
    args: argparse.Namespace,
) -> EvalResult:
    flipped_raw = flip_normals(prepared.pred_raw, flips)
    pred_pre_stats = summarize_norms(np.linalg.norm(flipped_raw, axis=2), prepared.valid_base)

    pred_raw_post = flipped_raw
    if affine_mode == "global":
        pred_raw_post = apply_global_affine(flipped_raw, affine_global_params[0], affine_global_params[1])
    elif affine_mode == "per_axis":
        pred_raw_post = apply_per_axis_affine(flipped_raw, affine_axis_params)
    elif affine_mode == "rotation":
        if rotation_matrix is None:
            raise ValueError("rotation_matrix must be provided when affine_mode is 'rotation'.")
        pred_raw_post = apply_rotation(flipped_raw, rotation_matrix)

    return evaluate_candidate(
        name=prepared.name,
        strategy=strategy,
        width=prepared.width,
        height=prepared.height,
        gt_unit=prepared.gt_unit,
        pred_raw_post=pred_raw_post,
        valid_base=prepared.valid_base,
        flips=flips,
        affine_mode=affine_mode,
        affine_global_params=affine_global_params,
        affine_axis_params=affine_axis_params,
        rotation_matrix=rotation_matrix,
        gt_black_ratio=prepared.gt_black_ratio,
        pred_black_ratio=prepared.pred_black_ratio,
        gt_norm_stats=prepared.gt_norm_stats,
        pred_pre_stats=pred_pre_stats,
        min_norm=args.min_norm,
    )


def iter_batches(
    items: Sequence[Tuple[str, Path, Path, Optional[Path]]],
    batch_size: int,
) -> List[Sequence[Tuple[str, Path, Path, Optional[Path]]]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def fixed_pred_flips_for_path(pred_path: Path) -> Tuple[bool, bool, bool]:
    return (any(part in PRED_X_FLIP_MODELS for part in pred_path.parts), False, False)


def evaluate_batch(
    batch: Sequence[Tuple[str, Path, Path, Optional[Path]]],
    args: argparse.Namespace,
    resized_pred_dir: Optional[Path],
) -> List[EvalResult]:
    global_fit_stats = np.zeros(5, dtype=np.float64)
    axis_fit_stats = np.zeros((3, 5), dtype=np.float64)

    if args.fixed_affine_mode != "none":
        for name, gt_path, pred_path, mask_path in batch:
            prepared = prepare_pair(gt_path, pred_path, mask_path, args, None, name)
            if int(prepared.valid_base.sum()) == 0:
                continue
            fixed_flips = fixed_pred_flips_for_path(pred_path)
            fixed_raw = flip_normals(prepared.pred_raw, fixed_flips)
            pred_valid = fixed_raw[prepared.valid_base]
            gt_valid = prepared.gt_unit[prepared.valid_base]
            accumulate_global_affine_stats(global_fit_stats, pred_valid, gt_valid)
            accumulate_per_axis_affine_stats(axis_fit_stats, pred_valid, gt_valid)

    fixed_global_params = solve_global_affine_from_stats(global_fit_stats)
    fixed_axis_params = solve_per_axis_affine_from_stats(axis_fit_stats)
    fixed_affine_mode = args.fixed_affine_mode
    fixed_affine_global_params = (
        fixed_global_params if fixed_affine_mode == "global" else (float("nan"), float("nan"))
    )
    fixed_affine_axis_params = fixed_axis_params if fixed_affine_mode == "per_axis" else []

    results: List[EvalResult] = []
    for name, gt_path, pred_path, mask_path in batch:
        prepared = prepare_pair(gt_path, pred_path, mask_path, args, resized_pred_dir, name)
        fixed_flips = fixed_pred_flips_for_path(pred_path)
        results.append(
            evaluate_prepared_candidate(
                prepared,
                strategy="fixed",
                flips=fixed_flips,
                affine_mode=fixed_affine_mode,
                affine_global_params=fixed_affine_global_params,
                affine_axis_params=fixed_affine_axis_params,
                rotation_matrix=None,
                args=args,
            )
        )
    return results


def summarize(results: Sequence[EvalResult]) -> Dict[str, float]:
    def mean_valid(values: List[float]) -> float:
        vals = [float(v) for v in values if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def median_valid(values: List[float]) -> float:
        vals = [float(v) for v in values if np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    def std_valid(values: List[float]) -> float:
        vals = [float(v) for v in values if np.isfinite(v)]
        return float(np.std(vals)) if vals else float("nan")

    total_valid = int(sum(r.valid_pixels for r in results))
    total_sum_ang = float(sum(r.sum_angular for r in results))
    total_sum_sq_ang = float(sum(r.sum_sq_angular for r in results))
    total_acc_5 = int(sum(r.count_acc_5 for r in results))
    total_acc_7_5 = int(sum(r.count_acc_7_5 for r in results))
    total_acc_11_25 = int(sum(r.count_acc_11_25 for r in results))
    total_acc_22_5 = int(sum(r.count_acc_22_5 for r in results))
    total_acc_30 = int(sum(r.count_acc_30 for r in results))

    if total_valid > 0:
        mean_micro = total_sum_ang / total_valid
        rmse_micro = math.sqrt(total_sum_sq_ang / total_valid)
        acc_5_micro = total_acc_5 / total_valid
        acc_7_5_micro = total_acc_7_5 / total_valid
        acc_11_25_micro = total_acc_11_25 / total_valid
        acc_22_5_micro = total_acc_22_5 / total_valid
        acc_30_micro = total_acc_30 / total_valid
    else:
        mean_micro = rmse_micro = float("nan")
        acc_5_micro = acc_7_5_micro = acc_11_25_micro = acc_22_5_micro = acc_30_micro = float("nan")

    return {
        "count": len(results),
        "valid_pixels_total": total_valid,
        "mean_angular_mean": mean_valid([r.mean_angular for r in results]),
        "median_angular_mean": mean_valid([r.median_angular for r in results]),
        "rmse_angular_mean": mean_valid([r.rmse_angular for r in results]),
        "acc_5_mean": mean_valid([r.acc_5 for r in results]),
        "acc_7_5_mean": mean_valid([r.acc_7_5 for r in results]),
        "acc_11_25_mean": mean_valid([r.acc_11_25 for r in results]),
        "acc_22_5_mean": mean_valid([r.acc_22_5 for r in results]),
        "acc_30_mean": mean_valid([r.acc_30 for r in results]),
        "mean_angular_median": median_valid([r.mean_angular for r in results]),
        "mean_angular_std": std_valid([r.mean_angular for r in results]),
        "valid_ratio_mean": mean_valid([r.valid_ratio for r in results]),
        "gt_black_ratio_mean": mean_valid([r.gt_black_ratio for r in results]),
        "pred_black_ratio_mean": mean_valid([r.pred_black_ratio for r in results]),
        "pred_norm_mean_pre": mean_valid([r.pred_norm_mean_pre for r in results]),
        "pred_norm_mean_post": mean_valid([r.pred_norm_mean_post for r in results]),
        "mean_angular_micro": float(mean_micro),
        "rmse_angular_micro": float(rmse_micro),
        "acc_5_micro": float(acc_5_micro),
        "acc_7_5_micro": float(acc_7_5_micro),
        "acc_11_25_micro": float(acc_11_25_micro),
        "acc_22_5_micro": float(acc_22_5_micro),
        "acc_30_micro": float(acc_30_micro),
    }


def split_results_by_strategy(results: Sequence[EvalResult]) -> Dict[str, List[EvalResult]]:
    grouped: Dict[str, List[EvalResult]] = {}
    for result in results:
        grouped.setdefault(result.strategy, []).append(result)
    return grouped


def save_results(
    output_dir: Path,
    results: Sequence[EvalResult],
    summaries: Dict[str, Dict[str, float]],
    pairing_info: Optional[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "normal_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "strategy",
                "width",
                "height",
                "total_pixels",
                "valid_pixels",
                "valid_ratio",
                "pred_flip_x",
                "pred_flip_y",
                "pred_flip_z",
                "affine_mode",
                "affine_global_scale",
                "affine_global_shift",
                "affine_x_scale",
                "affine_x_shift",
                "affine_y_scale",
                "affine_y_shift",
                "affine_z_scale",
                "affine_z_shift",
                "rotation_r00",
                "rotation_r01",
                "rotation_r02",
                "rotation_r10",
                "rotation_r11",
                "rotation_r12",
                "rotation_r20",
                "rotation_r21",
                "rotation_r22",
                "gt_black_ratio",
                "pred_black_ratio",
                "gt_norm_mean",
                "gt_norm_std",
                "pred_norm_mean_pre",
                "pred_norm_std_pre",
                "pred_norm_mean_post",
                "pred_norm_std_post",
                "mean_angular",
                "median_angular",
                "rmse_angular",
                "acc_5",
                "acc_7_5",
                "acc_11_25",
                "acc_22_5",
                "acc_30",
            ],
        )
        writer.writeheader()
        for result in results:
            row = asdict(result)
            for hidden_key in [
                "sum_angular",
                "sum_sq_angular",
                "count_acc_5",
                "count_acc_7_5",
                "count_acc_11_25",
                "count_acc_22_5",
                "count_acc_30",
            ]:
                row.pop(hidden_key, None)
            writer.writerow(row)

    payload = {
        "config": {
            "pred_x_flip_models": sorted(PRED_X_FLIP_MODELS),
            "fixed_affine_mode": args.fixed_affine_mode,
            "black_threshold": args.black_threshold,
            "ignore_black_gt": args.ignore_black_gt,
            "ignore_black_pred": args.ignore_black_pred,
            "erode_valid_pixels": args.erode_valid_pixels,
            "ignore_back_facing_gt": args.ignore_back_facing_gt,
            "resize_interp": args.resize_interp,
            "mask_enabled": not args.disable_mask,
            "batch_size": args.batch_size,
        },
        "pairing_info": pairing_info,
        "summaries": summaries,
        "results": [asdict(r) for r in results],
    }
    with open(output_dir / "normal_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_result(result: EvalResult) -> None:
    print(
        f"[{result.name}][{result.strategy}] mean={result.mean_angular:.4f} | "
        f"median={result.median_angular:.4f} | rmse={result.rmse_angular:.4f} | "
        f"acc11.25={result.acc_11_25:.4f} | acc22.5={result.acc_22_5:.4f} | "
        f"acc30={result.acc_30:.4f} | valid={result.valid_pixels} | "
        f"flip=({int(result.pred_flip_x)},{int(result.pred_flip_y)},{int(result.pred_flip_z)}) | "
        f"affine={result.affine_mode}"
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer.")

    single_mode = bool(args.gt_path and args.pred_path)
    dir_mode = bool(args.gt_dir and args.pred_dir)
    if single_mode == dir_mode:
        raise ValueError("Use either single-pair mode (--gt_path/--pred_path) or directory mode (--gt_dir/--pred_dir).")

    output_dir = Path(args.output_dir)
    resized_pred_dir = output_dir / "resized_predictions" if args.save_resized_preds else None
    pairing_info: Optional[Dict[str, object]] = None

    if single_mode:
        pairs = [(Path(args.gt_path), Path(args.pred_path), Path(args.mask_path) if args.mask_path else None)]
        gt_dir = None
    else:
        gt_dir = Path(args.gt_dir)
        pred_dir = Path(args.pred_dir)
        mask_dir = Path(args.mask_dir) if args.mask_dir else None
        pairs, pairing_info = build_pairs_from_dirs(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            mask_dir=mask_dir,
            pair_mode=args.pair_mode,
            gt_suffix_to_strip=args.gt_suffix_to_strip,
            pred_suffix_to_strip=args.pred_suffix_to_strip,
            mask_suffix_to_strip=args.mask_suffix_to_strip,
            recursive=args.recursive,
        )
        if len(pairs) == 0:
            raise RuntimeError("No matched GT/prediction pairs found.")

    named_pairs = [
        (
            gt_path.stem if single_mode else gt_path.relative_to(gt_dir).as_posix(),
            gt_path,
            pred_path,
            mask_path,
        )
        for gt_path, pred_path, mask_path in pairs
    ]

    all_results: List[EvalResult] = []
    for batch in iter_batches(named_pairs, args.batch_size):
        batch_results = evaluate_batch(
            batch=batch,
            args=args,
            resized_pred_dir=resized_pred_dir,
        )
        all_results.extend(batch_results)
        for result in batch_results:
            print_result(result)

    grouped = split_results_by_strategy(all_results)
    summaries = {strategy: summarize(records) for strategy, records in grouped.items()}
    save_results(output_dir=output_dir, results=all_results, summaries=summaries, pairing_info=pairing_info, args=args)

    print("\n=== Summary ===")
    for strategy, summary in summaries.items():
        print(f"\n[{strategy}]")
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
