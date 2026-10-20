"""Semantic region merging with support-surface and image-border handling."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import json
import os
from PIL import Image

from scene_inference_contextual import (
    ndi, load_binary_mask, mask_bbox_xyxy, canonical_label, class_group, prompt_group,
    get_effective_thresholds, get_object_underlay_ids_for_class, get_neighbor_stack,
    fill_unlabeled_gaps, repair_large_class_holes, rescue_zero_regions,
    CLASS_ORDER, CLASS_TO_ID, PALETTE, PROMPTS_STRUCTURE, PROMPTS_OPENINGS,
    PROMPTS_FURNITURE, PROMPTS_OBJECTS, ACTIVE_PROFILE_CORE_CLASSES,
    PROFILE_CORE_SCORE_BONUS, CLASS_MERGE_SUPPORT_BONUS, ensure_dir,
)


# ==========================================================
# Region competition settings
# ==========================================================

PARTIAL_MIN_COMPONENT_RATIO = 0.72
PARTIAL_MAX_COMPONENT_COUNT = 12
PARTIAL_MIN_BBOX_FILL = 0.08
PARTIAL_LOW_BBOX_FILL = 0.42
PARTIAL_LOW_SOLIDITY = 0.58
PARTIAL_MIN_AREA_TO_TRUST = 96

BORDER_RESCUE_MIN_AREA_RATIO = 0.015
BORDER_RESCUE_MIN_SCORE_MARGIN = 0.04
BORDER_RESCUE_MAX_TOUCH_COUNT = 2

MERGE_PARTIAL_PENALTY = 0.14
MERGE_TOUCH_PENALTY = 0.05
MERGE_INTERNAL_TOUCH_PENALTY = 0.08
MERGE_SUPPORT_SWALLOW_PENALTY = 0.10
MERGE_BORDER_RESCUE_BONUS = 0.06
MERGE_IMAGE_BORDER_BONUS = 0.03
MERGE_SMALL_MASK_PENALTY = 0.03

STRONG_RECROP_PAD_RATIO = 0.38
STRONG_RECROP_MIN_SIDE = 384

SUPPORT_SURFACE_CLASSES = {
    "table",
    "dining table",
    "conference table",
    "coffee table",
    "side table",
    "desk",
    "countertop",
    "cabinet",
    "kitchen cabinet",
    "bookshelf",
    "display shelf",
    "shelf",
    "refrigerator",
    "piano",
    "upright piano",
    "digital piano",
}

SUPPORT_SENSITIVE_CLASSES = {
    "chair",
    "dining chair",
    "office chair",
    "armchair",
    "stool",
    "bar stool",
    "bench",
    "sofa",
    "table",
    "dining table",
    "conference table",
    "coffee table",
    "side table",
    "desk",
    "cabinet",
    "kitchen cabinet",
    "bookshelf",
    "display shelf",
    "shelf",
    "refrigerator",
    "piano",
    "upright piano",
    "digital piano",
}

BORDER_RESCUE_CLASSES = {
    "table",
    "dining table",
    "conference table",
    "coffee table",
    "side table",
    "desk",
    "cabinet",
    "kitchen cabinet",
    "bookshelf",
    "display shelf",
    "shelf",
    "refrigerator",
    "piano",
    "upright piano",
    "digital piano",
    "sofa",
    "bed",
    "bathtub",
}

STRUCTURE_LIKE_CLASSES = {"wall", "floor", "ceiling", "window", "door", "glass door", "curtain", "blinds"}


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


# ==========================================================
# mask quality / semantic partial analysis
# ==========================================================

def get_mask_component_stats(mask_bool: np.ndarray) -> Dict[str, float]:
    area = int(mask_bool.sum())
    if area <= 0:
        return {
            "area": 0,
            "component_count": 0,
            "largest_component_area": 0,
            "largest_component_ratio": 0.0,
            "bbox_fill_ratio": 0.0,
            "solidity_proxy": 0.0,
        }

    bbox = mask_bbox_xyxy(mask_bool)
    if bbox is None:
        bbox_area = 1
    else:
        x1, y1, x2, y2 = bbox
        bbox_area = max(1, (x2 - x1) * (y2 - y1))

    component_count = 1
    largest = area
    if ndi is not None:
        labeled, ncomp = ndi.label(mask_bool.astype(np.uint8))
        component_count = int(ncomp)
        if ncomp > 0:
            sizes = np.bincount(labeled.ravel())[1:]
            if sizes.size > 0:
                largest = int(sizes.max())

    bbox_fill_ratio = float(area) / float(bbox_area)
    largest_ratio = float(largest) / float(max(1, area))
    solidity_proxy = largest_ratio * bbox_fill_ratio

    return {
        "area": area,
        "component_count": component_count,
        "largest_component_area": largest,
        "largest_component_ratio": largest_ratio,
        "bbox_fill_ratio": bbox_fill_ratio,
        "solidity_proxy": solidity_proxy,
    }


def get_touch_info_generic(mask_bool: np.ndarray, margin: int = 2) -> Dict[str, object]:
    h, w = mask_bool.shape
    if h <= 0 or w <= 0 or not mask_bool.any():
        return {
            "touch_boundary": False,
            "touch_sides": [],
            "touch_count": 0,
        }

    ys, xs = np.where(mask_bool)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    touch_left = x_min <= margin
    touch_right = x_max >= (w - 1 - margin)
    touch_top = y_min <= margin
    touch_bottom = y_max >= (h - 1 - margin)

    sides = []
    if touch_left:
        sides.append("left")
    if touch_right:
        sides.append("right")
    if touch_top:
        sides.append("top")
    if touch_bottom:
        sides.append("bottom")

    return {
        "touch_boundary": len(sides) > 0,
        "touch_sides": sides,
        "touch_count": len(sides),
    }


def infer_full_image_touch(mask_bool: np.ndarray, full_shape: Optional[Tuple[int, int]] = None, margin: int = 2) -> Dict[str, object]:
    if full_shape is None:
        return {
            "touch_image_boundary": False,
            "touch_image_sides": [],
            "touch_image_count": 0,
        }
    return {
        "touch_image_boundary": get_touch_info_generic(mask_bool, margin)["touch_boundary"],
        "touch_image_sides": get_touch_info_generic(mask_bool, margin)["touch_sides"],
        "touch_image_count": get_touch_info_generic(mask_bool, margin)["touch_count"],
    }


def class_prefers_border_rescue(class_name: str) -> bool:
    return canonical_label(class_name) in BORDER_RESCUE_CLASSES


def class_is_support_surface(class_name: str) -> bool:
    return canonical_label(class_name) in SUPPORT_SURFACE_CLASSES


def class_is_support_sensitive(class_name: str) -> bool:
    return canonical_label(class_name) in SUPPORT_SENSITIVE_CLASSES


def classify_semantic_partial(
    class_name: str,
    mask_bool: np.ndarray,
    item: Optional[dict] = None,
    full_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, object]:
    class_name = canonical_label(class_name)
    stats = get_mask_component_stats(mask_bool)
    area = int(stats["area"])
    if area < PARTIAL_MIN_AREA_TO_TRUST:
        return {
            **stats,
            "semantic_partial": False,
            "semantic_partial_reasons": [],
            "border_large_object_candidate": False,
            "touch_crop_boundary": False,
            "touch_internal_crop_boundary": False,
            "touch_image_boundary": False,
        }

    crop_touch = bool(item.get("touch_crop_boundary", False)) if item else False
    crop_touch_count = _safe_int(item.get("touch_count", 0)) if item else 0
    internal_touch = bool(item.get("touch_internal_crop_boundary", False)) if item else False
    internal_touch_count = _safe_int(item.get("touch_internal_count", 0)) if item else 0
    touch_image = bool(item.get("touch_image_boundary", False)) if item else False

    if item is None or "touch_image_boundary" not in item:
        full_touch = infer_full_image_touch(mask_bool, full_shape)
        touch_image = bool(full_touch["touch_image_boundary"])

    reasons: List[str] = []

    if class_is_support_sensitive(class_name):
        if stats["bbox_fill_ratio"] < PARTIAL_LOW_BBOX_FILL:
            reasons.append("low_bbox_fill")
        if stats["solidity_proxy"] < PARTIAL_LOW_SOLIDITY:
            reasons.append("low_solidity")
        if stats["largest_component_ratio"] < PARTIAL_MIN_COMPONENT_RATIO:
            reasons.append("fragmented")
        if stats["component_count"] > PARTIAL_MAX_COMPONENT_COUNT:
            reasons.append("too_many_components")
        if internal_touch and internal_touch_count >= 1:
            reasons.append("internal_boundary_touch")
        if crop_touch and crop_touch_count >= 1 and not touch_image:
            reasons.append("crop_boundary_not_image_boundary")

    semantic_partial = len(reasons) > 0

    border_large_object_candidate = (
        class_prefers_border_rescue(class_name)
        and touch_image
        and (crop_touch or internal_touch)
        and stats["bbox_fill_ratio"] >= PARTIAL_MIN_BBOX_FILL
        and stats["largest_component_ratio"] >= 0.55
    )

    # image-border large objects should not be treated as semantic-partial as aggressively
    if border_large_object_candidate:
        filtered = [r for r in reasons if r not in {"crop_boundary_not_image_boundary", "internal_boundary_touch"}]
        semantic_partial = len(filtered) > 0
        reasons = filtered

    return {
        **stats,
        "semantic_partial": semantic_partial,
        "semantic_partial_reasons": reasons,
        "border_large_object_candidate": border_large_object_candidate,
        "touch_crop_boundary": crop_touch,
        "touch_internal_crop_boundary": internal_touch,
        "touch_image_boundary": touch_image,
    }


# ==========================================================
# stronger stage2 completeness / rescue
# ==========================================================

def stage2_item_is_complete(class_name: str, item: dict) -> bool:
    class_name = canonical_label(class_name)
    mask = load_binary_mask(item["mask_path"])
    info = classify_semantic_partial(class_name, mask, item=item)

    score_thresh, min_area = get_effective_thresholds(class_name, second_stage=True)
    score = _safe_float(item.get("score", 0.0))
    area = int(info["area"])

    # easy pass: really good non-partial result
    if not info["semantic_partial"] and not item.get("touch_crop_boundary", False):
        return score >= score_thresh and area >= min_area

    # border rescue path: keep large true border objects from being rejected forever
    if info["border_large_object_candidate"]:
        return (
            score >= (score_thresh + BORDER_RESCUE_MIN_SCORE_MARGIN)
            and area >= max(min_area, int(mask.size * BORDER_RESCUE_MIN_AREA_RATIO))
            and _safe_int(item.get("touch_count", 0)) <= BORDER_RESCUE_MAX_TOUCH_COUNT
        )

    # support-sensitive furniture must pass a stricter completeness check
    if info["semantic_partial"]:
        return False

    # weaker fallback for classes that still touch one crop side but look geometrically coherent
    return (
        score >= (score_thresh + 0.08)
        and area >= max(int(min_area * 1.10), min_area + 96)
        and info["largest_component_ratio"] >= 0.78
        and info["bbox_fill_ratio"] >= 0.18
    )


def stage2_results_are_complete(class_name: str, prompt_results: List[dict]) -> bool:
    return any(stage2_item_is_complete(class_name, item) for item in prompt_results)


def build_directional_recrop(
    item: dict,
    full_width: int,
    full_height: int,
    min_side: int = STRONG_RECROP_MIN_SIDE,
    pad_ratio: float = STRONG_RECROP_PAD_RATIO,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Stronger recrop for half-object cases.
    Expand more aggressively toward the missing side instead of symmetric growth only.
    """
    crop_box = item.get("crop_box")
    if not isinstance(crop_box, list) or len(crop_box) != 4:
        return None

    mask = load_binary_mask(item["mask_path"])
    bbox = mask_bbox_xyxy(mask)
    if bbox is None:
        return None

    x1f, y1f, x2f, y2f = crop_box
    bx1, by1, bx2, by2 = bbox
    x1 = x1f + bx1
    y1 = y1f + by1
    x2 = x1f + bx2
    y2 = y1f + by2

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad_x = int(round(w * pad_ratio))
    pad_y = int(round(h * pad_ratio))

    sides = set(item.get("touch_sides", [])) | set(item.get("touch_internal_sides", []))

    # asymmetric expansion: grow more toward the likely-missing direction
    grow_left = pad_x * (2 if "left" in sides else 1)
    grow_right = pad_x * (2 if "right" in sides else 1)
    grow_top = pad_y * (2 if "top" in sides else 1)
    grow_bottom = pad_y * (2 if "bottom" in sides else 1)

    rx1 = max(0, x1 - grow_left)
    ry1 = max(0, y1 - grow_top)
    rx2 = min(full_width, x2 + grow_right)
    ry2 = min(full_height, y2 + grow_bottom)

    # minimum side guarantee
    if (rx2 - rx1) < min_side:
        extra = min_side - (rx2 - rx1)
        rx1 = max(0, rx1 - extra // 2)
        rx2 = min(full_width, rx2 + extra - extra // 2)
    if (ry2 - ry1) < min_side:
        extra = min_side - (ry2 - ry1)
        ry1 = max(0, ry1 - extra // 2)
        ry2 = min(full_height, ry2 + extra - extra // 2)

    return int(rx1), int(ry1), int(rx2), int(ry2)


# ==========================================================
# Region competition scores
# ==========================================================

def get_item_compete_score(item: dict, label_map: np.ndarray) -> float:
    class_name = canonical_label(item["prompt"])
    score = _safe_float(item.get("score", 0.0))
    mask = load_binary_mask(item["mask_path"])
    info = classify_semantic_partial(class_name, mask, item=item)

    compete = score

    if class_name in ACTIVE_PROFILE_CORE_CLASSES:
        compete += PROFILE_CORE_SCORE_BONUS

    support_map_bonus = min(3.0, max(0.0, _safe_float(item.get("support_count_bonus", 0.0))))
    compete += support_map_bonus * CLASS_MERGE_SUPPORT_BONUS

    if info["semantic_partial"]:
        compete -= MERGE_PARTIAL_PENALTY

    if bool(item.get("touch_crop_boundary", False)):
        compete -= MERGE_TOUCH_PENALTY

    if bool(item.get("touch_internal_crop_boundary", False)):
        compete -= MERGE_INTERNAL_TOUCH_PENALTY

    if info["border_large_object_candidate"]:
        compete += MERGE_BORDER_RESCUE_BONUS

    if bool(item.get("touch_image_boundary", False)) and class_prefers_border_rescue(class_name):
        compete += MERGE_IMAGE_BORDER_BONUS

    if info["bbox_fill_ratio"] < 0.12 and class_is_support_sensitive(class_name):
        compete -= MERGE_SMALL_MASK_PENALTY

    # discourage support surfaces from swallowing already-labeled independent furniture
    if class_is_support_surface(class_name):
        existing = label_map[mask]
        if existing.size > 0:
            occupied_ids, counts = np.unique(existing[existing > 0], return_counts=True)
            for cid, cnt in zip(occupied_ids, counts):
                existing_class = None
                for name, mapped in CLASS_TO_ID.items():
                    if mapped == int(cid):
                        existing_class = name
                        break
                if existing_class is None:
                    continue
                if existing_class == class_name:
                    continue
                if class_is_support_sensitive(existing_class):
                    overlap_ratio = float(cnt) / float(max(1, mask.sum()))
                    if overlap_ratio >= 0.08:
                        compete -= MERGE_SUPPORT_SWALLOW_PENALTY
                        break

    return compete


def apply_group(
    label_map: np.ndarray,
    items: List[dict],
    prompts: List[str],
    overwrite: bool = True,
    allowed_underlay_ids: Optional[set] = None,
) -> None:
    group_items = [x for x in items if x["prompt"] in prompts]
    if not group_items:
        return

    group_label_map = np.zeros_like(label_map)
    group_score_map = np.full(label_map.shape, -np.inf, dtype=np.float32)

    items_by_class: Dict[str, List[dict]] = {}
    for item in group_items:
        items_by_class.setdefault(canonical_label(item["prompt"]), []).append(item)

    for class_name in CLASS_ORDER:
        class_items = items_by_class.get(class_name)
        if not class_items:
            continue

        class_score_map = np.full(label_map.shape, -np.inf, dtype=np.float32)
        class_support_map = np.zeros(label_map.shape, dtype=np.uint8)

        for item in class_items:
            mask = load_binary_mask(item["mask_path"])
            class_support_map[mask] += 1

        for item in class_items:
            mask = load_binary_mask(item["mask_path"])
            item = dict(item)
            # stash support count bonus for score computation
            item["support_count_bonus"] = 0.0
            if mask.any():
                support_vals = class_support_map[mask]
                if support_vals.size > 0:
                    item["support_count_bonus"] = max(0.0, float(np.median(support_vals)) - 1.0)
            compete_score = get_item_compete_score(item, label_map)
            update = mask & (compete_score > class_score_map)
            class_score_map[update] = compete_score

        valid = class_score_map > -np.inf
        if not valid.any():
            continue

        cid = CLASS_TO_ID[class_name]
        update = valid & (class_score_map > group_score_map)
        group_label_map[update] = cid
        group_score_map[update] = class_score_map[update]

    valid = group_label_map > 0
    if overwrite:
        label_map[valid] = group_label_map[valid]
    else:
        if allowed_underlay_ids is None:
            write_mask = valid & (label_map == 0)
        else:
            underlay_mask = np.isin(label_map, np.array(sorted(allowed_underlay_ids), dtype=label_map.dtype))
            write_mask = valid & underlay_mask
        label_map[write_mask] = group_label_map[write_mask]


def apply_border_large_object_rescue(label_map: np.ndarray, results: List[dict]) -> np.ndarray:
    """
    Final-pass rescue for large objects touching the image boundary.
    Only fills blank pixels or structure-like underlay, so it won't recklessly erase
    already stable small objects.
    """
    structure_ids = {CLASS_TO_ID[c] for c in STRUCTURE_LIKE_CLASSES if c in CLASS_TO_ID}
    rescue_underlay = {0} | structure_ids

    candidates = []
    for item in results:
        class_name = canonical_label(item["prompt"])
        if not class_prefers_border_rescue(class_name):
            continue
        mask = load_binary_mask(item["mask_path"])
        info = classify_semantic_partial(class_name, mask, item=item)
        if not info["border_large_object_candidate"]:
            continue
        score = get_item_compete_score(item, label_map)
        candidates.append((score, class_name, mask))

    candidates.sort(key=lambda x: x[0])
    for _, class_name, mask in candidates:
        cid = CLASS_TO_ID[class_name]
        underlay = np.isin(label_map, np.array(sorted(rescue_underlay), dtype=label_map.dtype))
        write_mask = mask & underlay
        label_map[write_mask] = cid
    return label_map


def merge_scene(summary_json: str, out_dir: str):
    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    height = data["height"]
    width = data["width"]
    results = data["results"]

    label_map = np.zeros((height, width), dtype=np.uint8)

    apply_group(label_map, results, PROMPTS_STRUCTURE, overwrite=True)
    apply_group(label_map, results, PROMPTS_OPENINGS, overwrite=True)
    apply_group(label_map, results, PROMPTS_FURNITURE, overwrite=True)

    object_items = [x for x in results if x["prompt"] in PROMPTS_OBJECTS]
    if object_items:
        items_by_class: Dict[str, List[dict]] = {}
        for item in object_items:
            items_by_class.setdefault(canonical_label(item["prompt"]), []).append(item)
        for class_name in CLASS_ORDER:
            class_items = items_by_class.get(class_name)
            if not class_items:
                continue
            class_prompts = sorted({item["prompt"] for item in class_items})
            underlay_ids = get_object_underlay_ids_for_class(class_name)
            apply_group(
                label_map,
                class_items,
                class_prompts,
                overwrite=False,
                allowed_underlay_ids=underlay_ids,
            )

    label_map = apply_border_large_object_rescue(label_map, results)
    label_map = fill_unlabeled_gaps(label_map)
    label_map = repair_large_class_holes(label_map)
    label_map = rescue_zero_regions(label_map)
    label_map = fill_unlabeled_gaps(label_map, num_passes=1, min_neighbors=3)

    label_map_path = os.path.join(out_dir, "scene_label_map.png")
    Image.fromarray(label_map).save(label_map_path)

    color_map = np.zeros((height, width, 3), dtype=np.uint8)
    for cid, rgb in PALETTE.items():
        color_map[label_map == cid] = rgb

    color_map_path = os.path.join(out_dir, "scene_label_map_color.png")
    Image.fromarray(color_map).save(color_map_path)

    label_json_path = os.path.join(out_dir, "label_ids.json")
    with open(label_json_path, "w", encoding="utf-8") as f:
        json.dump(CLASS_TO_ID, f, ensure_ascii=False, indent=2)

    print(f"[INFO] saved label map: {label_map_path}")
    print(f"[INFO] saved color map: {color_map_path}")
    print(f"[INFO] saved label ids: {label_json_path}")

    return label_map_path, color_map_path, label_json_path
