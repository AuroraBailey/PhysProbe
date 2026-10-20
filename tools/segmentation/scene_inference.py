import os
import json
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# =========================
# 1. 路径配置
# =========================
IMG_PATH = "test.png"   # 改成你的原图路径
OUT_DIR = "scene_seg_outputs"
SCENE_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "configs", "indoor.json")
DEFAULT_CUDA_INDEX = os.environ.get("SAM3_CUDA_DEVICE", "3")

# overlay 透明度
ALPHA = 0.45
GAP_FILL_PASSES = 2
GAP_FILL_MIN_NEIGHBORS = 5

# 对 stage2 / wall_focus 的 crop 结果做边界截断检测
# 如果 mask 贴着 crop 边界，说明它大概率是不完整的局部结果
CROP_EDGE_TOUCH_MARGIN = 3

# 对边缘目标做一次更宽松的回退保留，提升召回率
FALLBACK_RULES = {
    "structure": {"score_margin": 0.04, "min_area_ratio": 0.70, "min_area_floor": 1500},
    "openings": {"score_margin": 0.08, "min_area_ratio": 0.50, "min_area_floor": 180},
    "furniture": {"score_margin": 0.10, "min_area_ratio": 0.40, "min_area_floor": 120},
    "objects": {"score_margin": 0.12, "min_area_ratio": 0.25, "min_area_floor": 24},
}

SECOND_STAGE_RELAX_RULES = {
    "structure": {"score_margin": 0.00, "min_area_ratio": 1.00, "min_area_floor": 1500},
    "openings": {"score_margin": 0.03, "min_area_ratio": 0.60, "min_area_floor": 96},
    "furniture": {"score_margin": 0.05, "min_area_ratio": 0.40, "min_area_floor": 64},
    "objects": {"score_margin": 0.05, "min_area_ratio": 0.20, "min_area_floor": 16},
}
SECOND_STAGE_MAX_CROPS = 16
SECOND_STAGE_MIN_CROP_SIDE = 320
SECOND_STAGE_GRID_ROWS = 3
SECOND_STAGE_GRID_COLS = 3
SECOND_STAGE_GRID_CROP_RATIO = 0.58
CLASS_MERGE_SUPPORT_BONUS = 0.015

SMALL_OBJECT_CLASSES = {
    "pendant light",
    "chandelier",
    "recessed ceiling light",
    "spotlight",
    "ceiling light",
    "table lamp",
    "floor lamp",
    "faucet",
    "toy",
    "climbing holds",
    "play equipment",
    "plate",
    "bowl",
    "cup",
    "glass",
    "bottle",
    "tray",
    "placemat",
}
SMALL_OBJECT_STAGE1_RULES = {
    "score_margin": 0.02,
    "min_area_ratio": 0.60,
    "min_area_floor": 12,
    "max_area_ratio_floor": 0.06,
}
SMALL_OBJECT_STAGE2_RULES = {
    "score_margin": 0.05,
    "min_area_ratio": 0.35,
    "min_area_floor": 8,
    "max_area_ratio_floor": 0.10,
}
WALL_FOCUS_MIN_WALL_RATIO = 0.14
WALL_FOCUS_MAX_FLOOR_RATIO = 0.04
WALL_FOCUS_MAX_FURNITURE_CLASSES = 3
WALL_FOCUS_MIN_LIGHT_COUNT = 2
WALL_FOCUS_TARGET_MIN_RATIO = {
    "ceiling": 0.10,
}
WALL_FOCUS_MAX_AREA_RATIO_OVERRIDE = {
    "ceiling": 0.65,
    "poster": 0.18,
    "bulletin board": 0.18,
    "display screen": 0.14,
    "spotlight": 0.03,
}

# =========================
# 2. 从 JSON 读取场景词表
# =========================
def load_scene_vocab(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    groups = cfg["groups"]
    prompts_structure = list(groups["structure"])
    prompts_openings = list(groups["openings"])
    prompts_furniture = list(groups["furniture"])
    prompts_objects = list(groups["objects"])
    all_prompts = prompts_structure + prompts_openings + prompts_furniture + prompts_objects

    prompt_to_class = dict(cfg.get("prompt_to_class", {}))
    for prompt in all_prompts:
        prompt_to_class.setdefault(prompt, prompt)

    thresholds = {
        name: {
            "score": float(conf["score"]),
            "min_area": int(conf["min_area"]),
        }
        for name, conf in cfg["thresholds"].items()
    }
    max_area_ratio = {
        name: float(value)
        for name, value in cfg.get("max_area_ratio", {}).items()
    }

    class_order = list(cfg["class_order"])
    class_to_id = {c: i + 1 for i, c in enumerate(class_order)}

    missing_thresholds = [name for name in class_order if name not in thresholds]
    if missing_thresholds:
        raise ValueError(f"Missing thresholds for classes: {missing_thresholds}")

    invalid_targets = sorted(
        {label for label in prompt_to_class.values() if label not in class_to_id}
    )
    if invalid_targets:
        raise ValueError(f"Unknown canonical classes in prompt mapping: {invalid_targets}")

    class_palette = cfg["class_palette"]
    missing_palette = [name for name in class_order if name not in class_palette]
    if missing_palette:
        raise ValueError(f"Missing palette entries for classes: {missing_palette}")

    palette = {0: (0, 0, 0)}
    for class_name in class_order:
        rgb = tuple(int(v) for v in class_palette[class_name])
        if len(rgb) != 3:
            raise ValueError(f"Palette entry for {class_name!r} must be RGB")
        palette[class_to_id[class_name]] = rgb

    return {
        "prompts_structure": prompts_structure,
        "prompts_openings": prompts_openings,
        "prompts_furniture": prompts_furniture,
        "prompts_objects": prompts_objects,
        "all_prompts": all_prompts,
        "prompt_to_class": prompt_to_class,
        "thresholds": thresholds,
        "max_area_ratio": max_area_ratio,
        "class_order": class_order,
        "class_to_id": class_to_id,
        "palette": palette,
    }


SCENE_VOCAB = load_scene_vocab(SCENE_VOCAB_PATH)
PROMPTS_STRUCTURE = SCENE_VOCAB["prompts_structure"]
PROMPTS_OPENINGS = SCENE_VOCAB["prompts_openings"]
PROMPTS_FURNITURE = SCENE_VOCAB["prompts_furniture"]
PROMPTS_OBJECTS = SCENE_VOCAB["prompts_objects"]
ALL_PROMPTS = SCENE_VOCAB["all_prompts"]
PROMPT_TO_CLASS = SCENE_VOCAB["prompt_to_class"]
PROMPT_GROUP = {
    **{prompt: "structure" for prompt in PROMPTS_STRUCTURE},
    **{prompt: "openings" for prompt in PROMPTS_OPENINGS},
    **{prompt: "furniture" for prompt in PROMPTS_FURNITURE},
    **{prompt: "objects" for prompt in PROMPTS_OBJECTS},
}

# =========================
# 3. 每类阈值
# =========================
THRESHOLDS: Dict[str, Dict[str, float]] = SCENE_VOCAB["thresholds"]

# 限制结构类最大面积比例，避免一张巨型 mask 把全图吞掉
MAX_AREA_RATIO = SCENE_VOCAB["max_area_ratio"]

# =========================
# 4. 类别顺序与配色
# =========================
CLASS_ORDER = SCENE_VOCAB["class_order"]
CLASS_TO_ID = SCENE_VOCAB["class_to_id"]
PALETTE = SCENE_VOCAB["palette"]
CLASS_GROUP = {
    **{canonical: "structure" for canonical in CLASS_ORDER if canonical in PROMPTS_STRUCTURE},
    **{
        canonical: "openings"
        for canonical in CLASS_ORDER
        if canonical in [PROMPT_TO_CLASS.get(p, p) for p in PROMPTS_OPENINGS]
    },
    **{
        canonical: "furniture"
        for canonical in CLASS_ORDER
        if canonical in [PROMPT_TO_CLASS.get(p, p) for p in PROMPTS_FURNITURE]
    },
    **{
        canonical: "objects"
        for canonical in CLASS_ORDER
        if canonical in [PROMPT_TO_CLASS.get(p, p) for p in PROMPTS_OBJECTS]
    },
}
CANONICAL_TO_PROMPTS: Dict[str, List[str]] = {}
for prompt in ALL_PROMPTS:
    CANONICAL_TO_PROMPTS.setdefault(PROMPT_TO_CLASS.get(prompt, prompt), []).append(prompt)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_numpy_mask(mask) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.array(mask)
    while mask.ndim > 2:
        mask = mask[0]
    return mask > 0


def save_binary_mask(mask_bool: np.ndarray, path: str) -> None:
    Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(path)


def load_binary_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def get_neighbor_stack(label_map: np.ndarray) -> np.ndarray:
    padded = np.pad(label_map, 1, mode="edge")
    neighbors = []
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    for dy, dx in offsets:
        neighbors.append(
            padded[1 + dy:1 + dy + label_map.shape[0], 1 + dx:1 + dx + label_map.shape[1]]
        )
    return np.stack(neighbors, axis=0)


def fill_unlabeled_gaps(
    label_map: np.ndarray,
    num_passes: int = GAP_FILL_PASSES,
    min_neighbors: int = GAP_FILL_MIN_NEIGHBORS,
) -> np.ndarray:
    """
    用邻域多数投票补掉细小未标注缝隙，减少可视化中的黑色边线。
    仅填充 label=0 的像素，不主动改写已有类别。
    """
    refined = label_map.copy()
    label_ids = [cid for cid in np.unique(refined) if cid != 0]
    if not label_ids:
        return refined

    for _ in range(num_passes):
        zero_mask = refined == 0
        if not zero_mask.any():
            break

        neighbors = get_neighbor_stack(refined)
        best_label = np.zeros_like(refined)
        best_count = np.zeros_like(refined, dtype=np.uint8)

        for cid in label_ids:
            count = (neighbors == cid).sum(axis=0).astype(np.uint8)
            update = zero_mask & (count >= min_neighbors) & (count > best_count)
            best_label[update] = cid
            best_count[update] = count[update]

        changed = zero_mask & (best_label > 0)
        if not changed.any():
            break
        refined[changed] = best_label[changed]

    return refined


def get_score_and_min_area(prompt: str) -> Tuple[float, int]:
    conf = THRESHOLDS.get(prompt)
    if conf is None:
        conf = THRESHOLDS.get(canonical_label(prompt), {"score": 0.2, "min_area": 100})
    return float(conf["score"]), int(conf["min_area"])


def canonical_label(prompt: str) -> str:
    return PROMPT_TO_CLASS.get(prompt, prompt)


def prompt_group(prompt: str) -> str:
    return PROMPT_GROUP.get(prompt, "objects")


def class_group(class_name: str) -> str:
    return CLASS_GROUP.get(class_name, "objects")


def get_max_area_ratio(prompt: str) -> float:
    max_ratio = MAX_AREA_RATIO.get(prompt)
    if max_ratio is None:
        max_ratio = MAX_AREA_RATIO.get(canonical_label(prompt), 1.0)
    return float(max_ratio)


def should_keep_as_fallback(
    prompt: str,
    score_val: float,
    area: int,
    score_thresh: float,
    min_area: int,
    image_area: int,
    second_stage: bool = False,
    max_area_ratio_override: float | None = None,
) -> bool:
    rules = FALLBACK_RULES[prompt_group(prompt)]
    min_score = max(0.0, score_thresh - rules["score_margin"])
    min_fallback_area = max(int(min_area * rules["min_area_ratio"]), rules["min_area_floor"])
    max_ratio = max_area_ratio_override
    if max_ratio is None:
        max_ratio = get_effective_max_area_ratio(prompt, second_stage=second_stage)
    max_area = int(image_area * max_ratio)
    return (
        score_val >= min_score
        and area >= min_fallback_area
        and area <= max_area
    )


def is_small_object_class(prompt_or_class: str) -> bool:
    return canonical_label(prompt_or_class) in SMALL_OBJECT_CLASSES


def resolve_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"

    if DEFAULT_CUDA_INDEX.isdigit():
        requested_index = int(DEFAULT_CUDA_INDEX)
        if requested_index < torch.cuda.device_count():
            torch.cuda.set_device(requested_index)
            return f"cuda:{requested_index}"
        print(f"[WARN] requested cuda:{requested_index} unavailable, fallback to cuda:0")

    torch.cuda.set_device(0)
    return "cuda:0"


def set_image_state(processor: Sam3Processor, image: Image.Image, device: str):
    with torch.inference_mode():
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", enabled=False):
                return processor.set_image(image)
        return processor.set_image(image)


def run_prompt(processor: Sam3Processor, state, prompt: str, device: str):
    with torch.inference_mode():
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", enabled=False):
                return processor.set_text_prompt(state=state, prompt=prompt)
        return processor.set_text_prompt(state=state, prompt=prompt)


def get_effective_thresholds(prompt: str, second_stage: bool = False) -> Tuple[float, int]:
    score_thresh, min_area = get_score_and_min_area(prompt)
    if is_small_object_class(prompt):
        rules = SMALL_OBJECT_STAGE2_RULES if second_stage else SMALL_OBJECT_STAGE1_RULES
        score_thresh = max(0.0, score_thresh - rules["score_margin"])
        min_area = max(int(min_area * rules["min_area_ratio"]), rules["min_area_floor"])

    if not second_stage:
        return score_thresh, min_area

    rules = SECOND_STAGE_RELAX_RULES[prompt_group(prompt)]
    score_thresh = max(0.0, score_thresh - rules["score_margin"])
    min_area = max(int(min_area * rules["min_area_ratio"]), rules["min_area_floor"])
    return score_thresh, min_area


def get_effective_max_area_ratio(prompt: str, second_stage: bool = False) -> float:
    max_ratio = get_max_area_ratio(prompt)
    if is_small_object_class(prompt):
        rules = SMALL_OBJECT_STAGE2_RULES if second_stage else SMALL_OBJECT_STAGE1_RULES
        max_ratio = max(max_ratio, rules["max_area_ratio_floor"])
    return max_ratio


def mask_bbox_xyxy(mask_bool: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def get_crop_touch_info(mask_bool: np.ndarray, margin: int = CROP_EDGE_TOUCH_MARGIN) -> dict:
    """
    只在 crop-local mask 上调用。
    判断 mask 是否贴着 crop 边界，作为“可能被裁断”的通用信号。
    """
    h, w = mask_bool.shape
    if h == 0 or w == 0 or not mask_bool.any():
        return {
            "touch_crop_boundary": False,
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

    touch_sides = []
    if touch_left:
        touch_sides.append("left")
    if touch_right:
        touch_sides.append("right")
    if touch_top:
        touch_sides.append("top")
    if touch_bottom:
        touch_sides.append("bottom")

    return {
        "touch_crop_boundary": len(touch_sides) > 0,
        "touch_sides": touch_sides,
        "touch_count": len(touch_sides),
    }


def normalize_crop_box(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    min_side: int = SECOND_STAGE_MIN_CROP_SIDE,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w < min_side:
        grow = min_side - crop_w
        x1 -= grow // 2
        x2 += grow - grow // 2
    if crop_h < min_side:
        grow = min_side - crop_h
        y1 -= grow // 2
        y2 += grow - grow // 2

    if x1 < 0:
        x2 = min(width, x2 - x1)
        x1 = 0
    if y1 < 0:
        y2 = min(height, y2 - y1)
        y1 = 0
    if x2 > width:
        x1 = max(0, x1 - (x2 - width))
        x2 = width
    if y2 > height:
        y1 = max(0, y1 - (y2 - height))
        y2 = height

    return int(x1), int(y1), int(x2), int(y2)


def expand_crop_box(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    pad_ratio: float = 0.18,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    return normalize_crop_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def add_crop_if_new(
    crops: List[dict],
    seen: set,
    name: str,
    box: Tuple[int, int, int, int],
) -> None:
    key = tuple(int(v) for v in box)
    if key in seen:
        return
    seen.add(key)
    crops.append({"name": name, "box": key})


def evenly_spaced_starts(length: int, window: int, count: int) -> List[int]:
    if count <= 1 or window >= length:
        return [0]
    max_start = length - window
    return sorted({int(round(i * max_start / (count - 1))) for i in range(count)})


def add_dense_grid_crops(
    crops: List[dict],
    seen: set,
    width: int,
    height: int,
) -> None:
    crop_w = max(SECOND_STAGE_MIN_CROP_SIDE, int(round(width * SECOND_STAGE_GRID_CROP_RATIO)))
    crop_h = max(SECOND_STAGE_MIN_CROP_SIDE, int(round(height * SECOND_STAGE_GRID_CROP_RATIO)))
    crop_w = min(crop_w, width)
    crop_h = min(crop_h, height)

    x_starts = evenly_spaced_starts(width, crop_w, SECOND_STAGE_GRID_COLS)
    y_starts = evenly_spaced_starts(height, crop_h, SECOND_STAGE_GRID_ROWS)

    for row, y1 in enumerate(y_starts):
        for col, x1 in enumerate(x_starts):
            box = normalize_crop_box((x1, y1, x1 + crop_w, y1 + crop_h), width, height)
            add_crop_if_new(crops, seen, f"grid_r{row}_c{col}", box)


def build_second_stage_crops(
    image: Image.Image,
    stage1_results: List[dict],
) -> List[dict]:
    width, height = image.size
    crops: List[dict] = []
    seen = set()

    add_dense_grid_crops(crops, seen, width, height)
    add_crop_if_new(
        crops,
        seen,
        "center_large",
        normalize_crop_box(
            (int(width * 0.12), int(height * 0.12), int(width * 0.88), int(height * 0.88)),
            width,
            height,
        ),
    )

    anchor_classes = [
        "floor",
        "wall",
        "rug",
        "window",
        "cabinet",
        "dresser",
        "bookshelf",
        "display shelf",
        "bed",
        "sofa",
        "desk",
    ]
    best_items: Dict[str, dict] = {}
    for item in stage1_results:
        canonical = canonical_label(item["prompt"])
        if canonical not in anchor_classes:
            continue
        prev = best_items.get(canonical)
        if prev is None or float(item["score"]) > float(prev["score"]):
            best_items[canonical] = item

    for class_name in anchor_classes:
        item = best_items.get(class_name)
        if item is None:
            continue
        mask = load_binary_mask(item["mask_path"])
        bbox = mask_bbox_xyxy(mask)
        if bbox is None:
            continue
        add_crop_if_new(
            crops,
            seen,
            f"anchor_{class_name.replace(' ', '_')}",
            expand_crop_box(bbox, width, height),
        )

    return crops[:SECOND_STAGE_MAX_CROPS]


def should_retry_class_in_stage2(class_name: str) -> bool:
    if class_group(class_name) == "structure":
        return False
    conf = THRESHOLDS.get(class_name, {"min_area": 1000})
    max_ratio = MAX_AREA_RATIO.get(class_name, 1.0)
    return int(conf["min_area"]) <= 1600 or float(max_ratio) <= 0.20


def get_second_stage_target_classes(stage1_results: List[dict]) -> List[str]:
    present = {canonical_label(item["prompt"]) for item in stage1_results}
    return [
        class_name
        for class_name in CLASS_ORDER
        if class_name not in present and should_retry_class_in_stage2(class_name)
    ]


def get_prompts_for_canonical_class(class_name: str) -> List[str]:
    prompts = list(CANONICAL_TO_PROMPTS.get(class_name, [class_name]))
    return sorted(
        prompts,
        key=lambda prompt: (
            prompt != class_name,
            ALL_PROMPTS.index(prompt) if prompt in ALL_PROMPTS else 10**9,
        ),
    )


def get_best_area_by_class(results: List[dict]) -> Dict[str, int]:
    best_area: Dict[str, int] = {}
    for item in results:
        class_name = canonical_label(item["prompt"])
        area = int(item["area"])
        if area > best_area.get(class_name, 0):
            best_area[class_name] = area
    return best_area


def count_light_hits(results: List[dict]) -> int:
    light_classes = {"recessed ceiling light", "ceiling light", "spotlight"}
    return sum(1 for item in results if canonical_label(item["prompt"]) in light_classes)


def should_enable_wall_focus(results: List[dict], image_area: int) -> bool:
    best_area = get_best_area_by_class(results)
    wall_like_ratio = max(
        best_area.get("wall", 0),
        best_area.get("decorative wall", 0),
        best_area.get("wainscoting", 0),
    ) / max(image_area, 1)
    floor_ratio = best_area.get("floor", 0) / max(image_area, 1)
    furniture_present = {
        canonical_label(item["prompt"])
        for item in results
        if class_group(canonical_label(item["prompt"])) == "furniture" and int(item["area"]) >= 400
    }
    target_missing = (
        best_area.get("ceiling", 0) < int(image_area * WALL_FOCUS_TARGET_MIN_RATIO["ceiling"])
        or "poster" not in best_area
        or "bulletin board" not in best_area
        or "display screen" not in best_area
        or "spotlight" not in best_area
    )
    return (
        wall_like_ratio >= WALL_FOCUS_MIN_WALL_RATIO
        and floor_ratio <= WALL_FOCUS_MAX_FLOOR_RATIO
        and len(furniture_present) <= WALL_FOCUS_MAX_FURNITURE_CLASSES
        and count_light_hits(results) >= WALL_FOCUS_MIN_LIGHT_COUNT
        and target_missing
    )


def build_wall_focus_crops(image: Image.Image) -> List[dict]:
    width, height = image.size
    crops: List[dict] = []
    seen = set()
    crop_defs = [
        ("wall_focus_top_band", (0, 0, width, int(height * 0.62))),
        ("wall_focus_upper_center", (int(width * 0.10), int(height * 0.08), int(width * 0.90), int(height * 0.70))),
        ("wall_focus_wall_band", (0, int(height * 0.28), width, int(height * 0.95))),
        ("wall_focus_mid_center", (int(width * 0.18), int(height * 0.25), int(width * 0.82), int(height * 0.88))),
    ]
    for name, box in crop_defs:
        add_crop_if_new(crops, seen, name, normalize_crop_box(box, width, height))
    return crops


def get_wall_focus_target_classes(results: List[dict], image_area: int) -> List[str]:
    best_area = get_best_area_by_class(results)
    targets: List[str] = []
    if best_area.get("ceiling", 0) < int(image_area * WALL_FOCUS_TARGET_MIN_RATIO["ceiling"]):
        targets.append("ceiling")
    if "poster" not in best_area:
        targets.append("poster")
    if "bulletin board" not in best_area:
        targets.append("bulletin board")
    if "display screen" not in best_area:
        targets.append("display screen")
    if count_light_hits(results) >= WALL_FOCUS_MIN_LIGHT_COUNT and "spotlight" not in best_area:
        targets.append("spotlight")
    return targets


def get_wall_focus_crop_names(class_name: str) -> List[str]:
    crop_map = {
        "ceiling": ["wall_focus_top_band", "wall_focus_upper_center"],
        "spotlight": ["wall_focus_top_band", "wall_focus_upper_center"],
        "poster": ["wall_focus_wall_band", "wall_focus_mid_center"],
        "bulletin board": ["wall_focus_wall_band", "wall_focus_mid_center"],
        "display screen": ["wall_focus_wall_band", "wall_focus_mid_center"],
    }
    return crop_map.get(class_name, ["wall_focus_wall_band"])


def make_full_image_mask(
    crop_mask: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    full_shape: Tuple[int, int],
) -> np.ndarray:
    full_mask = np.zeros(full_shape, dtype=bool)
    x1, y1, x2, y2 = crop_box
    full_mask[y1:y2, x1:x2] = crop_mask
    return full_mask


def collect_results_for_prompt(
    prompt: str,
    output,
    image_area: int,
    prompt_dir: str,
    second_stage: bool = False,
    stage_tag: str = "stage1",
    crop_name: str | None = None,
    crop_box: Tuple[int, int, int, int] | None = None,
    full_shape: Tuple[int, int] | None = None,
    max_area_ratio_override: float | None = None,
) -> Tuple[List[dict], int]:
    score_thresh, min_area = get_effective_thresholds(prompt, second_stage=second_stage)

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    results: List[dict] = []
    fallback_candidates = []
    total = 0

    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        total += 1
        score_val = float(score.item()) if hasattr(score, "item") else float(score)

        # 先在 crop-local 坐标下拿 mask，并做是否贴边检测
        crop_local_mask_bool = to_numpy_mask(mask)
        touch_info = {
            "touch_crop_boundary": False,
            "touch_sides": [],
            "touch_count": 0,
        }
        if crop_box is not None:
            touch_info = get_crop_touch_info(crop_local_mask_bool)

        mask_bool = crop_local_mask_bool
        if crop_box is not None and full_shape is not None:
            mask_bool = make_full_image_mask(mask_bool, crop_box, full_shape)

        area = int(mask_bool.sum())
        max_ratio = max_area_ratio_override
        if max_ratio is None:
            max_ratio = get_effective_max_area_ratio(prompt, second_stage=second_stage)
        max_area = int(image_area * max_ratio)
        passes_strict = (
            score_val >= score_thresh
            and area >= min_area
            and area <= max_area
        )
        if not passes_strict:
            if should_keep_as_fallback(
                prompt=prompt,
                score_val=score_val,
                area=area,
                score_thresh=score_thresh,
                min_area=min_area,
                image_area=image_area,
                second_stage=second_stage,
                max_area_ratio_override=max_area_ratio_override,
            ):
                fallback_candidates.append({
                    "index": i,
                    "mask_bool": mask_bool,
                    "box": box,
                    "score": score_val,
                    "area": area,
                    "touch_info": touch_info,
                })
            continue

        crop_suffix = f"_{crop_name}" if crop_name else ""
        mask_name = f"{stage_tag}{crop_suffix}_{i:03d}_score_{score_val:.3f}_area_{area}.png"
        mask_path = os.path.join(prompt_dir, mask_name)
        save_binary_mask(mask_bool, mask_path)

        item = {
            "prompt": prompt,
            "mask_path": mask_path,
            "score": score_val,
            "area": area,
            "box": str(box),
            "stage": stage_tag,
            **touch_info,
        }
        if crop_box is not None:
            item["crop_name"] = crop_name
            item["crop_box"] = list(crop_box)
        results.append(item)

    if not results and fallback_candidates:
        best = max(fallback_candidates, key=lambda x: (x["score"], x["area"]))
        crop_suffix = f"_{crop_name}" if crop_name else ""
        mask_name = (
            f"{stage_tag}{crop_suffix}_{best['index']:03d}_"
            f"score_{best['score']:.3f}_area_{best['area']}_fallback.png"
        )
        mask_path = os.path.join(prompt_dir, mask_name)
        save_binary_mask(best["mask_bool"], mask_path)

        item = {
            "prompt": prompt,
            "mask_path": mask_path,
            "score": best["score"],
            "area": best["area"],
            "box": str(best["box"]),
            "fallback": True,
            "stage": stage_tag,
            **best["touch_info"],
        }
        if crop_box is not None:
            item["crop_name"] = crop_name
            item["crop_box"] = list(crop_box)
        results.append(item)

    return results, total


def run_segmentation(
    img_path: str,
    out_dir: str,
) -> Tuple[str, str]:
    """
    跑所有 prompt，保存单个 mask 和 summary.json
    返回: (summary_json_path, original_image_path)
    """
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "masks"))

    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    image_area = width * height

    device = resolve_device()
    print(f"[INFO] device = {device}")
    if device.startswith("cuda"):
        gpu_index = int(device.split(":")[1]) if ":" in device else 0
        print(f"[INFO] gpu = {torch.cuda.get_device_name(gpu_index)}")
        print(f"[INFO] capability = {torch.cuda.get_device_capability(gpu_index)}")

    build_device = "cuda" if device.startswith("cuda") else "cpu"
    model = build_sam3_image_model(device=build_device)
    model = model.to(device=device).float().eval()
    processor = Sam3Processor(model, device=device)

    state = set_image_state(processor, image, device)

    all_results = []

    # 第一阶段：整图分割
    for prompt in ALL_PROMPTS:
        print(f"\n=== prompt: {prompt} ===")
        prompt_dir = os.path.join(out_dir, "masks", prompt.replace(" ", "_"))
        ensure_dir(prompt_dir)

        output = run_prompt(processor, state, prompt, device)
        prompt_results, total = collect_results_for_prompt(
            prompt=prompt,
            output=output,
            image_area=image_area,
            prompt_dir=prompt_dir,
            second_stage=False,
            stage_tag="stage1",
        )
        all_results.extend(prompt_results)
        kept = len(prompt_results)
        print(f"[INFO] total={total}, kept={kept}")

    # 第二阶段：对第一阶段缺失类别进行局部放大重查
    # 新规则：只有拿到“非贴边”的 crop 结果，才认为该类真正完成
    second_stage_target_classes = get_second_stage_target_classes(all_results)
    second_stage_crops = build_second_stage_crops(image, all_results)
    if second_stage_target_classes and second_stage_crops:
        print(
            f"\n[INFO] stage2 enabled: missing_classes={len(second_stage_target_classes)}, "
            f"crops={len(second_stage_crops)}"
        )
        remaining_classes = set(second_stage_target_classes)
        full_shape = (height, width)
        for crop in second_stage_crops:
            if not remaining_classes:
                break

            x1, y1, x2, y2 = crop["box"]
            crop_image = image.crop((x1, y1, x2, y2))
            crop_state = set_image_state(processor, crop_image, device)
            print(
                f"\n[INFO] stage2 crop={crop['name']} box={crop['box']} "
                f"remaining_classes={len(remaining_classes)}"
            )

            for class_name in list(remaining_classes):
                prompt_dir = os.path.join(out_dir, "masks", class_name.replace(" ", "_"))
                ensure_dir(prompt_dir)

                class_found_complete = False
                class_found_any = False

                for prompt in get_prompts_for_canonical_class(class_name):
                    output = run_prompt(processor, crop_state, prompt, device)
                    prompt_results, total = collect_results_for_prompt(
                        prompt=prompt,
                        output=output,
                        image_area=image_area,
                        prompt_dir=prompt_dir,
                        second_stage=True,
                        stage_tag="stage2",
                        crop_name=crop["name"],
                        crop_box=crop["box"],
                        full_shape=full_shape,
                    )
                    if not prompt_results:
                        continue

                    all_results.extend(prompt_results)
                    class_found_any = True

                    has_non_truncated = any(
                        not item.get("touch_crop_boundary", False)
                        for item in prompt_results
                    )

                    print(
                        f"[INFO] stage2 class={class_name}, prompt={prompt}, "
                        f"crop={crop['name']}, total={total}, kept={len(prompt_results)}, "
                        f"complete={has_non_truncated}"
                    )

                    # 只有出现非贴边结果，才认为这个类真正完整
                    if has_non_truncated:
                        class_found_complete = True
                        break

                if class_found_complete:
                    remaining_classes.remove(class_name)
                elif class_found_any:
                    print(
                        f"[INFO] stage2 class={class_name} only got boundary-touch results "
                        f"on crop={crop['name']}, keep searching next crops"
                    )

    if should_enable_wall_focus(all_results, image_area):
        wall_focus_targets = get_wall_focus_target_classes(all_results, image_area)
        wall_focus_crops = {crop["name"]: crop for crop in build_wall_focus_crops(image)}
        if wall_focus_targets and wall_focus_crops:
            print(
                f"\n[INFO] wall_focus enabled: targets={wall_focus_targets}, "
                f"crops={list(wall_focus_crops)}"
            )
            full_shape = (height, width)
            crop_states = {}
            for crop_name, crop in wall_focus_crops.items():
                x1, y1, x2, y2 = crop["box"]
                crop_image = image.crop((x1, y1, x2, y2))
                crop_states[crop_name] = set_image_state(processor, crop_image, device)

            for class_name in wall_focus_targets:
                class_found = False
                for crop_name in get_wall_focus_crop_names(class_name):
                    crop = wall_focus_crops.get(crop_name)
                    if crop is None:
                        continue
                    prompt_dir = os.path.join(out_dir, "masks", class_name.replace(" ", "_"))
                    ensure_dir(prompt_dir)
                    crop_state = crop_states[crop_name]
                    max_area_override = WALL_FOCUS_MAX_AREA_RATIO_OVERRIDE.get(class_name)
                    for prompt in get_prompts_for_canonical_class(class_name):
                        output = run_prompt(processor, crop_state, prompt, device)
                        prompt_results, total = collect_results_for_prompt(
                            prompt=prompt,
                            output=output,
                            image_area=image_area,
                            prompt_dir=prompt_dir,
                            second_stage=True,
                            stage_tag="wall_focus",
                            crop_name=crop_name,
                            crop_box=crop["box"],
                            full_shape=full_shape,
                            max_area_ratio_override=max_area_override,
                        )
                        if prompt_results:
                            all_results.extend(prompt_results)
                            class_found = True
                            print(
                                f"[INFO] wall_focus class={class_name}, prompt={prompt}, "
                                f"crop={crop_name}, total={total}, kept={len(prompt_results)}"
                            )
                            break
                    if class_found:
                        break

    summary = {
        "image": img_path,
        "width": width,
        "height": height,
        "device": device,
        "num_results": len(all_results),
        "results": all_results,
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] saved summary: {summary_path}")
    return summary_path, img_path


def apply_group(
    label_map: np.ndarray,
    items: List[dict],
    allowed_prompts: List[str],
    overwrite: bool = True,
) -> None:
    """
    先在 canonical class 内合并同义词/多次命中，再在类之间做竞争。
    同类的多 prompt 共识会得到轻微加成，减少近义类别互相抢占。
    """
    group_items = [x for x in items if x["prompt"] in allowed_prompts]
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
            score = float(item["score"])
            class_support_map[mask] += 1
            update = mask & (score > class_score_map)
            class_score_map[update] = score

        valid = class_score_map > -np.inf
        if not valid.any():
            continue

        compete_score_map = class_score_map.copy()
        support_bonus = np.clip(class_support_map.astype(np.float32) - 1.0, 0.0, 3.0)
        compete_score_map[valid] += support_bonus[valid] * CLASS_MERGE_SUPPORT_BONUS

        cid = CLASS_TO_ID[class_name]
        update = valid & (compete_score_map > group_score_map)
        group_label_map[update] = cid
        group_score_map[update] = compete_score_map[update]

    valid = group_label_map > 0
    if overwrite:
        label_map[valid] = group_label_map[valid]
    else:
        write_mask = valid & (label_map == 0)
        label_map[write_mask] = group_label_map[write_mask]


def merge_scene(summary_json: str, out_dir: str) -> Tuple[str, str, str]:
    """
    分层合并:
    1) structure 铺底
    2) furniture 覆盖 structure
    3) objects 覆盖 furniture + structure
    返回:
    - 灰度标签图路径
    - 彩色标签图路径
    - label_ids.json 路径
    """
    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    height = data["height"]
    width = data["width"]
    results = data["results"]

    label_map = np.zeros((height, width), dtype=np.uint8)

    # 第1层：结构
    apply_group(label_map, results, PROMPTS_STRUCTURE, overwrite=True)

    # 第2层：门窗开口，覆盖结构
    apply_group(label_map, results, PROMPTS_OPENINGS, overwrite=True)

    # 第3层：家具
    apply_group(label_map, results, PROMPTS_FURNITURE, overwrite=True)

    # 第4层：小物体
    apply_group(label_map, results, PROMPTS_OBJECTS, overwrite=True)

    label_map = fill_unlabeled_gaps(label_map)

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


def visualize_overlay(
    img_path: str,
    label_map_path: str,
    color_map_path: str,
    out_dir: str,
    alpha: float = 0.45,
) -> str:
    image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
    label_map = np.array(Image.open(label_map_path))
    seg = np.array(Image.open(color_map_path).convert("RGB")).astype(np.float32)

    overlay = image.copy()
    valid = label_map > 0
    overlay[valid] = (image[valid] * (1 - alpha) + seg[valid] * alpha)
    overlay = overlay.clip(0, 255).astype(np.uint8)
    overlay_path = os.path.join(out_dir, "scene_overlay.png")
    Image.fromarray(overlay).save(overlay_path)

    print(f"[INFO] saved overlay: {overlay_path}")
    return overlay_path


def main():
    ensure_dir(OUT_DIR)

    summary_json, original_img = run_segmentation(IMG_PATH, OUT_DIR)
    label_map_path, color_map_path, _ = merge_scene(summary_json, OUT_DIR)
    visualize_overlay(original_img, label_map_path, color_map_path, OUT_DIR, alpha=ALPHA)

    print("\n[DONE]")
    print(f"Outputs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()