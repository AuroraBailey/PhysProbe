import os
import json
from typing import Dict, List, Tuple, Set

import numpy as np
from PIL import Image
import torch

try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# =========================
# 1. 路径配置
# =========================
IMG_PATH = "test.png"   # 改成你的原图路径
OUT_DIR = "scene_seg_outputs"
SCENE_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "configs", "indoor_contextual.json")
DEFAULT_CUDA_INDEX = os.environ.get("SAM3_CUDA_DEVICE", "3")

# overlay 透明度
ALPHA = 0.45
GAP_FILL_PASSES = 3
GAP_FILL_MIN_NEIGHBORS = 4

# 场景先验:
# generic: 默认小而稳的通用词表
# full: 原始大词表（更慢，类间竞争更强）
# bathroom / bedroom / kitchen / living_room / office / dining_room / conference_room / mixed_open_plan ...
SCENE_PROFILE = os.environ.get("SAM3_SCENE_PROFILE", "generic").strip().lower()

PROFILE_ENHANCED_MODE = os.environ.get("SAM3_PROFILE_ENHANCED", "auto").strip().lower()
PROFILE_CORE_SCORE_BONUS = 0.010

ZERO_RESCUE_MODE = os.environ.get("SAM3_ZERO_RESCUE", "on").strip().lower()
ZERO_RESCUE_MIN_AREA = 48
ZERO_RESCUE_MAX_AREA = 12000
ZERO_RESCUE_MAX_AREA_RATIO = 0.03
ZERO_RESCUE_DOMINANCE = 0.82
ZERO_RESCUE_TOUCH_MARGIN = 1
ZERO_RESCUE_CLASSES = {
    "bed",
    "sofa",
    "armchair",
    "chair",
    "table",
    "conference table",
    "desk",
    "cabinet",
    "kitchen cabinet",
    "bookshelf",
    "display shelf",
    "shelf",
    "bathtub",
    "sink",
    "countertop",
    "floor",
    "wall",
}

LARGE_ZERO_RESCUE_MODE = os.environ.get("SAM3_LARGE_ZERO_RESCUE", "on").strip().lower()
LARGE_ZERO_RESCUE_MIN_AREA = 1200
LARGE_ZERO_RESCUE_MAX_AREA_RATIO = 0.45
LARGE_ZERO_RESCUE_DOMINANCE = 0.84
LARGE_ZERO_RESCUE_BORDER_DOMINANCE = 0.90
LARGE_ZERO_RESCUE_MAX_NEIGHBOR_CLASSES = 4
LARGE_ZERO_RESCUE_BORDER_FLOOR_SHARE = 0.32
LARGE_ZERO_RESCUE_BORDER_CEILING_SHARE = 0.26
LARGE_ZERO_RESCUE_CLASSES = {
    "floor",
    "ceiling",
    "wall",
    "cabinet",
    "kitchen cabinet",
    "bookshelf",
    "display shelf",
    "shelf",
    "countertop",
    "table",
    "desk",
    "conference table",
    "bed",
    "sofa",
    "bathtub",
    "refrigerator",
}

# 对 stage2 / wall_focus 的 crop 结果做边界截断检测
# 如果 mask 贴着 crop 边界，说明它大概率是不完整的局部结果
CROP_EDGE_TOUCH_MARGIN = 3
STAGE2_CROP_BORDER_TRIM = 3
STAGE2_INTERNAL_TOUCH_SCORE_PENALTY = 0.08
STAGE2_COMPLETE_SCORE_BONUS = 0.015
STAGE2_RECROP_SCORE_BONUS = 0.025
STAGE2_FALLBACK_SCORE_PENALTY = 0.02
STAGE2_RECROP_PAD_RATIO = 0.28
STAGE2_RECROP_MIN_MARGIN = 64
STAGE2_RECROP_MAX_PER_CLASS = 3
STAGE2_RECROP_TOUCH_EXPAND_RATIO = 0.55
STAGE2_RECROP_OPPOSITE_EXPAND_RATIO = 0.18

PARTIAL_MASK_FILTER_MODE = os.environ.get("SAM3_PARTIAL_MASK_FILTER", "on").strip().lower()
PARTIAL_MASK_KEEP_MIN_SCORE = 0.60
PARTIAL_MASK_KEEP_MAX_TOUCH_COUNT = 1
PARTIAL_MASK_KEEP_MIN_FILL_RATIO = 0.55
PARTIAL_MASK_KEEP_MAX_SPAN_RATIO = 0.72
PARTIAL_MASK_KEEP_ALLOW_RECROP_ONLY = True

# 对边缘目标做一次更稳妥的回退保留。
# 旧版对 furniture / objects 过于宽松，容易把碎片或错误类别也保留下来。
FALLBACK_RULES = {
    "structure": {"score_margin": 0.03, "min_area_ratio": 0.80, "min_area_floor": 2000},
    "openings": {"score_margin": 0.05, "min_area_ratio": 0.60, "min_area_floor": 250},
    "furniture": {"score_margin": 0.06, "min_area_ratio": 0.55, "min_area_floor": 250},
    "objects": {"score_margin": 0.05, "min_area_ratio": 0.50, "min_area_floor": 60},
}

SECOND_STAGE_RELAX_RULES = {
    "structure": {"score_margin": 0.00, "min_area_ratio": 1.00, "min_area_floor": 2000},
    "openings": {"score_margin": 0.03, "min_area_ratio": 0.70, "min_area_floor": 128},
    "furniture": {"score_margin": 0.04, "min_area_ratio": 0.55, "min_area_floor": 96},
    "objects": {"score_margin": 0.04, "min_area_ratio": 0.35, "min_area_floor": 24},
}
SECOND_STAGE_MAX_CROPS = 20
SECOND_STAGE_MIN_CROP_SIDE = 320
SECOND_STAGE_GRID_ROWS = 3
SECOND_STAGE_GRID_COLS = 3
SECOND_STAGE_GRID_CROP_RATIO = 0.58
CLASS_MERGE_SUPPORT_BONUS = 0.015

SECOND_STAGE_SCORE_DELTA = 0.03
SECOND_STAGE_AREA_RATIO = 1.25
HIGH_VALUE_STAGE2_CLASSES = {
    "bed",
    "sofa",
    "cabinet",
    "kitchen cabinet",
    "dresser",
    "bookshelf",
    "display shelf",
    "shelf",
    "desk",
    "dining table",
    "coffee table",
    "side table",
    "conference table",
    "table",
    "office chair",
    "dining chair",
    "armchair",
    "chair",
    "sink",
    "countertop",
    "toilet",
    "bathtub",
    "display screen",
    "piano",
    "container",
    "floor",
    "ceiling",
}

STAGE2_STRUCTURE_RETRY_CLASSES = {"floor", "ceiling"}
STAGE2_STRUCTURE_MIN_AREA_RATIO = {
    "floor": 0.10,
    "ceiling": 0.06,
}

GENERIC_CLASS_SCORE_PENALTY = {
    "container": 0.035,
    "chair": 0.015,
    "table": 0.015,
    "shelf": 0.015,
}
SPECIFIC_CLASS_SCORE_BONUS = {
    "trash bin": 0.020,
    "storage bin": 0.020,
    "laundry basket": 0.020,
    "office chair": 0.015,
    "dining chair": 0.015,
    "conference table": 0.015,
    "coffee table": 0.015,
    "side table": 0.015,
    "display shelf": 0.015,
    "bookshelf": 0.015,
    "kitchen cabinet": 0.015,
}


STAGE2_PROFILE_ONLY_MODE = os.environ.get("SAM3_STAGE2_PROFILE_ONLY", "on").strip().lower()
STAGE2_PARTIAL_COMPLETE_SCORE_MARGIN = 0.10
STAGE2_PARTIAL_COMPLETE_AREA_RATIO = 1.15
STAGE2_PARTIAL_COMPLETE_MIN_EXTRA_AREA = 96
STAGE2_PARTIAL_COMPLETE_MAX_TOUCH_COUNT = 2
STAGE2_PARTIAL_COMPLETE_MIN_CROP_FILL = 0.08
ATTACHED_OBJECT_CLASSES = {
    "display screen",
    "television",
    "mirror",
    "pillow",
    "table lamp",
    "floor lamp",
    "wall sconce",
}

LIGHT_PREPROCESS_MODE = os.environ.get("SAM3_LIGHT_PREPROCESS", "auto").strip().lower()
LIGHT_PREPROCESS_Q_LOW = 1.0
LIGHT_PREPROCESS_Q_HIGH = 99.0
LIGHT_PREPROCESS_GAMMA = 0.95

DEFAULT_STAGE2_ANCHORS = [
    "floor",
    "wall",
    "rug",
    "window",
    "cabinet",
    "dresser",
    "bookshelf",
    "display shelf",
    "shelf",
    "bed",
    "sofa",
    "desk",
    "table",
    "chair",
    "sink",
    "toilet",
    "bathtub",
    "display screen",
    "container",
]

DEFAULT_PROFILE_BACKSTOP_CLASSES = {
    "generic": ["chair", "table", "shelf", "container", "display screen", "sink", "toilet"],
}

DEFAULT_PROFILE_STAGE2_ANCHORS = {
    "generic": ["wall", "floor", "bed", "sofa", "desk", "table", "chair", "cabinet", "shelf", "sink", "toilet"],
}

HOLE_FILL_CLASSES = {
    "bed": 4096,
    "sofa": 2048,
    "armchair": 768,
    "chair": 320,
    "table": 1024,
    "dining table": 1024,
    "conference table": 1600,
    "desk": 1400,
    "cabinet": 1536,
    "kitchen cabinet": 1536,
    "bookshelf": 1280,
    "display shelf": 1280,
    "shelf": 1280,
    "bathtub": 4096,
    "sink": 512,
    "floor": 8192,
    "wall": 8192,
}

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
def normalize_scene_profile_name(scene_profile: str | None) -> str:
    scene_profile = (scene_profile or "generic").strip().lower()
    return scene_profile.replace(" ", "_").replace("-", "_")



def load_scene_vocab(path: str, scene_profile: str | None = None) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    scene_profile = normalize_scene_profile_name(scene_profile)
    profiles = cfg.get("scene_profiles", {})
    groups = profiles.get(scene_profile, cfg["groups"])
    if scene_profile not in profiles:
        print(f"[WARN] unknown scene profile={scene_profile!r}, fallback to default groups")

    prompts_structure = list(groups["structure"])
    prompts_openings = list(groups["openings"])
    prompts_furniture = list(groups["furniture"])
    prompts_objects = list(groups["objects"])
    all_prompts = prompts_structure + prompts_openings + prompts_furniture + prompts_objects

    prompt_to_class = dict(cfg.get("prompt_to_class", {}))
    profile_prompt_to_class = {
        normalize_scene_profile_name(name): dict(values)
        for name, values in cfg.get("profile_prompt_to_class", {}).items()
    }
    prompt_to_class.update(profile_prompt_to_class.get(scene_profile, {}))
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

    profile_backstop_classes = {
        normalize_scene_profile_name(name): list(values)
        for name, values in cfg.get("profile_backstop_classes", {}).items()
    }
    profile_stage2_anchors = {
        normalize_scene_profile_name(name): list(values)
        for name, values in cfg.get("profile_stage2_anchors", {}).items()
    }
    profile_core_classes = {
        normalize_scene_profile_name(name): list(values)
        for name, values in cfg.get("profile_core_classes", {}).items()
    }
    profile_optional_classes = {
        normalize_scene_profile_name(name): list(values)
        for name, values in cfg.get("profile_optional_classes", {}).items()
    }
    profile_band_crops = {
        normalize_scene_profile_name(name): list(values)
        for name, values in cfg.get("profile_band_crops", {}).items()
    }
    default_object_underlay_policy = list(cfg.get("default_object_underlay_policy", ["__blank__", "__structure__", "__openings__"]))
    object_underlay_policies = {
        key: list(values)
        for key, values in cfg.get("object_underlay_policies", {}).items()
    }

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
        "scene_profile": scene_profile,
        "profile_backstop_classes": profile_backstop_classes,
        "profile_stage2_anchors": profile_stage2_anchors,
        "profile_core_classes": profile_core_classes,
        "profile_optional_classes": profile_optional_classes,
        "profile_band_crops": profile_band_crops,
        "default_object_underlay_policy": default_object_underlay_policy,
        "object_underlay_policies": object_underlay_policies,
    }


SCENE_VOCAB = load_scene_vocab(SCENE_VOCAB_PATH, scene_profile=SCENE_PROFILE)
PROMPTS_STRUCTURE = SCENE_VOCAB["prompts_structure"]
PROMPTS_OPENINGS = SCENE_VOCAB["prompts_openings"]
PROMPTS_FURNITURE = SCENE_VOCAB["prompts_furniture"]
PROMPTS_OBJECTS = SCENE_VOCAB["prompts_objects"]
ALL_PROMPTS = SCENE_VOCAB["all_prompts"]
PROMPT_TO_CLASS = SCENE_VOCAB["prompt_to_class"]
PROFILE_BACKSTOP_CLASSES = {
    **DEFAULT_PROFILE_BACKSTOP_CLASSES,
    **SCENE_VOCAB.get("profile_backstop_classes", {}),
}
PROFILE_STAGE2_ANCHORS = {
    **DEFAULT_PROFILE_STAGE2_ANCHORS,
    **SCENE_VOCAB.get("profile_stage2_anchors", {}),
}
ACTIVE_PROFILE_BACKSTOP_CLASSES = set(PROFILE_BACKSTOP_CLASSES.get(SCENE_PROFILE, PROFILE_BACKSTOP_CLASSES.get("generic", [])))
ACTIVE_PROFILE_STAGE2_ANCHORS = PROFILE_STAGE2_ANCHORS.get(SCENE_PROFILE, PROFILE_STAGE2_ANCHORS.get("generic", DEFAULT_STAGE2_ANCHORS))
PROFILE_CORE_CLASSES = SCENE_VOCAB.get("profile_core_classes", {})
PROFILE_OPTIONAL_CLASSES = SCENE_VOCAB.get("profile_optional_classes", {})
PROFILE_BAND_CROPS = SCENE_VOCAB.get("profile_band_crops", {})
ACTIVE_PROFILE_CORE_CLASSES = set(PROFILE_CORE_CLASSES.get(SCENE_PROFILE, []))
ACTIVE_PROFILE_OPTIONAL_CLASSES = set(PROFILE_OPTIONAL_CLASSES.get(SCENE_PROFILE, []))
ACTIVE_PROFILE_BAND_CROPS = PROFILE_BAND_CROPS.get(SCENE_PROFILE, [])
DEFAULT_OBJECT_UNDERLAY_POLICY = list(SCENE_VOCAB.get("default_object_underlay_policy", ["__blank__", "__structure__", "__openings__"]))
OBJECT_UNDERLAY_POLICIES = {
    PROMPT_TO_CLASS.get(prompt, prompt): list(values)
    for prompt, values in SCENE_VOCAB.get("object_underlay_policies", {}).items()
}
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


def should_apply_light_preprocess(img_path: str) -> bool:
    mode = LIGHT_PREPROCESS_MODE
    if mode in {"0", "false", "off", "none", "disable", "disabled"}:
        return False
    if mode in {"1", "true", "on", "enable", "enabled", "always"}:
        return True
    name = os.path.basename(img_path).lower()
    return "difflight" in name or "maindifflight" in img_path.lower()


def preprocess_image_for_segmentation(image: Image.Image, img_path: str) -> Image.Image:
    if not should_apply_light_preprocess(img_path):
        return image
    arr = np.asarray(image).astype(np.float32) / 255.0
    q_low = np.percentile(arr, LIGHT_PREPROCESS_Q_LOW)
    q_high = np.percentile(arr, LIGHT_PREPROCESS_Q_HIGH)
    if q_high <= q_low + 1e-6:
        return image
    arr = np.clip((arr - q_low) / (q_high - q_low), 0.0, 1.0)
    arr = np.power(arr, LIGHT_PREPROCESS_GAMMA)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def get_active_stage2_anchor_classes() -> List[str]:
    ordered = []
    seen = set()
    for class_name in list(ACTIVE_PROFILE_STAGE2_ANCHORS) + list(DEFAULT_STAGE2_ANCHORS):
        if class_name in CLASS_TO_ID and class_name not in seen:
            ordered.append(class_name)
            seen.add(class_name)
    return ordered


def get_profile_priority_classes() -> Set[str]:
    core = {class_name for class_name in ACTIVE_PROFILE_CORE_CLASSES if class_name in CLASS_TO_ID}
    if core:
        return core | {class_name for class_name in ACTIVE_PROFILE_BACKSTOP_CLASSES if class_name in CLASS_TO_ID}
    return {class_name for class_name in ACTIVE_PROFILE_BACKSTOP_CLASSES if class_name in CLASS_TO_ID}


def is_stage2_profile_only_enabled() -> bool:
    mode = STAGE2_PROFILE_ONLY_MODE
    return mode not in {"0", "false", "off", "none", "disable", "disabled"}


def get_active_stage2_search_space_classes() -> Set[str]:
    classes = {PROMPT_TO_CLASS.get(prompt, prompt) for prompt in ALL_PROMPTS if PROMPT_TO_CLASS.get(prompt, prompt) in CLASS_TO_ID}
    classes |= {class_name for class_name in ACTIVE_PROFILE_CORE_CLASSES if class_name in CLASS_TO_ID}
    classes |= {class_name for class_name in ACTIVE_PROFILE_OPTIONAL_CLASSES if class_name in CLASS_TO_ID}
    classes |= {class_name for class_name in ACTIVE_PROFILE_BACKSTOP_CLASSES if class_name in CLASS_TO_ID}
    classes |= {class_name for class_name in ACTIVE_PROFILE_STAGE2_ANCHORS if class_name in CLASS_TO_ID}
    return classes


def is_profile_enhanced_enabled() -> bool:
    mode = PROFILE_ENHANCED_MODE
    if mode in {"0", "false", "off", "none", "disable", "disabled"}:
        return False
    if mode in {"1", "true", "on", "enable", "enabled", "always"}:
        return True
    return SCENE_PROFILE not in {"generic", "full"}


def get_active_profile_band_crops() -> List[dict]:
    if not is_profile_enhanced_enabled():
        return []
    return list(ACTIVE_PROFILE_BAND_CROPS)


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


def bbox_fill_ratio(mask_bool: np.ndarray, bbox: Tuple[int, int, int, int] | None = None) -> float:
    if bbox is None:
        bbox = mask_bbox_xyxy(mask_bool)
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    box_area = max(1, (x2 - x1) * (y2 - y1))
    return float(mask_bool.sum()) / float(box_area)


def get_crop_touch_info(
    mask_bool: np.ndarray,
    crop_box: Tuple[int, int, int, int] | None = None,
    full_shape: Tuple[int, int] | None = None,
    margin: int = CROP_EDGE_TOUCH_MARGIN,
) -> dict:
    """
    只在 crop-local mask 上调用。
    判断 mask 是否贴着 crop 边界，并区分：
    - 贴的是 crop 的内部边界（更可能是被裁断）
    - 贴的是整张图的真实边界（不一定是不完整）
    """
    h, w = mask_bool.shape
    if h == 0 or w == 0 or not mask_bool.any():
        return {
            "touch_crop_boundary": False,
            "touch_sides": [],
            "touch_count": 0,
            "touch_internal_crop_boundary": False,
            "touch_internal_sides": [],
            "touch_internal_count": 0,
            "touch_image_boundary": False,
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

    internal_touch_sides: List[str] = []
    image_touch = False
    if crop_box is not None and full_shape is not None and len(crop_box) == 4:
        full_h, full_w = full_shape
        x1, y1, x2, y2 = [int(v) for v in crop_box]
        for side in touch_sides:
            if side == "left":
                if x1 > 0:
                    internal_touch_sides.append(side)
                else:
                    image_touch = True
            elif side == "right":
                if x2 < full_w:
                    internal_touch_sides.append(side)
                else:
                    image_touch = True
            elif side == "top":
                if y1 > 0:
                    internal_touch_sides.append(side)
                else:
                    image_touch = True
            elif side == "bottom":
                if y2 < full_h:
                    internal_touch_sides.append(side)
                else:
                    image_touch = True

    return {
        "touch_crop_boundary": len(touch_sides) > 0,
        "touch_sides": touch_sides,
        "touch_count": len(touch_sides),
        "touch_internal_crop_boundary": len(internal_touch_sides) > 0,
        "touch_internal_sides": internal_touch_sides,
        "touch_internal_count": len(internal_touch_sides),
        "touch_image_boundary": image_touch,
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



def add_profile_band_crops(
    crops: List[dict],
    seen: set,
    width: int,
    height: int,
) -> None:
    for spec in get_active_profile_band_crops():
        name = str(spec.get("name", "profile_band"))
        box = spec.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1 = int(round(float(box[0]) * width))
        y1 = int(round(float(box[1]) * height))
        x2 = int(round(float(box[2]) * width))
        y2 = int(round(float(box[3]) * height))
        add_crop_if_new(crops, seen, name, normalize_crop_box((x1, y1, x2, y2), width, height))



def stage2_item_is_complete(class_name: str, item: dict) -> bool:
    del class_name
    return not bool(item.get("touch_internal_crop_boundary", False))


def stage2_results_are_complete(class_name: str, prompt_results: List[dict]) -> bool:
    return any(stage2_item_is_complete(class_name, item) for item in prompt_results)


def stage2_result_rank_key(item: dict) -> Tuple[int, int, int, float, int]:
    stage = str(item.get("stage", ""))
    internal_touch_count = int(item.get("touch_internal_count", item.get("touch_count", 0)))
    return (
        1 if not item.get("touch_internal_crop_boundary", False) else 0,
        1 if stage.startswith("stage2_recrop") else 0,
        -internal_touch_count,
        float(item.get("score", 0.0)),
        int(item.get("area", 0)),
    )


def choose_best_stage2_result(prompt_results: List[dict]) -> dict | None:
    if not prompt_results:
        return None
    return max(prompt_results, key=stage2_result_rank_key)


def build_second_stage_recrop(
    item: dict,
    width: int,
    height: int,
) -> Tuple[int, int, int, int] | None:
    mask_path = item.get("mask_path")
    if not mask_path or not os.path.exists(mask_path):
        return None
    mask = load_binary_mask(mask_path)
    bbox = mask_bbox_xyxy(mask)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    base_pad_x = max(STAGE2_RECROP_MIN_MARGIN, int(round(box_w * STAGE2_RECROP_PAD_RATIO)))
    base_pad_y = max(STAGE2_RECROP_MIN_MARGIN, int(round(box_h * STAGE2_RECROP_PAD_RATIO)))

    expand_left = base_pad_x
    expand_right = base_pad_x
    expand_top = base_pad_y
    expand_bottom = base_pad_y

    touch_sides = set(item.get("touch_internal_sides", []) or [])
    strong_pad_x = max(STAGE2_RECROP_MIN_MARGIN, int(round(box_w * STAGE2_RECROP_TOUCH_EXPAND_RATIO)))
    strong_pad_y = max(STAGE2_RECROP_MIN_MARGIN, int(round(box_h * STAGE2_RECROP_TOUCH_EXPAND_RATIO)))
    weak_pad_x = max(base_pad_x, int(round(box_w * STAGE2_RECROP_OPPOSITE_EXPAND_RATIO)))
    weak_pad_y = max(base_pad_y, int(round(box_h * STAGE2_RECROP_OPPOSITE_EXPAND_RATIO)))

    if "left" in touch_sides:
        expand_left = max(expand_left, strong_pad_x)
        expand_right = max(expand_right, weak_pad_x)
    if "right" in touch_sides:
        expand_right = max(expand_right, strong_pad_x)
        expand_left = max(expand_left, weak_pad_x)
    if "top" in touch_sides:
        expand_top = max(expand_top, strong_pad_y)
        expand_bottom = max(expand_bottom, weak_pad_y)
    if "bottom" in touch_sides:
        expand_bottom = max(expand_bottom, strong_pad_y)
        expand_top = max(expand_top, weak_pad_y)

    min_side = max(SECOND_STAGE_MIN_CROP_SIDE, int(round(max(box_w, box_h) * 1.45)))
    return normalize_crop_box(
        (x1 - expand_left, y1 - expand_top, x2 + expand_right, y2 + expand_bottom),
        width,
        height,
        min_side=min_side,
    )


def build_second_stage_crops(
    image: Image.Image,
    stage1_results: List[dict],
) -> List[dict]:
    width, height = image.size
    crops: List[dict] = []
    seen = set()

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
    add_crop_if_new(
        crops,
        seen,
        "top_band",
        normalize_crop_box((0, 0, width, int(height * 0.46)), width, height),
    )
    add_crop_if_new(
        crops,
        seen,
        "bottom_band",
        normalize_crop_box((0, int(height * 0.52), width, height), width, height),
    )
    add_profile_band_crops(crops, seen, width, height)
    add_dense_grid_crops(crops, seen, width, height)

    anchor_classes = get_active_stage2_anchor_classes()
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
    if class_name in STAGE2_STRUCTURE_RETRY_CLASSES:
        return True
    if class_group(class_name) == "structure":
        return False
    if is_stage2_profile_only_enabled() and class_name not in get_active_stage2_search_space_classes():
        return False
    if class_name in ACTIVE_PROFILE_CORE_CLASSES:
        return True
    if class_name in get_profile_priority_classes():
        return True
    if class_name in HIGH_VALUE_STAGE2_CLASSES:
        return True
    conf = THRESHOLDS.get(class_name, {"min_area": 1000})
    max_ratio = MAX_AREA_RATIO.get(class_name, 1.0)
    return int(conf["min_area"]) <= 1600 or float(max_ratio) <= 0.20



def get_second_stage_target_classes(stage1_results: List[dict]) -> List[str]:
    present = {canonical_label(item["prompt"]) for item in stage1_results}
    class_stats = summarize_class_results(stage1_results)
    best_area = get_best_area_by_class(stage1_results)
    image_area = 1
    if stage1_results:
        # run_segmentation summary uses the original full-image masks, so area is comparable here.
        image_area = max((int(item.get("area", 0)) for item in stage1_results), default=1)
        image_area = max(image_area, 1)
    priority = get_profile_priority_classes()
    optional = {class_name for class_name in ACTIVE_PROFILE_OPTIONAL_CLASSES if class_name in CLASS_TO_ID}
    search_space = get_active_stage2_search_space_classes() | STAGE2_STRUCTURE_RETRY_CLASSES

    priority_targets: List[str] = []
    normal_targets: List[str] = []
    optional_targets: List[str] = []
    for class_name in [name for name in CLASS_ORDER if name in search_space]:
        if not should_retry_class_in_stage2(class_name):
            continue
        need_retry = False
        if class_name in STAGE2_STRUCTURE_RETRY_CLASSES:
            area_ratio = best_area.get(class_name, 0) / max(307200, image_area)
            need_retry = class_name not in present or area_ratio < STAGE2_STRUCTURE_MIN_AREA_RATIO.get(class_name, 0.05)
        elif class_name not in present:
            need_retry = True
        else:
            stats = class_stats.get(class_name)
            if stats and should_retry_low_quality_present_class(class_name, stats):
                need_retry = True
        if not need_retry:
            continue
        if class_name in priority or class_name in STAGE2_STRUCTURE_RETRY_CLASSES:
            priority_targets.append(class_name)
        elif class_name in optional:
            optional_targets.append(class_name)
        else:
            normal_targets.append(class_name)
    return priority_targets + normal_targets + optional_targets


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

def summarize_class_results(results: List[dict]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for item in results:
        class_name = canonical_label(item["prompt"])
        stats = summary.setdefault(
            class_name,
            {
                "num_results": 0,
                "best_score": -1.0,
                "best_area": 0,
                "has_non_fallback": False,
            },
        )
        stats["num_results"] += 1
        score = float(item["score"])
        area = int(item["area"])
        if score > stats["best_score"]:
            stats["best_score"] = score
        if area > stats["best_area"]:
            stats["best_area"] = area
        if not item.get("fallback", False):
            stats["has_non_fallback"] = True
    return summary


def should_retry_low_quality_present_class(class_name: str, class_stats: dict) -> bool:
    if class_name not in HIGH_VALUE_STAGE2_CLASSES:
        return False
    if class_group(class_name) == "structure":
        return False

    score_thresh, min_area = get_effective_thresholds(class_name, second_stage=False)
    best_score = float(class_stats.get("best_score", -1.0))
    best_area = int(class_stats.get("best_area", 0))
    has_non_fallback = bool(class_stats.get("has_non_fallback", False))

    low_score = best_score < (score_thresh + SECOND_STAGE_SCORE_DELTA)
    low_area = best_area < max(int(min_area * SECOND_STAGE_AREA_RATIO), min_area + 64)
    only_fallback = not has_non_fallback
    return only_fallback or low_score or low_area


def get_class_ids_for_prompts(prompts: List[str]) -> Set[int]:
    class_ids: Set[int] = set()
    for prompt in prompts:
        class_name = canonical_label(prompt)
        cid = CLASS_TO_ID.get(class_name)
        if cid is not None:
            class_ids.add(cid)
    return class_ids


def get_group_class_ids(group_name: str) -> Set[int]:
    if group_name == "structure":
        return get_class_ids_for_prompts(PROMPTS_STRUCTURE)
    if group_name == "openings":
        return get_class_ids_for_prompts(PROMPTS_OPENINGS)
    if group_name == "furniture":
        return get_class_ids_for_prompts(PROMPTS_FURNITURE)
    if group_name == "objects":
        return get_class_ids_for_prompts(PROMPTS_OBJECTS)
    return set()


def resolve_underlay_policy_tokens(tokens: List[str]) -> Set[int]:
    class_ids: Set[int] = set()
    for token in tokens:
        if token == "__blank__":
            class_ids.add(0)
        elif token == "__structure__":
            class_ids |= get_group_class_ids("structure")
        elif token == "__openings__":
            class_ids |= get_group_class_ids("openings")
        elif token == "__furniture__":
            class_ids |= get_group_class_ids("furniture")
        elif token == "__objects__":
            class_ids |= get_group_class_ids("objects")
        else:
            cid = CLASS_TO_ID.get(token)
            if cid is not None:
                class_ids.add(cid)
    return class_ids


def get_object_underlay_ids_for_class(class_name: str) -> Set[int]:
    tokens = OBJECT_UNDERLAY_POLICIES.get(class_name, DEFAULT_OBJECT_UNDERLAY_POLICY)
    return resolve_underlay_policy_tokens(tokens)


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


def trim_mask_at_internal_crop_boundaries(
    full_mask: np.ndarray,
    crop_box: Tuple[int, int, int, int] | None,
    full_shape: Tuple[int, int] | None,
    touch_info: dict | None,
    trim_px: int = STAGE2_CROP_BORDER_TRIM,
) -> np.ndarray:
    if crop_box is None or full_shape is None or touch_info is None or trim_px <= 0:
        return full_mask
    internal_sides = list(touch_info.get("touch_internal_sides", []))
    if not internal_sides:
        return full_mask

    trimmed = full_mask.copy()
    x1, y1, x2, y2 = [int(v) for v in crop_box]
    full_h, full_w = full_shape

    if "left" in internal_sides:
        trimmed[:, max(0, x1):min(full_w, x1 + trim_px)] = False
    if "right" in internal_sides:
        trimmed[:, max(0, x2 - trim_px):min(full_w, x2)] = False
    if "top" in internal_sides:
        trimmed[max(0, y1):min(full_h, y1 + trim_px), :] = False
    if "bottom" in internal_sides:
        trimmed[max(0, y2 - trim_px):min(full_h, y2), :] = False
    return trimmed


def get_effective_merge_score(item: dict) -> float:
    score = float(item.get("score", 0.0))
    stage = str(item.get("stage", ""))
    if item.get("fallback", False):
        score -= STAGE2_FALLBACK_SCORE_PENALTY
    if item.get("touch_internal_crop_boundary", False):
        penalty = STAGE2_INTERNAL_TOUCH_SCORE_PENALTY
        if stage.startswith("stage2_recrop"):
            penalty = max(0.04, penalty - 0.03)
        score -= penalty
    elif stage.startswith("stage2_recrop"):
        score += STAGE2_RECROP_SCORE_BONUS
    elif stage in {"stage2", "wall_focus"}:
        score += STAGE2_COMPLETE_SCORE_BONUS
    return score


def get_class_merge_adjustment(class_name: str) -> float:
    adjust = 0.0
    adjust -= GENERIC_CLASS_SCORE_PENALTY.get(class_name, 0.0)
    adjust += SPECIFIC_CLASS_SCORE_BONUS.get(class_name, 0.0)
    return adjust


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
            "touch_internal_crop_boundary": False,
            "touch_internal_sides": [],
            "touch_internal_count": 0,
            "touch_image_boundary": False,
        }
        if crop_box is not None:
            touch_info = get_crop_touch_info(crop_local_mask_bool, crop_box=crop_box, full_shape=full_shape)

        crop_local_bbox = mask_bbox_xyxy(crop_local_mask_bool)
        crop_local_fill_ratio = bbox_fill_ratio(crop_local_mask_bool, crop_local_bbox)

        mask_bool = crop_local_mask_bool
        if crop_box is not None and full_shape is not None:
            mask_bool = make_full_image_mask(mask_bool, crop_box, full_shape)
            mask_bool = trim_mask_at_internal_crop_boundaries(mask_bool, crop_box, full_shape, touch_info)

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
            if crop_local_bbox is not None:
                item["crop_local_bbox"] = list(crop_local_bbox)
            item["crop_local_fill_ratio"] = float(crop_local_fill_ratio)
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
            if crop_local_bbox is not None:
                item["crop_local_bbox"] = list(crop_local_bbox)
            item["crop_local_fill_ratio"] = float(crop_local_fill_ratio)
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
    seg_image = preprocess_image_for_segmentation(image, img_path)
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

    state = set_image_state(processor, seg_image, device)

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
    second_stage_crops = build_second_stage_crops(seg_image, all_results)
    if second_stage_target_classes and second_stage_crops:
        print(
            f"\n[INFO] stage2 enabled: missing_classes={len(second_stage_target_classes)}, "
            f"crops={len(second_stage_crops)}"
        )
        remaining_classes = list(second_stage_target_classes)
        full_shape = (height, width)
        crop_queue = list(second_stage_crops)
        crop_index = 0
        scheduled_recrops: Set[Tuple[str, Tuple[int, int, int, int]]] = set()
        class_recrop_counts = {class_name: 0 for class_name in second_stage_target_classes}

        while crop_index < len(crop_queue) and remaining_classes:
            crop = crop_queue[crop_index]
            crop_index += 1

            x1, y1, x2, y2 = crop["box"]
            crop_image = seg_image.crop((x1, y1, x2, y2))
            crop_state = set_image_state(processor, crop_image, device)
            print(
                f"\n[INFO] stage2 crop={crop['name']} box={crop['box']} "
                f"remaining_classes={len(remaining_classes)}"
            )

            for class_name in remaining_classes.copy():
                prompt_dir = os.path.join(out_dir, "masks", class_name.replace(" ", "_"))
                ensure_dir(prompt_dir)

                class_found_complete = False
                class_found_any = False
                best_partial_item = None

                for prompt in get_prompts_for_canonical_class(class_name):
                    output = run_prompt(processor, crop_state, prompt, device)
                    prompt_results, total = collect_results_for_prompt(
                        prompt=prompt,
                        output=output,
                        image_area=image_area,
                        prompt_dir=prompt_dir,
                        second_stage=True,
                        stage_tag="stage2_recrop" if crop.get("is_recrop", False) else "stage2",
                        crop_name=crop["name"],
                        crop_box=crop["box"],
                        full_shape=full_shape,
                    )
                    if not prompt_results:
                        continue

                    all_results.extend(prompt_results)
                    class_found_any = True

                    has_complete_result = stage2_results_are_complete(class_name, prompt_results)
                    best_prompt_item = choose_best_stage2_result(prompt_results)
                    if best_prompt_item is not None and (
                        best_partial_item is None
                        or stage2_result_rank_key(best_prompt_item) > stage2_result_rank_key(best_partial_item)
                    ):
                        best_partial_item = best_prompt_item

                    print(
                        f"[INFO] stage2 class={class_name}, prompt={prompt}, "
                        f"crop={crop['name']}, total={total}, kept={len(prompt_results)}, "
                        f"complete={has_complete_result}"
                    )

                    if has_complete_result:
                        class_found_complete = True
                        break

                if class_found_complete:
                    if class_name in remaining_classes:
                        remaining_classes.remove(class_name)
                    continue

                if not class_found_any:
                    continue

                if (
                    best_partial_item is not None
                    and best_partial_item.get("touch_internal_crop_boundary", False)
                    and class_recrop_counts.get(class_name, 0) < STAGE2_RECROP_MAX_PER_CLASS
                ):
                    recrop_box = build_second_stage_recrop(best_partial_item, width, height)
                    if recrop_box is not None:
                        recrop_key = (class_name, tuple(int(v) for v in recrop_box))
                        if recrop_key not in scheduled_recrops:
                            scheduled_recrops.add(recrop_key)
                            class_recrop_counts[class_name] = class_recrop_counts.get(class_name, 0) + 1
                            crop_queue.append({
                                "name": f"recrop_{class_name.replace(' ', '_')}_{class_recrop_counts[class_name]}",
                                "box": tuple(int(v) for v in recrop_box),
                                "is_recrop": True,
                                "target_class": class_name,
                            })
                            print(
                                f"[INFO] stage2 schedule recrop class={class_name}, "
                                f"from={crop['name']} -> box={tuple(int(v) for v in recrop_box)}"
                            )

                print(
                    f"[INFO] stage2 class={class_name} only got internal-boundary / partial results "
                    f"on crop={crop['name']}, keep searching next crops"
                )

    if should_enable_wall_focus(all_results, image_area):
        wall_focus_targets = get_wall_focus_target_classes(all_results, image_area)
        wall_focus_crops = {crop["name"]: crop for crop in build_wall_focus_crops(seg_image)}
        if wall_focus_targets and wall_focus_crops:
            print(
                f"\n[INFO] wall_focus enabled: targets={wall_focus_targets}, "
                f"crops={list(wall_focus_crops)}"
            )
            full_shape = (height, width)
            crop_states = {}
            for crop_name, crop in wall_focus_crops.items():
                x1, y1, x2, y2 = crop["box"]
                crop_image = seg_image.crop((x1, y1, x2, y2))
                crop_states[crop_name] = set_image_state(processor, crop_image, device)

            for class_name in wall_focus_targets:
                class_found_complete = False
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
                        if not prompt_results:
                            continue
                        all_results.extend(prompt_results)
                        has_complete_result = stage2_results_are_complete(class_name, prompt_results)
                        print(
                            f"[INFO] wall_focus class={class_name}, prompt={prompt}, "
                            f"crop={crop_name}, total={total}, kept={len(prompt_results)}, "
                            f"complete={has_complete_result}"
                        )
                        if has_complete_result:
                            class_found_complete = True
                            break
                    if class_found_complete:
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


def binary_fill_holes_limited(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    if not mask.any() or max_hole_area <= 0:
        return mask
    if ndi is None:
        return mask
    filled = ndi.binary_fill_holes(mask)
    holes = filled & (~mask)
    if not holes.any():
        return mask
    labeled, num = ndi.label(holes)
    result = mask.copy()
    for idx in range(1, num + 1):
        comp = labeled == idx
        if int(comp.sum()) <= max_hole_area:
            result[comp] = True
    return result


def repair_large_class_holes(label_map: np.ndarray) -> np.ndarray:
    if ndi is None:
        return label_map
    repaired = label_map.copy()
    for class_name, area_threshold in HOLE_FILL_CLASSES.items():
        cid = CLASS_TO_ID.get(class_name)
        if cid is None:
            continue
        mask = repaired == cid
        if int(mask.sum()) < max(64, area_threshold // 8):
            continue
        structure = np.ones((3, 3), dtype=bool)
        closed = ndi.binary_closing(mask, structure=structure, iterations=1)
        filled = binary_fill_holes_limited(closed, max_hole_area=area_threshold)
        repaired[filled] = cid
    return repaired



def should_enable_zero_region_rescue() -> bool:
    mode = ZERO_RESCUE_MODE
    return mode not in {"0", "false", "off", "none", "disable", "disabled"}


def touches_image_border(mask: np.ndarray, margin: int = ZERO_RESCUE_TOUCH_MARGIN) -> bool:
    if not mask.any():
        return False
    return (
        mask[:margin, :].any()
        or mask[-margin:, :].any()
        or mask[:, :margin].any()
        or mask[:, -margin:].any()
    )


def rescue_zero_regions(label_map: np.ndarray) -> np.ndarray:
    if ndi is None or not should_enable_zero_region_rescue():
        return label_map
    refined = label_map.copy()
    zero_mask = refined == 0
    if not zero_mask.any():
        return refined

    labeled, num = ndi.label(zero_mask)
    image_area = refined.shape[0] * refined.shape[1]
    max_area = min(ZERO_RESCUE_MAX_AREA, int(image_area * ZERO_RESCUE_MAX_AREA_RATIO))
    structure = np.ones((3, 3), dtype=bool)

    allowed_rescue_ids = {CLASS_TO_ID[name] for name in ZERO_RESCUE_CLASSES if name in CLASS_TO_ID}
    for idx in range(1, num + 1):
        comp = labeled == idx
        area = int(comp.sum())
        if area < ZERO_RESCUE_MIN_AREA or area > max_area:
            continue
        if touches_image_border(comp):
            continue

        dilated = ndi.binary_dilation(comp, structure=structure, iterations=1)
        ring = dilated & (~comp)
        neighbor_vals = refined[ring]
        neighbor_vals = neighbor_vals[neighbor_vals != 0]
        if neighbor_vals.size == 0:
            continue

        vals, counts = np.unique(neighbor_vals, return_counts=True)
        best_idx = int(np.argmax(counts))
        best_cid = int(vals[best_idx])
        dominance = float(counts[best_idx]) / float(counts.sum())
        if best_cid not in allowed_rescue_ids:
            continue
        if dominance < ZERO_RESCUE_DOMINANCE:
            continue
        refined[comp] = best_cid
    return refined


def should_enable_large_zero_region_rescue() -> bool:
    mode = LARGE_ZERO_RESCUE_MODE
    return mode not in {"0", "false", "off", "none", "disable", "disabled"}


def rescue_large_zero_regions(label_map: np.ndarray) -> np.ndarray:
    if ndi is None or not should_enable_large_zero_region_rescue():
        return label_map
    refined = label_map.copy()
    zero_mask = refined == 0
    if not zero_mask.any():
        return refined

    labeled, num = ndi.label(zero_mask)
    image_area = refined.shape[0] * refined.shape[1]
    max_area = int(image_area * LARGE_ZERO_RESCUE_MAX_AREA_RATIO)
    structure = np.ones((3, 3), dtype=bool)
    allowed_rescue_ids = {CLASS_TO_ID[name] for name in LARGE_ZERO_RESCUE_CLASSES if name in CLASS_TO_ID}
    id_to_name = {cid: name for name, cid in CLASS_TO_ID.items()}

    for idx in range(1, num + 1):
        comp = labeled == idx
        area = int(comp.sum())
        if area < LARGE_ZERO_RESCUE_MIN_AREA or area > max_area:
            continue

        border_touch = touches_image_border(comp)
        dilated = ndi.binary_dilation(comp, structure=structure, iterations=2)
        ring = dilated & (~comp)
        neighbor_vals = refined[ring]
        neighbor_vals = neighbor_vals[neighbor_vals != 0]
        if neighbor_vals.size == 0:
            continue

        vals, counts = np.unique(neighbor_vals, return_counts=True)
        order = np.argsort(-counts)
        vals = vals[order]
        counts = counts[order]
        if len(vals) > LARGE_ZERO_RESCUE_MAX_NEIGHBOR_CLASSES:
            continue

        best_cid = int(vals[0])
        if best_cid not in allowed_rescue_ids:
            continue
        dominance = float(counts[0]) / float(counts.sum())
        best_name = id_to_name.get(best_cid, "")

        threshold = LARGE_ZERO_RESCUE_BORDER_DOMINANCE if border_touch else LARGE_ZERO_RESCUE_DOMINANCE
        if border_touch and best_name == "floor" and comp[-1, :].any():
            threshold = min(threshold, LARGE_ZERO_RESCUE_BORDER_FLOOR_SHARE)
        if border_touch and best_name == "ceiling" and comp[:1, :].any():
            threshold = min(threshold, LARGE_ZERO_RESCUE_BORDER_CEILING_SHARE)

        if dominance < threshold:
            continue
        refined[comp] = best_cid
    return refined


def apply_group(
    label_map: np.ndarray,
    items: List[dict],
    allowed_prompts: List[str],
    overwrite: bool = True,
    allowed_underlay_ids: Set[int] | None = None,
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
            score = get_effective_merge_score(item)
            class_support_map[mask] += 1
            update = mask & (score > class_score_map)
            class_score_map[update] = score

        valid = class_score_map > -np.inf
        if not valid.any():
            continue

        compete_score_map = class_score_map.copy()
        support_bonus = np.clip(class_support_map.astype(np.float32) - 1.0, 0.0, 3.0)
        compete_score_map[valid] += support_bonus[valid] * CLASS_MERGE_SUPPORT_BONUS
        compete_score_map[valid] += get_class_merge_adjustment(class_name)
        if class_name in ACTIVE_PROFILE_CORE_CLASSES:
            compete_score_map[valid] += PROFILE_CORE_SCORE_BONUS

        cid = CLASS_TO_ID[class_name]
        update = valid & (compete_score_map > group_score_map)
        group_label_map[update] = cid
        group_score_map[update] = compete_score_map[update]

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



def apply_object_layer(
    label_map: np.ndarray,
    items: List[dict],
) -> None:
    object_items = [x for x in items if x["prompt"] in PROMPTS_OBJECTS]
    if not object_items:
        return

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


def should_enable_partial_mask_filter() -> bool:
    mode = PARTIAL_MASK_FILTER_MODE
    return mode not in {"0", "false", "off", "none", "disable", "disabled"}


def is_internal_partial_stage_item(item: dict) -> bool:
    if not bool(item.get("touch_internal_crop_boundary", False)):
        return False
    stage = str(item.get("stage", ""))
    return stage.startswith("stage2") or stage == "wall_focus"


def is_merge_safe_partial_item(class_name: str, item: dict) -> bool:
    if not is_internal_partial_stage_item(item):
        return True
    if class_group(class_name) not in {"furniture", "objects"}:
        return True
    if PARTIAL_MASK_KEEP_ALLOW_RECROP_ONLY and not str(item.get("stage", "")).startswith("stage2_recrop"):
        return False
    if float(item.get("score", 0.0)) < PARTIAL_MASK_KEEP_MIN_SCORE:
        return False
    if int(item.get("touch_internal_count", 0)) > PARTIAL_MASK_KEEP_MAX_TOUCH_COUNT:
        return False

    crop_box = item.get("crop_box") or []
    local_bbox = item.get("crop_local_bbox") or []
    if len(crop_box) == 4 and len(local_bbox) == 4:
        crop_w = max(1, int(crop_box[2]) - int(crop_box[0]))
        crop_h = max(1, int(crop_box[3]) - int(crop_box[1]))
        box_w = max(1, int(local_bbox[2]) - int(local_bbox[0]))
        box_h = max(1, int(local_bbox[3]) - int(local_bbox[1]))
        fill_ratio = float(item.get("crop_local_fill_ratio", 0.0))
        if fill_ratio < PARTIAL_MASK_KEEP_MIN_FILL_RATIO:
            return False
        touch_sides = set(item.get("touch_internal_sides", []) or [])
        if ("left" in touch_sides or "right" in touch_sides) and (box_w / crop_w) > PARTIAL_MASK_KEEP_MAX_SPAN_RATIO:
            return False
        if ("top" in touch_sides or "bottom" in touch_sides) and (box_h / crop_h) > PARTIAL_MASK_KEEP_MAX_SPAN_RATIO:
            return False
    return True


def filter_results_before_merge(results: List[dict]) -> List[dict]:
    if not should_enable_partial_mask_filter():
        return results

    kept: List[dict] = []
    items_by_class: Dict[str, List[dict]] = {}
    for item in results:
        items_by_class.setdefault(canonical_label(item["prompt"]), []).append(item)

    for class_name, class_items in items_by_class.items():
        if class_group(class_name) not in {"furniture", "objects"}:
            kept.extend(class_items)
            continue

        complete_items = [x for x in class_items if not is_internal_partial_stage_item(x)]
        partial_items = [x for x in class_items if is_internal_partial_stage_item(x)]

        if complete_items:
            kept.extend(complete_items)
            salvage = [x for x in partial_items if is_merge_safe_partial_item(class_name, x)]
            if salvage:
                best_salvage = max(salvage, key=stage2_result_rank_key)
                kept.append(best_salvage)
            continue

        safe_partial_items = [x for x in partial_items if is_merge_safe_partial_item(class_name, x)]
        if safe_partial_items:
            best_partial = max(safe_partial_items, key=stage2_result_rank_key)
            kept.append(best_partial)

    return kept


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
    results = filter_results_before_merge(data["results"])

    label_map = np.zeros((height, width), dtype=np.uint8)

    # 第1层：结构
    apply_group(label_map, results, PROMPTS_STRUCTURE, overwrite=True)

    # 第2层：门窗开口，覆盖结构
    apply_group(label_map, results, PROMPTS_OPENINGS, overwrite=True)

    # 第3层：家具
    apply_group(label_map, results, PROMPTS_FURNITURE, overwrite=True)

    # 第4层：小物体
    # 按 object 类别分别控制允许覆盖的底层类别：
    # 例如 display screen / mirror / pillow 允许覆盖部分家具或墙面，
    # 而 container / trash bin 等仍保持更保守的覆盖策略。
    apply_object_layer(label_map, results)

    label_map = fill_unlabeled_gaps(label_map)
    label_map = repair_large_class_holes(label_map)
    label_map = rescue_zero_regions(label_map)
    label_map = rescue_large_zero_regions(label_map)
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