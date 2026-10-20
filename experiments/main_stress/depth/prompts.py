"""Shared Gemini-derived depth instructions for all main/stress models."""
from __future__ import annotations

from typing import Any, Dict

PROMPT_VERSION = "relative_depth_gemini_shared_v2"

FIXED_DEPTH_POLARITY = "near_white"

def get_depth_semantics(depth_polarity: str = FIXED_DEPTH_POLARITY) -> Dict[str, Any]:
    normalized = str(depth_polarity or "").strip().lower()
    if normalized != FIXED_DEPTH_POLARITY:
        raise ValueError(
            f"This script only supports near_white depth polarity, got: {depth_polarity}"
        )

    return {
        "save_mode": FIXED_DEPTH_POLARITY,
        "prompt_mode": FIXED_DEPTH_POLARITY,
        "near_tone": "white",
        "far_tone": "black",
        "invert_after_save": False,
    }

def build_relative_depth_prompt(
    depth_polarity: str,
) -> str:
    semantics = get_depth_semantics(depth_polarity)
    near_tone = semantics["near_tone"]
    far_tone = semantics["far_tone"]

    common_body = (
    "This is a relative depth prediction task, not metric depth regression. "
    "Predict camera-centric scene depth up to an unknown affine transform. "
    "The output must be a single-channel grayscale depth-buffer-like map, pixel-aligned with the target RGB image. "
    "Do not crop, zoom, shift, recompose, add, remove, or move any scene content. "

    "Brighter values indicate surfaces closer to the camera; darker values indicate surfaces farther from the camera. "
    "Use a white near tone for close surfaces and a black far tone for distant surfaces. "

    "Infer the 3D layout first, then render only the depth map. "
    "Use geometric cues such as perspective, occlusion ordering, contact and support relations, object placement, surface orientation, room layout, and relative size. "
    "Occluding surfaces must be closer than the surfaces they occlude. "

    "Completely ignore photometric appearance when assigning depth: brightness, darkness, shadows, highlights, reflections, exposure, albedo, color, texture, material, gloss, image noise, and local contrast must not change the predicted depth. "
    "A bright patch on a floor, wall, door, cabinet, or tabletop remains at that surface's geometric depth. "
    "A dark shadow or dark material does not make a region farther away. "
    "Never make the result look like ambient occlusion, relighting, shading, or a grayscale rendering. "

    "Large planar or near-planar surfaces such as floors, walls, ceilings, doors, cabinets, and tabletops should be smooth and perspective-consistent. "
    "Within each visible surface face, depth should change only according to geometry and perspective, not according to illumination gradients or material texture. "
    "Reserve strong contrast mainly for true depth discontinuities: silhouettes, occlusion boundaries, contact edges, and object-to-background transitions. "

    "For openings, windows, mirrors, glass, and overexposed regions, estimate depth from the actual visible geometry. "
    "If a finite surface is visible, use that surface's depth. "
    "If the region is an opening with no visible finite surface, assign it far-background depth. "

    "Thin structures such as chair legs, table legs, lamp cords, plant stands, and object edges should remain coherent and aligned, but should not introduce texture-like grayscale noise. "
    "Use the full image-level depth range consistently: nearest valid visible surfaces should be close to white, farthest valid visible regions should be close to black, and intermediate surfaces should preserve correct ordinal ordering. "

    "The final image must be a clean training-label-style depth map, not a visually pleasing grayscale image. "
    "Do not stylize. Do not add text, labels, color, collage, borders, or overlays. "
    "Output exactly one grayscale depth map only."
    )

    prompt = (
        "You are a senior monocular depth estimation and 3D scene geometry expert. "
        "You are given exactly one input image: the target RGB image. "
        "Generate only a dense single-image relative depth map for that RGB image. "
        + common_body
    )


    return prompt
