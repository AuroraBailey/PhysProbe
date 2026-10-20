"""RGB-only metallic instructions with a shared Qwen/Wan variant."""
from __future__ import annotations

PROMPT_VERSION = "metallic_shared_rgb_v1"
PROMPT_PRESET_CHOICES = ("v3_visualprior_noboundary",)
SUPPORTED_MODELS = ("doubao", "gpt_image_15", "gpt_image_2", "qwen2", "wan", "gemini")
MODEL_CONFIGS = {
    model: {
        "PROMPT_FAMILY": PROMPT_VERSION,
        "PROMPT_VERSION": PROMPT_VERSION,
        "PROMPT_PRESET": PROMPT_PRESET_CHOICES[0],
        "PROMPT_PRESET_CHOICES": PROMPT_PRESET_CHOICES,
    }
    for model in SUPPORTED_MODELS
}
MODEL_CONFIGS["qwen2"]["DEFAULT_IMAGE_MODEL"] = "qwen-image-2.0"
MODEL_CONFIGS["wan"]["DEFAULT_IMAGE_MODEL"] = "wan2.7-image"

TEXTURE_GUIDANCE = (
    " As a weak visual prior, bright reflections, mirror-like appearance, or smooth highlights are not sufficient evidence of metal by themselves. "
    " Do not display object boundaries, panel boundaries, contour lines, reflective edges, or decorative lines as gray metallic structure. "
    " Do not use metallic values to preserve boundaries, silhouettes, rims, borders, or edge visibility on likely non-metal regions. "
    " Bright regions caused mainly by direct lighting, light pools, glossy floor reflections, bloom, or illumination falloff are not metal evidence by themselves. "
    " Spatial alignment means pixel correspondence only; do not encode object shape using edges, seams, contour lines, or decorative outlines. "
    " Thin borders, seams, frame lines, grout lines, and decorative contours should remain black unless they belong to a clearly filled exposed-metal surface. "
    " If metal evidence is local or uncertain, keep the rest of the object black rather than extending gray/white into the surrounding housing, frame, or panel. "
    " Outside clearly exposed metal regions, the output should be near-uniform black with no line-like traces or residual edge responses. "
    " When uncertain, do not preserve scene structure with low gray; prefer conservative near-black predictions for likely non-metal regions. "
)

METALLIC_PROMPT = (
    "You are a senior PBR material analysis expert. "
    "You are given exactly one input image: the target RGB image. "
    "Generate only the target metallic map for that RGB image. "
    "The output must be a single-channel grayscale metallic map spatially aligned with the input RGB content. "
    "Preserve the exact scene layout and object shapes from the RGB image. "
    "Do not add, remove, duplicate, move, deform, or hallucinate objects or structures. "
    "Black means dielectric or non-metal. White means clearly exposed metal surface with strong visible metal evidence. "
    "This target convention is effectively binary. Most indoor pixels should be pure black. "
    "Large non-metal regions must remain black with no contour tracing, no dark-gray embossing, and no structural outline rendering. "
    "Do not preserve object boundaries by drawing gray edges around non-metal objects. "
    "Do not output a relit grayscale scene, an edge map, or a reflectance sketch. "
    "Use white only when the visible surface itself provides clear evidence of exposed metal material, such as unmistakable bare metal hardware or an obviously metallic housing surface. "
    "Do not fill an entire object white unless the visible surface is unmistakably an exposed metal body or housing. If metal evidence is local or uncertain, keep the rest black rather than filling the whole object. "
    "Do not treat illumination-driven bright patches, light pools, floor highlights, glossy reflections, or lamp glow as metallic unless the visible surface itself clearly reads as exposed metal. "
    "Spatial alignment means pixel correspondence only; do not encode object shape using edges, seams, contour lines, or decorative outlines. "
    "Thin borders, seams, frame lines, grout lines, and decorative contours should remain black unless they belong to a clearly filled exposed-metal surface. "
    "Mirror frames, window frames, door frames, trims, moldings, cabinet borders, decorative borders, mirror edges, and glossy border strips should remain black unless exposed bare metal evidence is unmistakable. "
    "Large smooth panels, painted appliance sides, coated boards, cabinet faces, drawer fronts, glossy doors, and other broad plain surfaces should remain black unless explicit exposed-metal cues are clearly visible. "
    "Do not convert seams, grooves, panel boundaries, wood grain, tile joints, decorative edges, contour lines, or texture contrast on non-metal materials into gray metallic structure. "
    "Do not use low gray to preserve scene structure, object identity, silhouettes, rims, borders, or reflective outlines. "
    "Glass and mirror remain black. Ceramic, painted walls, stone, wood, plastic, laminate, painted board, coated furniture, and glossy painted surfaces remain black. "
    "For lamps, do not assume the bright diffuser, shade, glowing cover, rim highlight, or reflective ring is metallic; only clearly exposed opaque metal parts may be white. "
    "Small hardware such as knobs, hinges, handles, and brackets may be white only when exposed metal is visually unmistakable; otherwise keep them black with no gray outlining. "
    "If no unmistakable exposed metal surface is visible, output an almost-all-black map. "
    "Outside clearly exposed metal regions, enforce near-uniform black and suppress any residual line-like traces. "
    "Isolated line-like or hollow outline responses should be treated as non-metal unless they belong to a small filled exposed-metal patch with visible area. "
    "If evidence is ambiguous, choose black rather than gray. Mid-gray should be extremely rare. "
    "Do not infer metal from brightness, reflections, shadows, specular highlights, mirror-like appearance, or smoothness alone. "
    f"{TEXTURE_GUIDANCE}"
    "A good output should look almost empty except for clearly metallic parts. "
    "Prefer a sparse, semantically correct material-category map over a smooth grayscale rendering. "
    "When evidence is ambiguous, favor conservative non-metal predictions rather than inventing large metallic regions. "
    "Output exactly one grayscale metallic map only."
)

QWEN_WAN_METALLIC_PROMPT = (
    "Input: one target RGB image. Generate exactly one aligned grayscale metallic map only. "
    "Interpret this as sparse metallic labeling rather than grayscale scene reconstruction. "
    "Black means non-metal or dielectric. White means clearly exposed metallic material. "
    "Gray should be used rarely and only for genuinely ambiguous or partially metallic visible regions. "
    "Most indoor pixels should be pure black. Only relatively small exposed metal regions should become gray or white. "
    "The metallic map should be sparse; large contiguous gray/white regions are usually wrong. "
    "Output only the metallic map, not a grayscale photo, not a desaturated image, not a shaded scene, not a relit result, and not an edge or contour map. "
    "Do not copy brightness, shadows, reflections, highlights, ambient occlusion, texture contrast, or scene shading into the metallic map. "
    "Bright or white appearance alone is not evidence of metal. Smooth finish, glossy reflections, mirror-like highlights, ceramic-like highlights, porcelain, glazed ceramic, enamel-like surfaces, glossy paint, and smooth white fixtures should stay black unless exposed metal is clearly visible. "
    "Judge only the visible surface material, not the hidden substrate or what the object may be made of underneath paint or coating. "
    "Do not infer metal from object category alone: window frames, cabinets, doors, furniture, fixtures, and trim should default to near-black unless clearly visible bare metal is exposed. "
    "Treat transparent or translucent glass as non-metal by default; reflections or highlights on glass are not metal evidence. "
    "Do not brighten entire window panes, glass doors, outdoor scenery seen through glass, or room content visible behind glass; those broad glass-covered regions should stay near-black unless the visible surface itself is clearly exposed metal. "
    "As a conservative prior, treat glass as non-metal unless clearly exposed metallic surface is visible. "
    "Appliance fronts should stay near-black unless the visible face clearly looks like exposed bare stainless steel or brushed metal; only then is a larger contiguous metallic region allowed. "
    "Do not use gray metallic values to preserve boundaries, silhouettes, contour lines, reflective edges, decorative lines, or overall object structure in likely non-metal regions. "
    "Do not turn large smooth surfaces gray or white without clear visible evidence of exposed metal. "
    "Do not fill walls, ceilings, floors, cabinet doors, or countertops with gray/white just to preserve shape; keep them near-black unless clear exposed bare metal is visible. "
    "Non-metal surfaces should stay black unless exposed metal is clearly visible. "
    "When uncertain, choose conservative near-black or black."
)

MODEL_CONFIGS["qwen2"]["PROMPT_VERSION"] = "metallic_qwen_wan_rgb_v1"
MODEL_CONFIGS["wan"]["PROMPT_VERSION"] = "metallic_qwen_wan_rgb_v1"

PROMPTS = {
    model: QWEN_WAN_METALLIC_PROMPT if model in {"qwen2", "wan"} else METALLIC_PROMPT
    for model in SUPPORTED_MODELS
}


def build_metallic_prompt(
    input_mode: str = "rgb_only", *, model: str,
    prompt_preset: str = "v3_visualprior_noboundary", image_model: str | None = None,
) -> str:
    """Return the fixed instruction for the selected model family."""
    if input_mode != "rgb_only":
        raise ValueError("Main/stress generation supports only rgb_only")
    if prompt_preset not in PROMPT_PRESET_CHOICES:
        raise ValueError(f"Unsupported prompt preset: {prompt_preset}")
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported metallic prompt model: {model}")
    return PROMPTS[model]
