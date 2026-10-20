"""All main/stress roughness models use the ablation A0 instruction."""
from __future__ import annotations

PROMPT_VERSION = "roughness_shared_rgb_v2"
SUPPORTED_MODELS = ("doubao", "gpt_image_15", "gpt_image_2", "qwen2", "gemini")
MODEL_CONFIGS = {model: {"PROMPT_VERSION": PROMPT_VERSION} for model in SUPPORTED_MODELS}

# Keep main/stress text identical to the diagnostic reference condition.
from experiments.ablation.roughness.prompts import build_prompt as _build_ablation_prompt

ROUGHNESS_PROMPT = _build_ablation_prompt("a0")

PROMPTS = {model: ROUGHNESS_PROMPT for model in SUPPORTED_MODELS}


def build_roughness_prompt(input_mode: str = "rgb_only", *, model: str) -> str:
    """Return the exact A0 instruction for each supported model."""
    if input_mode != "rgb_only":
        raise ValueError("Main/stress generation supports only rgb_only")
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported roughness prompt model: {model}")
    return ROUGHNESS_PROMPT
