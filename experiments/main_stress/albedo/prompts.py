"""All main/stress albedo models use the ablation A0 instruction."""
from __future__ import annotations

from experiments.ablation.albedo.prompts import FULL_PROMPT_TEXT as ALBEDO_FULL_PROMPT

PROMPT_VERSION = "albedo_shared_a0_v1"
FULL_PROMPT_MODELS = frozenset({"gemini", "doubao", "gpt_image_15", "gpt_image_2", "qwen2", "wan"})


def make_albedo_prompt(model: str) -> str:
    """Return the exact A0 instruction for a supported model family."""
    if model in FULL_PROMPT_MODELS:
        return ALBEDO_FULL_PROMPT
    raise ValueError(f"Unsupported albedo prompt model: {model}")
