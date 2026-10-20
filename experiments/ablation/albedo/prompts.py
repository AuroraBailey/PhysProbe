"""Albedo diagnostic prompts; setting IDs follow the paper."""
from __future__ import annotations
FULL_PROMPT_TEXT = (
    "Convert the input image into a clean intrinsic albedo image. "
    "Preserve the exact scene layout, visible objects, object boundaries, and material regions. "
    "Remove illumination effects only: cast shadows, attached shading, highlights, reflections, "
    "interreflections, exposure variation, ambient occlusion, and illumination color cast. "
    "Keep the intrinsic reflectance or base color of each visible surface. "
    "Preserve low-frequency and mid-frequency material texture that belongs to the surface itself, "
    "such as wood color variation, subtle fabric color variation, printed material color, and "
    "material-region color boundaries, but do not preserve lighting-induced brightness gradients. "
    "Do not simplify the image into flat poster-like color blocks unless the original surface is truly uniform. "
    "Do not add new objects. Do not remove existing objects. Do not stylize. "
    "Output a clean, illumination-free, texture-preserving albedo map."
)

WEAKENED_PROMPT_TEXT = (
    "Convert the input indoor RGB image into an albedo image. "
    "Keep the scene geometry, objects, boundaries, and material colors. "
    "Reduce shadows, highlights, reflections, exposure changes, and other lighting effects. "
    "Preserve visible surface texture and do not add or remove objects."
)

MINIMAL_PROMPT_TEXT = "Generate an albedo map from the input image."

MINIMAL_FIXED_EXEMPLAR_PROMPT_TEXT = (
    "Input: (1) example RGB, (2) example intrinsic albedo, (3) target RGB. "
    "Task: generate the intrinsic albedo map for the target RGB image. "
    "Use the example RGB/albedo pair as a reference sample for translating appearance into intrinsic albedo. "
    "Preserve the target scene structure, visible objects, and object boundaries. "
    "Do not add or remove objects. Output one albedo image only."
)

from typing import Any, Dict

def build_single_image_analysis_prompt(image_name: str) -> str:
    return f"""
你将看到一张室内场景的原始 RGB 图像。你的任务不是生成图片，而是为后续该图像的 albedo 生成提供单图 textual analysis。

要求：
1. 仅根据这一张输入图做判断，不要假设其他视角信息。
2. 只总结最稳定、最确定的信息，不要长篇发挥。
3. 只关注材质固有色与哪些亮暗属于光照。
4. 强调哪些区域的亮暗来自阴影、高光、曝光或反射，而不属于材质本色。
5. 强调哪些表面存在应被保留的固有纹理，例如木材颜色变化、布料印花、墙面装饰颜色。
6. 不要编造不可见区域或被遮挡区域的细节。
7. 输出内容只作为该图像的辅助 textual conditioning，不是 ground truth。
8. 只输出 JSON，不要输出解释文字。

输入图像文件名如下：
- {image_name}

请严格输出下面这个 JSON 结构：
{{
  "scene_summary": "一句话概括场景",
  "albedo_material_notes": ["2~6 条最重要的材质/固有色说明"],
  "albedo_lighting_notes": ["2~6 条最重要的光照/阴影/高光说明"],
  "albedo_prompt_suffix": "一句话补充约束，强调保留固有纹理而去除光照",
  "albedo_hint": "该图像 albedo 生成的简短补充说明"
}}
""".strip()

def make_single_image_albedo_prompt(single_analysis: Dict[str, Any]) -> str:
    material_notes = "; ".join(single_analysis.get("albedo_material_notes", []))
    lighting_notes = "; ".join(single_analysis.get("albedo_lighting_notes", []))
    suffix = single_analysis.get("albedo_prompt_suffix", "")
    albedo_hint = single_analysis.get("albedo_hint", "")
    base = FULL_PROMPT_TEXT
    return (
        f"{base}\n\n"
        f"Scene summary: {single_analysis.get('scene_summary', '') or 'N/A'}\n"
        f"Albedo material notes: {material_notes or 'N/A'}\n"
        f"Albedo lighting notes: {lighting_notes or 'N/A'}\n"
        f"Image-specific hint: {albedo_hint or 'N/A'}\n"
        f"Additional constraints: {suffix or 'N/A'}"
    )
