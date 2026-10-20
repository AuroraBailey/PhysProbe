"""Paper normal ablations; all reference-conditioned inputs are query-first."""
from __future__ import annotations
from experiments.main_stress.normal.prompts import NORMAL_PROMPT

INPUT_ORDER = (
    "You are given three images in order: "
    "(1) the query indoor RGB image to convert, "
    "(2) an example indoor RGB image, "
    "(3) the target normal map for that example. "
)
FULL_RGB_PLUS_EXAMPLE_PROMPT = NORMAL_PROMPT

# Keep the full geometry/appearance restrictions when removing exemplar access.
# Reference-only instructions are excluded because A1 supplies one image.
_color = NORMAL_PROMPT.split("Color-meaning guide: ", 1)[1].split("Critical rules: ", 1)[0]
_geometry = NORMAL_PROMPT.split("Critical rules: ", 1)[1].split("Reference isolation rule: ", 1)[0]
_layout = NORMAL_PROMPT.split("Pixel-space fidelity rule: ", 1)[1].split("Priority order: ", 1)[0]
_layout = _layout.replace(
    "Adjacent orthogonal faces should use distinct reference families when the reference normal map separates them. ",
    "Adjacent orthogonal faces should use distinct camera-space orientation colors. "
).replace("the reference convention", "the camera-space encoding convention").replace("reference family", "orientation color family")
FULL_RGB_ONLY_PROMPT = (
    "You are given exactly one input image: the query indoor RGB image. "
    "Generate its camera-space normal map. Color-meaning guide: " + _color
    + "Critical rules: " + _geometry + "Pixel-space fidelity rule: " + _layout
    + "Preserve the exact query viewpoint, visible object inventory, pixel positions, crop and occlusion order. "
    "Output only the final normal map for the query RGB image."
)
MINIMAL_RGB_PLUS_EXAMPLE_PROMPT = INPUT_ORDER + "Generate the normal map for image (1). Output one normal map image only."
MINIMAL_RGB_ONLY_PROMPT = "Generate the normal map of the input RGB image. Output one normal map image only."
