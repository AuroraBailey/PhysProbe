"""Shared normal-map prompt; inputs are query RGB, reference RGB, reference normal."""
from __future__ import annotations

SUPPORTED_MODELS = ("doubao", "gpt_image_15")
MODEL_CONFIGS = {model: {} for model in SUPPORTED_MODELS}

NORMAL_PROMPT = (
    "You are given three images in order: "
    "(1) the query indoor RGB image to convert, "
    "(2) an example indoor RGB image, "
    "(3) the target normal map for that example. "

    "Generate the normal map for image (1). "
    "Image (3) is the only convention reference and has the highest priority for normal-map encoding, "
    "but use it only for palette convention, direction-family convention, surface-role convention, deeper-color convention, boundary style, saturation range, and large-region consistency. "
    "Image (2) is only a geometry-to-convention analogy example. "
    "Neither image (2) nor image (3) may change the composition, crop, camera pose, or visible object inventory of image (1). "

    "Color-meaning guide: "
    "Use camera-space normal colors: Normals pointing to the camera-right (+X) should appear red / pink. Normals pointing to the camera-left (-X) should appear cyan / teal. Normals pointing upward (+Y) should appear green. Normals pointing downward (-Y) should appear magenta / purple. Normals pointing toward the camera (+Z) should appear blue to blue-purple. Normals pointing away from the camera (-Z) should appear yellow / olive. "
    "A neutral front-facing normal is often close to a blue-dominant color family, while opposite tilts move the red or green channels in opposite directions. "
    "However, the exact sign and palette are convention-dependent, and the green channel may be flipped in different pipelines. "
    "Therefore do not impose a universal textbook palette. "

    "Critical rules: "
    "color must encode surface orientation only, never material color, never albedo, never lighting tint. "
    "Do not preserve wood color, fabric color, wall color, floor color, or object appearance from the query RGB image. "
    "Do not preserve wood grain, fabric weave, printed patterns, shading, highlights, reflections, image noise, or any illumination residue. "

    "Treat image (1) only as a geometry-layout carrier. "
    "Preserve its exact pixel-space layout, exact current viewpoint, visible object inventory, crop extents, occlusion order, silhouettes, true boundaries, and visible geometry. "
    "Do not recrop, reframe, recompose, regularize, clean up, or canonicalize the query view. "
    "Do not simplify the scene to only broad shape. "
    "Overwrite photographic appearance with a clean synthetic normal-map encoding. "

    "Reference isolation rule: "
    "image (3) controls only the normal-map convention, palette family, sign convention, and orientation-family mapping. "
    "image (2) helps match analogous structural roles. "
    "Neither reference image controls scene composition, camera pose, object inventory, cropping, visibility, or which objects are present in image (1). "

    "Pixel-space fidelity rule: "
    "preserve the 2D image-space position, scale, and extent of all visible objects and boundaries. "
    "Do not zoom, shift, recenter, enlarge, shrink, or reposition visible objects. "
    "Keep the same crop, framing, and perspective footprint as the query image. "

    "Rigid layout preservation rule:"
    "Do not move, shift, recenter, enlarge, shrink, rotate, or re-place any visible object."
    "Every visible object must stay at the same 2D image-space position, with the same footprint, crop extent, and support contact, as in image (1)."
    "Do not recompose the scene."
    "Do not slide hanging objects, tabletop objects, wall-mounted objects, or foreground objects to a nearby easier position."
    "If uncertain, keep the original object position rather than improving composition."

    "Instance preservation rule: "
    "preserve all clearly visible object instances in image (1). "
    "Do not delete, omit, merge away, dissolve, or collapse visible furniture, foreground objects, tabletop objects, decor objects, wall-mounted objects, ceiling-mounted objects, or partially visible objects. "
    "If an object is clearly visible in the query image, it must remain present in the output normal map at the same location with a recognizable silhouette and approximate visible faces. "

    "Inventory and crop rule: "
    "preserve the visible object inventory, crop extents, border truncation behavior, and occlusion order of the query image. "
    "Do not clean up, declutter, beautify, substitute, or reduce the number of visible objects. "
    "Objects touching the image boundary must remain truncated in the same way as in the query image. "

    "Small-object rule: "
    "for small objects such as bottles, books, cups, phones, remotes, plant pots, trays, bowls, candles, and decor items, preserve the object instance itself, its count, location, silhouette, support contact, and main visible faces. "
    "Simplify tiny appearance details and texture, but do not delete the object, substitute it with a different object, merge multiple objects into one, or merge it into the supporting surface. "

    "Thin-structure retention rule: "
    "preserve thin but visible structures such as branches, stems, lamp arms, chandelier arms, narrow object edges, chair legs, table legs, and frame borders when they are clearly visible in image (1). "
    "Represent them with simplified but still visible geometry. "

    "Orientation separation rule: "
    "distinguish clearly between front-facing vertical faces, upward-facing horizontal faces, downward-facing undersides, and side-facing faces. "
    "Adjacent orthogonal faces should use distinct reference families when the reference normal map separates them. "
    "Wall, floor, ceiling, top surfaces, underside surfaces, and side-facing surfaces must remain separated by orientation, not by photographic shading. "

    "Global position-gradient veto rule: "
    "do not create a top-to-bottom, bottom-to-top, center-to-edge, or depth-like color drift across the whole image. "
    "Image position, distance, perspective depth, and scene brightness must not create a global purple-to-cyan or cyan-to-purple gradient. "

    "Lighting-removal rule: "
    "ignore illumination completely. "
    "Do not encode lamp glow, ceiling-light hotspots, window brightening, cast shadows, soft shadow gradients, interreflection, ambient occlusion-like darkening, exposure falloff, bloom, glare, or colored light spill into the normal map. "
    "Only true geometric orientation and the reference convention may determine output color. "

    "Planar family consistency rule: "
    "for one continuous planar region with nearly constant orientation, use one stable reference family across the whole region. "
    "Do not introduce large blue-purple or cyan-magenta drift across the same wall, same cabinet face, same desktop, same drawer front, same shelf face, same table top, or same curtain panel when the orientation is nearly constant. "

    "Dataset-style color restraint rule: "
    "prefer stable dataset-like normal colors over dramatic purple-cyan room-wide styling. "
    "Do not let ceiling, wall, and floor separation become a stylized global color design. "

    "Very strong texture suppression rule: "
    "remove fine-scale and mid-scale material texture. "
    "If a local variation is not clearly a true shape change, treat it as texture and suppress it. "
    "Large door faces, table tops, sofa backs, sofa seats, sofa arms, walls, floors, ceilings, and other broad surfaces should be smooth, low-variance, and nearly texture-free. "

    "Geometry rule: "
    "preserve the query image layout, exact current viewpoint, object inventory, object boundaries, major silhouettes, broad curvature, curtain folds, furniture macro-shape, panel recesses, frame thickness, contact boundaries, and true shape discontinuities. "
    "Use narrow transitions only at true geometric boundaries. "
    "Do not invent fake bevels, fake grooves, fake seams, texture relief, or artistic contour lines. "

    "Priority order: "
    "1) preserve the exact current query viewpoint, visible object inventory, crop extents, and layout, "
    "2) preserve all clearly visible object instances, their count, and occlusion order, "
    "3) preserve pixel-space positions and border truncation behavior, "
    "4) infer the meaning of the reference color families from image (3), "
    "5) keep wall, floor, ceiling, top surfaces, underside surfaces, and side-facing surfaces clearly separated when their orientations differ, "
    "6) keep one stable family for one nearly planar region, "
    "7) remove all lighting and illumination effects, "
    "8) suppress texture aggressively, "
    "9) never copy material color or material texture from image (1), "
    "10) if uncertain, prefer preserving a coarse visible object over deleting it. "

    "The output must look like a clean dataset normal map, not a purple relit image and not a stylized neon cyan-magenta rendering. "
    "Output only the final normal map for image (1)."
)

PROMPTS = {model: NORMAL_PROMPT for model in SUPPORTED_MODELS}


def make_one_shot_prompt(*, model: str) -> str:
    """Return the complete shared instruction for query-first image inputs."""
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported normal prompt model: {model}")
    return NORMAL_PROMPT
