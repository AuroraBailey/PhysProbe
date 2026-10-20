"""Resolve same-view benchmark masks; unmasked scoring requires explicit opt-out."""
from pathlib import Path

def resolve_required_mask(gt_path, mask_path=None, *, disabled=False):
    if disabled:
        return None
    if mask_path is not None:
        if not Path(mask_path).is_file():
            raise FileNotFoundError(f"Required mask missing: {mask_path}")
        return Path(mask_path)
    gt = Path(gt_path)
    stems = [gt.stem]
    for suffix in ("_normal", "_albedo", "_roughness", "_im", "_depth", "_imnormal_5", "_imalbedo_5", "_imroughness_5", "_imdepth_5"):
        if gt.stem.endswith(suffix):
            stems.append(gt.stem[:-len(suffix)])
    candidates = [gt.with_name(stem + suffix) for stem in stems for suffix in ("_mask.png", "_immask_5.png")]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Required same-view mask missing for {gt}; supply --mask_path/--mask_dir or explicitly use --disable_mask")
