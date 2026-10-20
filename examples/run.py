#!/usr/bin/env python3
"""Run the two bundled samples with existing Azure generation/evaluation code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TARGETS = ("depth", "normal", "albedo", "roughness", "metallic")


def load_samples(asset_root: Path = HERE / "assets") -> dict:
    samples = json.loads((HERE / "samples.json").read_text())["samples"]
    result = {}
    for sample in samples:
        paths = {}
        for name, relative in sample["files"].items():
            path = asset_root / Path(relative).relative_to("assets")
            if not path.is_file():
                raise FileNotFoundError(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"][name]:
                raise ValueError(f"Sample asset differs from its recorded hash: {path}")
            paths[name] = path
        for target in sample["targets"]:
            result[target] = {**paths, "asset_root": asset_root}
    return result


def generation_dir(target: str, output: Path, files: dict) -> Path:
    base = output / "predictions" / target
    if target == "metallic":
        # Retain Image/camera layout for the existing metallic evaluator.
        return base / files["rgb"].parent.relative_to(files["asset_root"] / "pbr_rooms")
    return base


def prediction_path(target: str, output: Path, files: dict) -> Path:
    subdir = "relative_depth" if target == "depth" else target
    suffix = "relative_depth" if target == "depth" else target
    return generation_dir(target, output, files) / subdir / f"{files['rgb'].stem}_{suffix}.png"


def generation_command(target: str, output: Path, files: dict, deployment: str) -> list[str]:
    folder = ROOT / "experiments" / "main_stress" / target / "GPT-Image-1.5"
    scripts = {
        "depth": "depth_gpt_image_15.py",
        "normal": "normal_gpt_image_15.py",
        "albedo": "albedo_gpt_image_15.py",
        "roughness": "roughness_gpt_image_15.py",
        "metallic": "metallic_gpt_image_15.py",
    }
    cmd = [sys.executable, str(folder / scripts[target]),
           "--input_dir", str(files["rgb"].parent),
           "--output_dir", str(generation_dir(target, output, files)), "--max_generate", "1"]
    if target in {"normal", "albedo", "roughness"}:
        cmd += ["--filename_suffix", files["rgb"].name]
    if target == "normal":
        references = ROOT / "experiments" / "main_stress" / "normal" / "examples"
        for name in ("image.png", "normal.png"):
            if not (references / name).is_file():
                raise FileNotFoundError(references / name)
        cmd += ["--normal_deployment", deployment,
                "--example_rgb", str(references / "image.png"),
                "--example_normal", str(references / "normal.png"),
                "--gt_root", str(files["asset_root"])]
    elif target == "albedo":
        cmd += ["--albedo_deployment", deployment, "--gt_root", str(files["asset_root"])]
    else:
        model_flag = "--image_deployment" if target == "metallic" else "--image_model"
        cmd += [model_flag, deployment, "--input_mode", "rgb_only", "--skip_existing"]
    return cmd


def evaluation_command(target: str, output: Path, files: dict, lpips: bool) -> list[str]:
    script = f"{target}_eval.py"
    cmd = [sys.executable, str(ROOT / "tools" / "evaluation" / script),
           "--output_dir", str(output / "metrics" / target)]
    pred = prediction_path(target, output, files)
    if target == "depth":
        cmd += ["--gt_root", str(files["depth"].parent),
                "--mask_root", str(files["mask"].parent),
                "--pred_root", str(pred.parent), "--require_mask"]
    elif target == "metallic":
        cmd += ["--gt_root", str(files["asset_root"] / "pbr_rooms"),
                "--pred_root", str(output / "predictions" / target), "--mask_fallback", "none"]
    else:
        cmd += ["--gt_path", str(files[target]), "--pred_path", str(pred),
                "--mask_path", str(files["mask"])]
        if target == "roughness":
            cmd += ["--region_path", str(files["segmentation"])]
        if target in {"albedo", "roughness"}:
            cmd += ["--compute_lpips" if lpips else "--no_compute_lpips"]
    return cmd


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from common.utils.env import load_project_env
    load_project_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("all", *TARGETS), default="all")
    parser.add_argument("--stage", choices=("generate", "evaluate", "all"), default="all")
    parser.add_argument("--deployment", default="gpt-image-1.5", help="Azure image deployment name.")
    parser.add_argument("--asset-root", type=Path,
                        default=os.environ.get("EXAMPLE_ASSET_ROOT") or HERE / "assets")
    parser.add_argument("--output-root", type=Path,
                        default=os.environ.get("EXAMPLE_OUTPUT_ROOT") or ROOT / "outputs" / "examples" / "gpt-image-1.5")
    parser.add_argument("--lpips", action="store_true", help="Enable LPIPS; requires its runtime and model weights.")
    parser.add_argument("--dry-run", action="store_true", help="Check assets and print commands without executing them.")
    args = parser.parse_args()
    def repository_path(path: Path) -> Path:
        path = path.expanduser()
        return (path if path.is_absolute() else ROOT / path).resolve()

    asset_root = repository_path(args.asset_root)
    samples = load_samples(asset_root)
    output = repository_path(args.output_root)
    if output == HERE or HERE in output.parents or output == asset_root or asset_root in output.parents:
        parser.error("Choose an output root outside examples/ to preserve the bundled assets.")
    targets = TARGETS if args.target == "all" else (args.target,)
    for target in targets:
        files = samples[target]
        stages = ("generate", "evaluate") if args.stage == "all" else (args.stage,)
        for stage in stages:
            cmd = (generation_command(target, output, files, args.deployment) if stage == "generate"
                   else evaluation_command(target, output, files, args.lpips))
            print(f"[{target}/{stage}] {shlex.join(cmd)}", flush=True)
            if args.dry_run:
                continue
            if stage == "evaluate" and not prediction_path(target, output, files).is_file():
                raise FileNotFoundError(f"Expected prediction is missing: {prediction_path(target, output, files)}")
            subprocess.run(cmd, cwd=ROOT, check=True)
            if stage == "generate" and not prediction_path(target, output, files).is_file():
                raise RuntimeError(f"Generation finished without the expected {target} prediction; inspect its logs.")


if __name__ == "__main__":
    main()
