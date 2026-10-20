#!/usr/bin/env python3
"""Offline Python port of the original PowerShell three-model audit merger."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

MODEL_ORDER = ("doubao", "glm", "qwen")
MODEL_DIRS = {model: f"annotation_{model}" for model in MODEL_ORDER}
VOTE_FIELDS = (
    "layout_plausibility", "functional_coherence", "object_arrangement_coherence",
    "lighting_naturalness", "brightness_level", "illumination_level",
    "dynamic_range_level", "highlight_strength", "shadow_strength",
    "dark_region_ratio_level", "overall_plausibility",
)


class SeededRandom:
    """System.Random(int seed) compatible sequence, as used by the PS1 script.

    This is the legacy subtractive algorithm retained by seeded System.Random;
    Python's random.Random would select different records for the same seed.
    """
    MAX = 2147483647

    def __init__(self, seed: int) -> None:
        if not -(2**31) <= seed < 2**31:
            raise ValueError("seed must be a signed 32-bit integer")
        subtraction = self.MAX if seed == -(2**31) else abs(seed)
        mj = 161803398 - subtraction
        self.values = [0] * 56
        self.values[55] = mj
        mk = 1
        for i in range(1, 55):
            index = (21 * i) % 55
            self.values[index] = mk
            mk = mj - mk
            if mk < 0:
                mk += self.MAX
            mj = self.values[index]
        for _ in range(4):
            for i in range(1, 56):
                self.values[i] -= self.values[1 + (i + 30) % 55]
                if self.values[i] < 0:
                    self.values[i] += self.MAX
        self.index = 0
        self.partner = 21

    def next(self, count: int) -> int:
        if count <= 0:
            raise ValueError("count must be positive")
        self.index = self.index % 55 + 1
        self.partner = self.partner % 55 + 1
        sample = self.values[self.index] - self.values[self.partner]
        if sample == self.MAX:
            sample -= 1
        if sample < 0:
            sample += self.MAX
        self.values[self.index] = sample
        return int(sample * (1.0 / self.MAX) * count)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def parsed_value(record: dict[str, Any], key: str) -> Any:
    parsed = record.get("parsed")
    return parsed.get(key) if isinstance(parsed, dict) else None


def vote_key(value: Any) -> str:
    # PowerShell's default hashtables compare string keys case-insensitively.
    if value is None:
        return "__null__"
    if isinstance(value, bool):
        return f"bool:{value}"
    if isinstance(value, (int, float)):
        return f"num:{value:g}"
    if isinstance(value, list):
        return "arr:" + json.dumps(value, ensure_ascii=False, separators=(",", ":")).casefold()
    return "str:" + str(value).casefold()


def resolve_vote(values: list[Any]) -> Any:
    counts: dict[str, int] = {}
    for value in values:
        key = vote_key(value)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= 2:
            return next(item for item in values if vote_key(item) == key)
    return list(values)


def resolve_confidence(values: list[Any]) -> float | None:
    numbers = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError):
            continue
        # Non-finite values cannot be written as interoperable JSON numbers.
        if math.isfinite(number):
            numbers.append(number)
    if not numbers:
        return None
    winner = resolve_vote(numbers)
    return sum(numbers) / len(numbers) if isinstance(winner, list) else winner


def merge_records(records: list[dict[str, Any]], rng: SeededRandom) -> dict[str, Any]:
    if len(records) != 3:
        raise ValueError("exactly three records are required, in doubao/glm/qwen order")
    images = [as_list(record.get("images")) for record in records]
    largest = max(map(len, images))
    candidates = [i for i, names in enumerate(images) if len(names) == largest]
    chosen = records[candidates[rng.next(len(candidates))]]
    merged = {
        "scene": next((record["scene"] for record in records
                       if isinstance(record.get("scene"), str) and record["scene"]), ""),
        "images": images[candidates[0]],
        "raw": [record.get("raw") for record in records],
        "scene_category": resolve_vote([parsed_value(record, "scene_category") for record in records]),
    }
    for field in VOTE_FIELDS:
        merged[field] = resolve_vote([parsed_value(record, field) for record in records])
    merged.update(
        lighting_challenge_type=[parsed_value(record, "lighting_challenge_type") for record in records],
        confidence=resolve_confidence([parsed_value(record, "confidence") for record in records]),
        issues=[as_list(parsed_value(record, "issues")) for record in records],
        reason_short=[parsed_value(record, "reason_short") for record in records],
        parse_error=[record.get("parse_error") for record in records],
        computed=chosen.get("computed") if chosen.get("computed") is not None else {},
    )
    return merged


def merge_directories(root: Path, output: Path, seed: int = 42,
                      model_dirs: dict[str, str] | None = None) -> dict[str, Any]:
    root = root.resolve()
    output = (root / output).resolve()
    directories = {model: (root / name).resolve() for model, name in (model_dirs or MODEL_DIRS).items()}
    for model in MODEL_ORDER:
        directory = directories[model]
        if not directory.is_dir():
            raise ValueError(f"missing input directory: {directory}")
        if output == directory or directory in output.parents or output in directory.parents:
            raise ValueError("output and input directories must not overlap")
    relative_paths = sorted({path.relative_to(directory).as_posix()
                             for directory in directories.values()
                             for path in directory.rglob("*.json") if path.is_file()})
    rng = SeededRandom(seed)
    merged_count = 0
    skipped = []
    for relative in relative_paths:
        paths = [directories[model] / relative for model in MODEL_ORDER]
        missing = [model for model, path in zip(MODEL_ORDER, paths) if not path.is_file()]
        if missing:
            skipped.append({"path": relative, "missing_models": missing})
            continue
        records = []
        for path in paths:
            try:
                record = json.loads(path.read_text(encoding="utf-8-sig"))
            except (ValueError, OSError) as exc:
                raise ValueError(f"cannot read JSON: {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected a JSON object: {path}")
            records.append(record)
        merged = merge_records(records, rng)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(merged, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        merged_count += 1
    return {"output_dir": str(output), "merged_files": merged_count,
            "skipped_files": len(skipped), "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("annotation"),
                        help="Absolute output directory or path relative to --root.")
    parser.add_argument("--seed", type=int, default=42)
    for model in MODEL_ORDER:
        parser.add_argument(f"--{model}-dir", default=MODEL_DIRS[model],
                            help="Absolute input directory or path relative to --root.")
    args = parser.parse_args()
    try:
        report = merge_directories(args.root, args.output, args.seed,
                                   {model: getattr(args, f"{model}_dir") for model in MODEL_ORDER})
    except (ValueError, OSError) as exc:
        parser.exit(1, f"merge failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
