#!/usr/bin/env python3
"""Paired comparison of two complete Week 0 runs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.compare import compare  # noqa: E402
from image_jev.io import read_json, read_jsonl, write_json  # noqa: E402
from image_jev.report import render_compare  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--week0", type=Path, default=ROOT / "configs/week0.v0.1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--allow-subset", action="store_true", help="Compare the intersection of both runs (smokes only)")
    args = parser.parse_args()
    week0 = read_json(args.week0)
    release_dir = args.release_dir or (ROOT / "data/processed" / week0["release_id"])
    release = read_json(release_dir / "release.json")
    gold = read_jsonl(ROOT / release["artifacts"]["benchmark_gold"]["path"])
    left = read_jsonl(args.left)
    right = read_jsonl(args.right)
    if args.allow_subset:
        common = {row["item_id"] for row in left} & {row["item_id"] for row in right}
        gold = [item for item in gold if item["item_id"] in common]
        left = [row for row in left if row["item_id"] in common]
        right = [row for row in right if row["item_id"] in common]
    result = compare(gold, left, right, resamples=args.resamples)
    result["left"] = str(args.left)
    result["right"] = str(args.right)
    write_json(args.output, result)
    left_meta = read_json(args.left.with_suffix(".run.json")) if args.left.with_suffix(".run.json").is_file() else {}
    right_meta = read_json(args.right.with_suffix(".run.json")) if args.right.with_suffix(".run.json").is_file() else {}
    report_path = args.output.with_suffix(".md")
    report_path.write_text(render_compare(result, left_meta, right_meta), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
