#!/usr/bin/env python3
"""Score a Week 0 prediction JSONL against the frozen gold file."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.io import read_json, read_jsonl, sha256_file, write_json  # noqa: E402
from image_jev.report import render_score  # noqa: E402
from image_jev.scoring import summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--week0", type=Path, default=ROOT / "configs/week0.v0.1.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-untracked", action="store_true")
    args = parser.parse_args()
    week0 = read_json(args.week0)
    release_dir = args.release_dir or (ROOT / "data/processed" / week0["release_id"])
    release = read_json(release_dir / "release.json")
    gold_path = ROOT / release["artifacts"]["benchmark_gold"]["path"]
    if sha256_file(gold_path) != release["artifacts"]["benchmark_gold"]["sha256"]:
        raise SystemExit("gold manifest hash differs from frozen release")
    metadata_path = args.responses.with_suffix(".run.json")
    if not metadata_path.is_file() and not args.allow_untracked:
        raise SystemExit(f"run metadata missing: {metadata_path}")
    run_meta = read_json(metadata_path) if metadata_path.is_file() else {"release_id": release["release_id"]}
    gold = read_jsonl(gold_path)
    responses = read_jsonl(args.responses)
    if run_meta.get("items_requested") and int(run_meta["items_requested"]) < len(gold):
        gold = [item for item in gold if item["item_id"] in {row["item_id"] for row in responses}]
    score = summarize(gold, responses)
    score["release"] = release["release_id"]
    output = args.output or args.responses.with_suffix(".score.json")
    write_json(output, score)
    report_path = ROOT / "outputs/reports" / f"{args.responses.stem}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_score(score, run_meta), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
