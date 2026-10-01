#!/usr/bin/env python3
"""Build SPAQ + KonIQ source catalogs from configs/local.json."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.io import write_json, write_jsonl  # noqa: E402
from image_jev.runtime import load_local_config  # noqa: E402
from image_jev.sources import collect_sources  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/interim/source-catalog.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/audits/source-catalog.json")
    args = parser.parse_args()
    records = collect_sources(load_local_config())
    if not records:
        raise SystemExit("no sources found; copy configs/local.example.json to configs/local.json")
    slim = []
    counts: Counter[str] = Counter()
    for record in records:
        counts[record["source_dataset"]] += 1
        slim.append(
            {
                "source_dataset": record["source_dataset"],
                "source_id": record["source_id"],
                "group_id": record["group_id"],
                "labels": {"MOS": record["labels"]["MOS"]},
            }
        )
    write_jsonl(args.output, slim)
    write_json(args.report, {"n": len(records), "by_source": dict(counts)})
    print(json.dumps({"n": len(records), "by_source": dict(counts), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
