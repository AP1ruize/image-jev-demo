#!/usr/bin/env python3
"""Freeze a Week 0 technical-IQA eval split and copy selected assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.freeze import freeze_release  # noqa: E402
from image_jev.io import read_json  # noqa: E402
from image_jev.runtime import load_local_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/week0.v0.1.json")
    parser.add_argument("--dest", type=Path)
    args = parser.parse_args()
    spec = read_json(args.config)
    dest = args.dest or (ROOT / "data/processed" / spec["release_id"])
    release = freeze_release(spec, dest, local=load_local_config())
    print(json.dumps({"release_id": release["release_id"], "n_items": release["n_items"], "by_source": release["by_source"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
