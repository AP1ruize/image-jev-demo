from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import sha256_file, write_json, write_jsonl
from .runtime import ROOT
from .sources import collect_sources, copy_or_extract, rank_key, split_name


def select_test(records: list[dict], spec: dict) -> list[dict]:
    train = float(spec["split"]["train"])
    validation = float(spec["split"]["validation"])
    seed = spec["sampling_seed"]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        split = split_name(record["group_id"], seed=seed, train=train, validation=validation)
        if split != "test":
            continue
        by_source[record["source_dataset"]].append(record)
    selected = []
    for dataset, count in spec["benchmark_samples"].items():
        pool = sorted(by_source.get(dataset, []), key=lambda item: rank_key(dataset, item["source_id"], seed))
        if len(pool) < count:
            raise ValueError(f"{dataset}: need {count} test items, found {len(pool)}")
        selected.extend(pool[:count])
    return sorted(selected, key=lambda item: (item["source_dataset"], item["source_id"]))


def build_item(record: dict, asset: dict, spec: dict) -> dict:
    dataset = record["source_dataset"]
    meta = spec["datasets"][dataset]
    gold_key = meta["gold_key"]
    return {
        "item_id": f"{dataset}:{record['source_id']}:{gold_key}",
        "source_dataset": dataset,
        "source_id": record["source_id"],
        "split": "test",
        "group_id": record["group_id"],
        "task": {
            "dimension": meta["dimension"],
            "type": "continuous",
            "prompt_id": f"{spec['prompt_version']}:{dataset}",
            "scale": list(meta["scale"]),
            "gold_key": gold_key,
        },
        "assets": [asset],
        "gold": {"origin": "source_human", "value": float(record["labels"][gold_key]), "raw": record["labels"]},
    }


def freeze_release(spec: dict, dest: Path, *, local: dict | None = None) -> dict[str, Any]:
    records = collect_sources(local)
    selected = select_test(records, spec)
    items = []
    for record in selected:
        relative = Path("data/processed") / spec["release_id"] / "assets" / record["source_dataset"] / record["source_id"]
        target = ROOT / relative
        asset = copy_or_extract(record, target)
        asset["path"] = relative.as_posix()
        items.append(build_item(record, asset, spec))
    dest.mkdir(parents=True, exist_ok=True)
    run_items = [{k: v for k, v in item.items() if k != "gold"} for item in items]
    gold_items = items
    run_path = dest / "benchmark-run-items.jsonl"
    gold_path = dest / "benchmark-gold.jsonl"
    write_jsonl(run_path, run_items)
    write_jsonl(gold_path, gold_items)
    counts = defaultdict(int)
    for item in items:
        counts[item["source_dataset"]] += 1
    release = {
        "release_id": spec["release_id"],
        "sampling_seed": spec["sampling_seed"],
        "prompt_version": spec["prompt_version"],
        "purpose": spec.get("purpose"),
        "n_items": len(items),
        "by_source": dict(counts),
        "prompts": spec["prompts"],
        "datasets": spec["datasets"],
        "artifacts": {
            "benchmark_run_items": {"path": run_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(run_path)},
            "benchmark_gold": {"path": gold_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(gold_path)},
        },
    }
    write_json(dest / "release.json", release)
    return release
