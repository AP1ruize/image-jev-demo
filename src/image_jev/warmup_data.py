"""Small SPAQ score records for a decision-head warmup.

Records follow Valen's shape: one state, typed questions, and a target
distribution. Week 0's frozen test split is never selected.
"""

from __future__ import annotations

from typing import Any, Iterable

from .bins import bin_labels, format_bin
from .sources import rank_key, split_name


def score_markers(labels: Iterable[float]) -> tuple[list[str], str]:
    options = []
    for index, label in enumerate(labels):
        options.append(f"||OPT{index:02d}:{format_bin(float(label))}||")
    return options, "||DECIDE||"


def score_user_text(instruction: str, labels: list[float]) -> str:
    options, decide = score_markers(labels)
    lines = [instruction.strip(), "Ordered score levels, lowest to highest:"]
    lines.extend(options)
    lines.append(decide)
    return "\n".join(lines)


def select_train_records(
    records: list[dict[str, Any]],
    *,
    dataset: str,
    seed: str,
    train: float,
    validation: float,
    n: int,
    exclude_source_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    blocked = set(exclude_source_ids)
    pool = []
    for record in records:
        if record.get("source_dataset") != dataset:
            continue
        if record.get("source_id") in blocked:
            continue
        split = split_name(record["group_id"], seed=seed, train=train, validation=validation)
        if split != "train":
            continue
        pool.append(record)
    pool.sort(key=lambda record: rank_key(dataset, record["source_id"], seed))
    if len(pool) < n:
        raise ValueError(f"need {n} train records for {dataset}, found {len(pool)}")
    chosen = pool[:n]
    for record in chosen:
        split = split_name(record["group_id"], seed=seed, train=train, validation=validation)
        if split != "train":
            raise RuntimeError(f"refusing non-train record {record['source_id']}")
    return chosen


def score_record(record: dict[str, Any], *, instruction: str, labels: list[float], question_id: str = "mos") -> dict:
    target = _nearest(float(record["labels"]["MOS"]), labels)
    criteria = [format_bin(label) for label in labels]
    return {
        "group_id": record["group_id"],
        "source_dataset": record["source_dataset"],
        "source_id": record["source_id"],
        "image": record["path"],
        "request": {
            "state": {"image": record["path"], "text": instruction},
            "questions": {
                question_id: {
                    "type": "score",
                    "instructions": instruction,
                    "criteria": criteria,
                }
            },
        },
        "targets": {question_id: {"probabilities": dict(zip(criteria, target, strict=True))}},
        "gold_mos": float(record["labels"]["MOS"]),
    }


def _nearest(value: float, labels: list[float]) -> list[float]:
    from .decision import nearest_bin_distribution

    return nearest_bin_distribution(value, labels)


def spaq_labels(spec: dict) -> list[float]:
    meta = spec["datasets"]["spaq"]
    return bin_labels(meta["bin_start"], meta["bin_stop"], meta["bin_step"])
