from __future__ import annotations

from collections import defaultdict
import random
import statistics
from typing import Any

from .parse import in_scale
from .scoring import continuous


def value_for(item: dict, response: dict) -> float | None:
    if response.get("error"):
        return None
    value = response.get("prediction")
    if value is None:
        return None
    if not in_scale(value, item["task"]["scale"]):
        return None
    return float(value)


def compare(
    gold: list[dict],
    left: list[dict],
    right: list[dict],
    *,
    resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    gold_by_id = {item["item_id"]: item for item in gold}
    left_by_id = {item["item_id"]: item for item in left}
    right_by_id = {item["item_id"]: item for item in right}
    expected = set(gold_by_id)
    if (
        len(left_by_id) != len(left)
        or len(right_by_id) != len(right)
        or set(left_by_id) != expected
        or set(right_by_id) != expected
    ):
        raise ValueError("paired comparison requires exactly one response for every frozen item in each run")
    by_source: dict[str, list[tuple[dict, float | None, float | None]]] = defaultdict(list)
    for item_id, item in gold_by_id.items():
        by_source[item["source_dataset"]].append(
            (item, value_for(item, left_by_id[item_id]), value_for(item, right_by_id[item_id]))
        )
    results = {}
    rng = random.Random(seed)
    for source, rows in sorted(by_source.items()):
        paired = [(item, a, b) for item, a, b in rows if a is not None and b is not None]
        left_schema = [bool(left_by_id[item["item_id"]].get("schema_valid")) for item, _, _ in rows]
        right_schema = [bool(right_by_id[item["item_id"]].get("schema_valid")) for item, _, _ in rows]
        left_lat = [
            float(left_by_id[item["item_id"]]["latency_ms"])
            for item, _, _ in rows
            if isinstance(left_by_id[item["item_id"]].get("latency_ms"), (int, float))
        ]
        right_lat = [
            float(right_by_id[item["item_id"]]["latency_ms"])
            for item, _, _ in rows
            if isinstance(right_by_id[item["item_id"]].get("latency_ms"), (int, float))
        ]
        payload: dict[str, Any] = {
            "primary_metric": "spearman_rho",
            "n_total": len(rows),
            "n_paired_valid": len(paired),
            "left_schema_valid": sum(left_schema) / len(left_schema) if left_schema else None,
            "right_schema_valid": sum(right_schema) / len(right_schema) if right_schema else None,
            "left_median_latency_ms": statistics.median(left_lat) if left_lat else None,
            "right_median_latency_ms": statistics.median(right_lat) if right_lat else None,
        }
        if payload["left_median_latency_ms"] and payload["right_median_latency_ms"]:
            payload["latency_ratio_right_over_left"] = (
                payload["right_median_latency_ms"] / payload["left_median_latency_ms"]
            )
        if not paired:
            payload.update({"left_spearman_rho": None, "right_spearman_rho": None, "delta_right_minus_left": None, "ci95": None})
            results[source] = payload
            continue
        left_rows = [(item, a) for item, a, _ in paired]
        right_rows = [(item, b) for item, _, b in paired]
        base = continuous(left_rows)["spearman_rho"]
        changed = continuous(right_rows)["spearman_rho"]
        clusters: dict[str, list] = defaultdict(list)
        for pair in paired:
            clusters[pair[0]["group_id"]].append(pair)
        ids = sorted(clusters)
        differences = []
        for _ in range(resamples):
            sample = [pair for group in (rng.choice(ids) for _ in ids) for pair in clusters[group]]
            a = continuous([(item, left_value) for item, left_value, _ in sample])["spearman_rho"]
            b = continuous([(item, right_value) for item, _, right_value in sample])["spearman_rho"]
            if a is not None and b is not None:
                differences.append(b - a)
        differences.sort()
        ci = (
            [differences[int(0.025 * (len(differences) - 1))], differences[int(0.975 * (len(differences) - 1))]]
            if len(differences) >= resamples // 2
            else None
        )
        payload.update(
            {
                "left_spearman_rho": base,
                "right_spearman_rho": changed,
                "delta_right_minus_left": None if base is None or changed is None else changed - base,
                "ci95": ci,
            }
        )
        results[source] = payload
    return {"n_expected": len(gold), "by_source": results}
