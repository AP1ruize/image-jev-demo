from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
import statistics
from typing import Any

from .parse import in_scale


def ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average = (start + end + 1) / 2
        for position in range(start, end):
            result[ordered[position]] = average
        start = end
    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    mx, my = statistics.mean(left), statistics.mean(right)
    numerator = sum((x - mx) * (y - my) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in left) * sum((y - my) ** 2 for y in right))
    return numerator / denominator if denominator else None


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(ranks(left), ranks(right))


def continuous(rows: list[tuple[dict, float | None]]) -> dict[str, float | None]:
    valid = [
        (float(item["gold"]["value"]), float(prediction))
        for item, prediction in rows
        if prediction is not None
    ]
    if not valid:
        return {"spearman_rho": None, "pearson_r": None, "mae": None}
    left, right = zip(*valid)
    return {
        "spearman_rho": spearman(list(left), list(right)),
        "pearson_r": pearson(list(left), list(right)),
        "mae": statistics.mean(abs(x - y) for x, y in valid),
    }


def grouped_bootstrap_ci(
    rows: list[tuple[dict, float | None]],
    *,
    resamples: int = 2000,
    seed: int = 42,
) -> list[float] | None:
    clusters: dict[str, list[tuple[dict, float | None]]] = defaultdict(list)
    for item, prediction in rows:
        clusters[item["group_id"]].append((item, prediction))
    ids = sorted(clusters)
    if len(ids) < 2:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [pair for group in (rng.choice(ids) for _ in ids) for pair in clusters[group]]
        estimate = continuous(sample)["spearman_rho"]
        if estimate is not None:
            estimates.append(estimate)
    if len(estimates) < resamples // 2:
        return None
    estimates.sort()
    return [estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]]


def summarize(gold_items: list[dict], responses: list[dict]) -> dict[str, Any]:
    by_id = {item["item_id"]: item for item in gold_items}
    if len(by_id) != len(gold_items):
        raise ValueError("gold manifest has repeated item_id")
    seen: set[str] = set()
    grouped: dict[str, list[tuple[dict, float | None]]] = defaultdict(list)
    errors: Counter[str] = Counter()
    latency: dict[str, list[float]] = defaultdict(list)
    schema_flags: dict[str, list[bool]] = defaultdict(list)
    for response in responses:
        item_id = response["item_id"]
        if item_id in seen or item_id not in by_id:
            raise ValueError(f"duplicate/unknown response item_id: {item_id}")
        seen.add(item_id)
        item = by_id[item_id]
        source = item["source_dataset"]
        value = response.get("prediction") if not response.get("error") else None
        invalid_reason = None
        if value is not None and not in_scale(value, item["task"]["scale"]):
            value = None
            invalid_reason = "invalid_numeric_or_out_of_range"
        if value is None:
            errors[invalid_reason or response.get("error") or "missing_prediction"] += 1
        grouped[source].append((item, None if value is None else float(value)))
        if isinstance(response.get("latency_ms"), (int, float)):
            latency[source].append(float(response["latency_ms"]))
        schema_flags[source].append(bool(response.get("schema_valid")))
    if seen != set(by_id):
        errors["missing_response"] += len(set(by_id) - seen)
        for item_id in set(by_id) - seen:
            item = by_id[item_id]
            grouped[item["source_dataset"]].append((item, None))
            schema_flags[item["source_dataset"]].append(False)
    result = {}
    for source, rows in sorted(grouped.items()):
        valid = sum(prediction is not None for _, prediction in rows)
        flags = schema_flags[source]
        result[source] = {
            "task_type": "continuous",
            "n_total": len(rows),
            "n_valid": valid,
            "coverage": valid / len(rows) if rows else None,
            "schema_valid": sum(flags) / len(flags) if flags else None,
            "n_groups": len({item["group_id"] for item, _ in rows}),
            "median_latency_ms": statistics.median(latency[source]) if latency[source] else None,
            "primary_metric_ci95_group_bootstrap": grouped_bootstrap_ci(rows),
            **continuous(rows),
        }
    return {
        "n_expected": len(gold_items),
        "n_responses": len(responses),
        "errors": dict(errors),
        "by_source": result,
        "interpretation": (
            "Do not mix SPAQ and KonIQ MOS. Compare paired runs on identical items; "
            "report coverage and schema_valid beside Spearman/Pearson/MAE."
        ),
    }
