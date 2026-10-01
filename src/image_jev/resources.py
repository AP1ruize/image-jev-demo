"""Inference latency and memory meters shared by the pointer head and the base VLM."""

from __future__ import annotations

import statistics
from typing import Iterable


def mb(num_bytes: int | float) -> float:
    return round(float(num_bytes) / (1024 ** 2), 2)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round(fraction * (len(ordered) - 1)))
    return ordered[index]


def latency_summary(latencies: Iterable[float]) -> dict:
    values = [float(item) for item in latencies]
    if not values:
        return {"n": 0, "median_latency_ms": None, "p95_latency_ms": None}
    return {
        "n": len(values),
        "median_latency_ms": round(statistics.median(values), 2),
        "p95_latency_ms": round(percentile(values, 0.95), 2),
        "mean_latency_ms": round(statistics.mean(values), 2),
    }


def process_rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return mb(psutil.Process().memory_info().rss)


def cuda_snapshot() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False, "process_rss_mb": process_rss_mb()}
    return {
        "cuda": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_allocated_mb": mb(torch.cuda.memory_allocated()),
        "gpu_reserved_mb": mb(torch.cuda.memory_reserved()),
        "gpu_max_allocated_mb": mb(torch.cuda.max_memory_allocated()),
        "gpu_max_reserved_mb": mb(torch.cuda.max_memory_reserved()),
        "process_rss_mb": process_rss_mb(),
    }


def reset_cuda_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
