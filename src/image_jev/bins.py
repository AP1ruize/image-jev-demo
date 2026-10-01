from __future__ import annotations

import string
from typing import Iterable


def bin_labels(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("bin step must be positive")
    values = []
    current = float(start)
    # Guard against float drift at the end of the range.
    while current <= float(stop) + 1e-9:
        values.append(_clean(current))
        current += float(step)
    if not values or abs(values[-1] - float(stop)) > 1e-6:
        raise ValueError(f"bins [{start}, {stop}] step {step} do not land on stop")
    return values


def _clean(value: float) -> float:
    rounded = round(value, 6)
    if abs(rounded - round(rounded)) < 1e-9:
        return float(int(round(rounded)))
    return float(rounded)


def format_bin(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text


def letter_for_index(index: int) -> str:
    if index < 0 or index >= len(string.ascii_uppercase):
        raise ValueError(f"letter index out of range: {index}")
    return string.ascii_uppercase[index]


def expected_value(probabilities: Iterable[float], values: Iterable[float]) -> float:
    total = 0.0
    mass = 0.0
    for probability, value in zip(probabilities, values, strict=True):
        total += float(probability) * float(value)
        mass += float(probability)
    if mass <= 0:
        raise ValueError("probabilities must sum to a positive mass")
    return total / mass


def argmax_label(probabilities: list[float], labels: list[float]) -> float:
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must be non-empty and aligned")
    best = max(range(len(probabilities)), key=lambda i: (probabilities[i], -i))
    return labels[best]
