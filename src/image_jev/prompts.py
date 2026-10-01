from __future__ import annotations

from .bins import bin_labels, format_bin, letter_for_index


def generate_prompt(spec: dict, dataset: str) -> str:
    return spec["prompts"][dataset]["generate"]


def logits_prompt(spec: dict, dataset: str) -> str:
    labels = bin_labels(
        spec["datasets"][dataset]["bin_start"],
        spec["datasets"][dataset]["bin_stop"],
        spec["datasets"][dataset]["bin_step"],
    )
    lines = [spec["prompts"][dataset]["logits"]]
    for index, value in enumerate(labels):
        lines.append(f"{letter_for_index(index)}. {format_bin(value)}")
    lines.append("Answer with the letter only.")
    return "\n".join(lines)


def field_suffix(instruction: str, labels: list[float]) -> str:
    lines = [instruction]
    for index, value in enumerate(labels):
        lines.append(f"{letter_for_index(index)}. {format_bin(value)}")
    lines.append("Answer with the letter only.")
    return "\n".join(lines)
