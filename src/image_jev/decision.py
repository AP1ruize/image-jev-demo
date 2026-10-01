"""Pointer readout and question isolation for a Jev-like score head.

The hosted Jev model is closed. This module follows the public reconstruction:
a shared state prefix, questions that cannot read each other, and a pointer
that scores each option's hidden state against a decision position
(Archer Hume, "Jev's Architecture Unmasked"; the pointer used by Kev).

Qwen3.5 cannot use one packed sequence plus an attention mask for that
isolation. Its text stack alternates Gated DeltaNet blocks (`linear_attention`
in the checkpoint config) with full attention. DeltaNet is recurrent and does
not read the attention mask, so a later question packed into the same row can
see an earlier question. Kev therefore gives each question its own row and
reuses the state prefix. `hybrid_rows` is that layout. `tree_attention_mask`
is the attention-only equivalent, kept so the isolation rule can be tested
without a GPU.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


def nearest_bin_distribution(value: float, labels: Sequence[float]) -> list[float]:
    if not labels:
        raise ValueError("labels must be non-empty")
    best = min(range(len(labels)), key=lambda index: (abs(float(labels[index]) - float(value)), index))
    target = [0.0] * len(labels)
    target[best] = 1.0
    return target


def distribution_errors(probabilities: Iterable[float], *, tol: float = 1e-5) -> list[str]:
    values = [float(item) for item in probabilities]
    errors = []
    if not values:
        errors.append("empty")
        return errors
    if any(not math.isfinite(item) for item in values):
        errors.append("nonfinite")
    if any(item < -tol for item in values):
        errors.append("negative")
    if abs(sum(values) - 1.0) > tol:
        errors.append("sum")
    return errors


def softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ValueError("logits must be non-empty")
    peak = max(logits)
    weights = [math.exp(float(item) - peak) for item in logits]
    total = sum(weights)
    return [item / total for item in weights]


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> list[int]:
    if not needle:
        raise ValueError("needle must be non-empty")
    width = len(needle)
    hits = []
    for start in range(len(haystack) - width + 1):
        if list(haystack[start : start + width]) == list(needle):
            hits.append(start)
    return hits


def tree_attention_mask(state_len: int, branch_lengths: Sequence[int]) -> list[list[int]]:
    """Causal mask where each branch reads the state and itself, not other branches.

    mask[query][key] is 1 when the query token may read the key token.
    """
    if state_len < 0 or any(length <= 0 for length in branch_lengths):
        raise ValueError("state length and branch lengths must be non-negative, branches non-empty")
    spans = []
    cursor = state_len
    for length in branch_lengths:
        spans.append((cursor, cursor + length))
        cursor += length
    total = cursor
    mask = [[0] * total for _ in range(total)]
    for query in range(state_len):
        for key in range(query + 1):
            mask[query][key] = 1
    for begin, end in spans:
        for query in range(begin, end):
            for key in range(state_len):
                mask[query][key] = 1
            for key in range(begin, query + 1):
                mask[query][key] = 1
    return mask


def restarted_position_ids(state_len: int, branch_lengths: Sequence[int]) -> list[int]:
    """State keeps 0..S-1. Each branch starts again at S, so order inside a question is local."""
    positions = list(range(state_len))
    for length in branch_lengths:
        positions.extend(range(state_len, state_len + length))
    return positions


def hybrid_rows(state_ids: Sequence[int], branch_ids: Sequence[Sequence[int]]) -> list[list[int]]:
    """One row per question: the same state prefix plus that question only.

    This is the Qwen3.5 serving layout. Rows share a prefix cache at runtime;
    they are not concatenated, because DeltaNet would otherwise mix questions.
    """
    if not branch_ids:
        raise ValueError("at least one question branch is required")
    state = list(state_ids)
    return [state + list(branch) for branch in branch_ids]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def pointer_logits(
    w_query: Sequence[Sequence[float]],
    w_key: Sequence[Sequence[float]],
    hidden_decide: Sequence[float],
    hidden_options: Sequence[Sequence[float]],
) -> list[float]:
    """score_i = (Wq h_decide) · (Wk h_option_i) / sqrt(projection)."""
    query = [_dot(row, hidden_decide) for row in w_query]
    scale = 1.0 / math.sqrt(len(w_query))
    logits = []
    for hidden in hidden_options:
        key = [_dot(row, hidden) for row in w_key]
        logits.append(_dot(query, key) * scale)
    return logits


def cross_entropy(probabilities: Sequence[float], target: Sequence[float]) -> float:
    loss = 0.0
    for prob, gold in zip(probabilities, target, strict=True):
        if gold == 0.0:
            continue
        loss -= float(gold) * math.log(max(float(prob), 1e-12))
    return loss


def _mean_loss(w_query, w_key, examples) -> float:
    total = 0.0
    for hidden_decide, hidden_options, target in examples:
        probabilities = softmax(pointer_logits(w_query, w_key, hidden_decide, hidden_options))
        total += cross_entropy(probabilities, target)
    return total / len(examples)


def train_pointer_synthetic(*, steps: int = 40, seed: int = 0, step_size: float = 0.05) -> dict:
    """Finite-difference fit on a tiny pointer task. No torch, no images.

    The correct option's vector matches the decision vector, so a pointer can
    drive that option's probability to 1. Used to check the readout before any
    VLM forward.
    """
    rng = random.Random(seed)
    hidden = 4
    projection = 4
    options = 3
    examples = []
    for index in range(6):
        label = index % options
        hidden_decide = [0.0] * hidden
        hidden_decide[label] = 1.0
        hidden_options = [[rng.uniform(-0.2, 0.2) for _ in range(hidden)] for _ in range(options)]
        hidden_options[label] = list(hidden_decide)
        target = [0.0] * options
        target[label] = 1.0
        examples.append((hidden_decide, hidden_options, target))

    def init():
        return [[rng.uniform(-0.1, 0.1) for _ in range(hidden)] for _ in range(projection)]

    w_query = init()
    w_key = init()
    losses = [_mean_loss(w_query, w_key, examples)]
    epsilon = 1e-3
    for _ in range(steps):
        for matrix in (w_query, w_key):
            for row in matrix:
                for col in range(hidden):
                    original = row[col]
                    row[col] = original + epsilon
                    higher = _mean_loss(w_query, w_key, examples)
                    row[col] = original - epsilon
                    lower = _mean_loss(w_query, w_key, examples)
                    row[col] = original - step_size * ((higher - lower) / (2 * epsilon))
        losses.append(_mean_loss(w_query, w_key, examples))
    final_probs = [
        softmax(pointer_logits(w_query, w_key, hidden_decide, hidden_options))
        for hidden_decide, hidden_options, _target in examples
    ]
    return {"losses": losses, "probabilities": final_probs}
