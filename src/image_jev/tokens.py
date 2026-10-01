from __future__ import annotations

from dataclasses import dataclass
import math

from .bins import format_bin, letter_for_index


@dataclass(frozen=True)
class Candidate:
    label: float
    display: str
    letter: str
    token_id: int
    token_text: str


def _encode_variants(encode, text: str) -> list[list[int]]:
    variants = [text, f" {text}", f"\n{text}"]
    encoded = []
    for variant in variants:
        ids = encode(variant)
        if ids:
            encoded.append(ids)
    return encoded


def resolve_candidates(encode, id_to_text: dict[int, str], labels: list[float]) -> list[Candidate]:
    """Map MOS bins to single token ids.

    Prefer a single-token encoding of the number (with optional leading space).
    Fall back to option letters A, B, … which Qwen tokenizers treat as one token.
    """
    candidates: list[Candidate] = []
    for index, label in enumerate(labels):
        letter = letter_for_index(index)
        display = format_bin(label)
        token_id = None
        token_text = None
        for ids in _encode_variants(encode, display):
            if len(ids) == 1:
                token_id = ids[0]
                token_text = id_to_text.get(ids[0], display)
                break
        if token_id is None:
            letter_ids = encode(letter) or encode(f" {letter}")
            if not letter_ids or len(letter_ids) != 1:
                # Last resort: first token of the letter encoding.
                if not letter_ids:
                    raise ValueError(f"tokenizer cannot encode option letter {letter}")
                letter_ids = letter_ids[:1]
            token_id = letter_ids[0]
            token_text = id_to_text.get(token_id, letter)
        candidates.append(
            Candidate(label=label, display=display, letter=letter, token_id=token_id, token_text=token_text)
        )
    ids = [item.token_id for item in candidates]
    if len(set(ids)) != len(ids):
        # Numeric tokens collided; force letters.
        rebuilt = []
        for index, label in enumerate(labels):
            letter = letter_for_index(index)
            letter_ids = encode(letter) or encode(f" {letter}")
            if not letter_ids:
                raise ValueError(f"tokenizer cannot encode option letter {letter}")
            rebuilt.append(
                Candidate(
                    label=label,
                    display=format_bin(label),
                    letter=letter,
                    token_id=letter_ids[0],
                    token_text=id_to_text.get(letter_ids[0], letter),
                )
            )
        ids = [item.token_id for item in rebuilt]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate token ids are not unique even after letter fallback")
        return rebuilt
    return candidates


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [value / temperature for value in values]
    peak = max(scaled)
    exps = [math.exp(value - peak) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]
