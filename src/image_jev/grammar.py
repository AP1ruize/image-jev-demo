from __future__ import annotations

import re
from collections.abc import Callable, Sequence


_NUMBER = re.compile(r"^[ ]?(?:\d+\.?\d*|\.\d+)?$")
_COMPLETE = re.compile(r"^[ ]?\d+(?:\.\d+)?$")


def is_number_prefix(text: str) -> bool:
    return bool(_NUMBER.fullmatch(text))


def is_complete_number(text: str) -> bool:
    return bool(_COMPLETE.fullmatch(text.strip()) if text.strip() else False)


def json_number_allowed(decoded_suffix: str, token_text: str) -> bool:
    """Whether emitting token_text after decoded_suffix stays inside `N` or `N}`."""
    if token_text == "":
        return False
    candidate = decoded_suffix + token_text
    if candidate.endswith("}"):
        body = candidate[:-1]
        return is_complete_number(body) and "}" not in body
    if "}" in candidate:
        return False
    return is_number_prefix(candidate)


def compile_number_tokens(id_to_text: dict[int, str]) -> dict[str, list[int]]:
    digits: list[int] = []
    dots: list[int] = []
    braces: list[int] = []
    spaces: list[int] = []
    fragments: list[int] = []
    for token_id, text in id_to_text.items():
        if text == "}":
            braces.append(token_id)
        elif text == ".":
            dots.append(token_id)
        elif text == " ":
            spaces.append(token_id)
        elif text.isdigit():
            digits.append(token_id)
            fragments.append(token_id)
        elif is_number_prefix(text) and text not in {".", " ", ""}:
            fragments.append(token_id)
    return {
        "digits": digits,
        "dots": dots,
        "braces": braces,
        "spaces": spaces,
        "fragments": fragments,
    }


def allowed_token_ids(
    decoded_suffix: str,
    groups: dict[str, list[int]],
    id_to_text: dict[int, str],
) -> list[int]:
    pool: list[int] = []
    seen: set[int] = set()
    for key in ("fragments", "digits", "dots", "spaces", "braces"):
        for token_id in groups[key]:
            if token_id in seen:
                continue
            seen.add(token_id)
            if json_number_allowed(decoded_suffix, id_to_text[token_id]):
                pool.append(token_id)
    return pool


def make_prefix_allowed_fn(
    prompt_len: int,
    decode_ids: Callable[[Sequence[int]], str],
    groups: dict[str, list[int]],
    id_to_text: dict[int, str],
    eos_id: int | None,
) -> Callable[[int, Sequence[int]], list[int]]:
    def prefix_allowed_tokens_fn(_batch_id: int, input_ids: Sequence[int]) -> list[int]:
        generated = list(input_ids[prompt_len:])
        decoded = decode_ids(generated)
        if decoded.endswith("}"):
            return [eos_id] if eos_id is not None else []
        allowed = allowed_token_ids(decoded, groups, id_to_text)
        if not allowed and eos_id is not None and is_complete_number(decoded):
            return [eos_id]
        return allowed

    return prefix_allowed_tokens_fn
