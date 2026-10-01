from __future__ import annotations

import time
from typing import Any

from .bins import argmax_label, bin_labels, expected_value
from .grammar import compile_number_tokens, make_prefix_allowed_fn
from .models import tokenizer_of
from .parse import parse_value
from .prompts import generate_prompt, logits_prompt
from .tokens import resolve_candidates, softmax


JSON_PREFIX = '{"value":'


def _move(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _chat(processor, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        return processor.apply_chat_template(
            messages,
            chat_template_kwargs={"enable_thinking": enable_thinking},
            **kwargs,
        )
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def _inputs(processor, image, prompt: str, enable_thinking: bool, device, suffix: str = ""):
    text = _chat(processor, prompt, enable_thinking) + suffix
    packed = processor(text=[text], images=[image], return_tensors="pt")
    return _move(packed, device)


def _decode_new(processor, output_ids, prompt_len: int) -> str:
    return processor.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()


def _encode(processor, text: str) -> list[int]:
    tokenizer = tokenizer_of(processor)
    return list(tokenizer.encode(text, add_special_tokens=False))


def _id_to_text(processor, vocab_limit: int | None = None) -> dict[int, str]:
    tokenizer = tokenizer_of(processor)
    size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    if vocab_limit is not None:
        size = min(size, vocab_limit)
    mapping = {}
    for idx in range(size):
        mapping[idx] = tokenizer.decode([idx], skip_special_tokens=True)
    return mapping


def generate_json(model, processor, image, prompt: str, *, max_new_tokens: int, enable_thinking: bool) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    inputs = _inputs(processor, image, prompt, enable_thinking, device)
    prompt_len = inputs["input_ids"].shape[1]
    started = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    raw = _decode_new(processor, output[0], prompt_len)
    value, error = parse_value(raw)
    return {
        "mode": "generate",
        "prediction": value,
        "error": error,
        "schema_valid": error is None,
        "raw_response": raw,
        "latency_ms": latency_ms,
        "n_new_tokens": int(output.shape[1] - prompt_len),
        "n_forward_passes": int(output.shape[1] - prompt_len),
        "bin_label": None,
        "bin_probability": None,
        "probabilities": None,
    }


def generate_grammar(
    model,
    processor,
    image,
    prompt: str,
    *,
    max_new_tokens: int,
    enable_thinking: bool,
    id_to_text: dict[int, str],
    groups: dict[str, list[int]],
) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    tokenizer = tokenizer_of(processor)
    inputs = _inputs(processor, image, prompt, enable_thinking, device, suffix=JSON_PREFIX)
    prompt_len = inputs["input_ids"].shape[1]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    prefix_fn = make_prefix_allowed_fn(
        prompt_len,
        lambda ids: tokenizer.decode(list(ids), skip_special_tokens=True),
        groups,
        id_to_text,
        eos_id,
    )
    started = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            prefix_allowed_tokens_fn=prefix_fn,
        )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    body = _decode_new(processor, output[0], prompt_len)
    raw = JSON_PREFIX + body
    if not raw.endswith("}") and body.strip():
        raw = raw.rstrip() + "}"
    value, error = parse_value(raw)
    schema_valid = error is None
    return {
        "mode": "grammar",
        "prediction": value,
        "error": error,
        "schema_valid": schema_valid,
        "raw_response": raw,
        "latency_ms": latency_ms,
        "n_new_tokens": int(output.shape[1] - prompt_len),
        "n_forward_passes": int(output.shape[1] - prompt_len),
        "bin_label": None,
        "bin_probability": None,
        "probabilities": None,
    }


def score_logits(
    model,
    processor,
    image,
    prompt: str,
    labels: list[float],
    *,
    enable_thinking: bool,
    temperature: float = 1.0,
) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    encode = lambda text: _encode(processor, text)
    candidates = resolve_candidates(encode, {}, labels)
    inputs = _inputs(processor, image, prompt, enable_thinking, device)
    started = time.perf_counter()
    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits[0, -1]
        scores = [float(logits[candidate.token_id].item()) for candidate in candidates]
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    probs = softmax(scores, temperature=temperature)
    prediction = expected_value(probs, labels)
    chosen = argmax_label(probs, labels)
    chosen_index = labels.index(chosen)
    distribution = {candidate.letter: prob for candidate, prob in zip(candidates, probs)}
    return {
        "mode": "logits",
        "prediction": prediction,
        "error": None,
        "schema_valid": True,
        "raw_response": None,
        "latency_ms": latency_ms,
        "n_new_tokens": 0,
        "n_forward_passes": 1,
        "bin_label": candidates[chosen_index].display,
        "bin_probability": probs[chosen_index],
        "probabilities": distribution,
    }


def prepare_grammar_tables(processor) -> tuple[dict[int, str], dict[str, list[int]]]:
    id_to_text = _id_to_text(processor)
    return id_to_text, compile_number_tokens(id_to_text)


def run_item(
    model,
    processor,
    image,
    item: dict,
    spec: dict,
    mode: str,
    *,
    max_new_tokens: int,
    enable_thinking: bool,
    grammar_tables=None,
) -> dict[str, Any]:
    dataset = item["source_dataset"]
    if mode == "generate":
        result = generate_json(
            model,
            processor,
            image,
            generate_prompt(spec, dataset),
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )
    elif mode == "grammar":
        if grammar_tables is None:
            grammar_tables = prepare_grammar_tables(processor)
        id_to_text, groups = grammar_tables
        result = generate_grammar(
            model,
            processor,
            image,
            generate_prompt(spec, dataset),
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            id_to_text=id_to_text,
            groups=groups,
        )
    elif mode == "logits":
        meta = spec["datasets"][dataset]
        labels = bin_labels(meta["bin_start"], meta["bin_stop"], meta["bin_step"])
        result = score_logits(
            model,
            processor,
            image,
            logits_prompt(spec, dataset),
            labels,
            enable_thinking=enable_thinking,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    result["item_id"] = item["item_id"]
    return result
