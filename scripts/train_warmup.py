#!/usr/bin/env python3
"""Train a pointer score head on a handful of SPAQ train photos.

Valen calls this stage warmup: the backbone stays frozen and only the shared
decision head is updated. Qwen3.5-2B bf16 does not fit in 4GB, so the frozen
forward uses the Week 0 4-bit loader. Hidden states are cached, then the head
is fit on CPU. One score question per image, so question isolation is not
exercised at runtime; see image_jev.decision for the row layout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.runtime import apply_runtime_env, load_local_config  # noqa: E402

apply_runtime_env(load_local_config())

from image_jev.bins import expected_value, format_bin  # noqa: E402
from image_jev.decision import distribution_errors, find_subsequence  # noqa: E402
from image_jev.io import read_json, write_json, write_jsonl  # noqa: E402
from image_jev.sources import collect_sources  # noqa: E402
from image_jev.warmup_data import score_markers, score_record, score_user_text, select_train_records, spaq_labels  # noqa: E402


def _encode_marker(processor, marker: str) -> list[int]:
    from image_jev.models import tokenizer_of

    tokenizer = tokenizer_of(processor)
    return list(tokenizer.encode(marker, add_special_tokens=False))


def _span_end(input_ids: list[int], needle: list[int], marker: str) -> int:
    hits = find_subsequence(input_ids, needle)
    if len(hits) != 1:
        raise RuntimeError(f"marker {marker} found {len(hits)} times in input ids")
    return hits[0] + len(needle) - 1


def _last_language_layer(model):
    import torch

    preferred = []
    fallback = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue
        if name.endswith("layers") and "visual" not in name:
            if "language" in name:
                preferred.append(module)
            else:
                fallback.append(module)
    if preferred:
        return preferred[-1][-1]
    if fallback:
        return fallback[-1][-1]
    raise RuntimeError("could not find a language-model layer list")


def _cache_features(model, processor, records: list[dict], labels: list[float], instruction: str, device) -> list[dict]:
    import torch
    from PIL import Image

    from image_jev.inference import _inputs

    options, decide = score_markers(labels)
    option_needles = [_encode_marker(processor, marker) for marker in options]
    decide_needle = _encode_marker(processor, decide)
    prompt = score_user_text(instruction, labels)
    layer = _last_language_layer(model)
    cached = []
    for index, record in enumerate(records, start=1):
        with Image.open(record["image"]) as image:
            rgb = image.convert("RGB")
        inputs = _inputs(processor, rgb, prompt, False, device)
        captured: dict = {}

        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = tensor

        handle = layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(**inputs)
        finally:
            handle.remove()
        if "hidden" not in captured:
            raise RuntimeError("language-layer hook did not capture a hidden state")
        # Clone outside inference mode so the pointer head can backprop into a normal tensor.
        hidden = captured["hidden"][0].detach().to(device="cpu", dtype=torch.float32).clone()
        ids = inputs["input_ids"][0].tolist()
        option_rows = [hidden[_span_end(ids, needle, marker)] for needle, marker in zip(option_needles, options, strict=True)]
        decide_row = hidden[_span_end(ids, decide_needle, decide)]
        target = record["targets"]["mos"]["probabilities"]
        ordered = [float(target[format_bin(label)]) for label in labels]
        cached.append(
            {
                "source_id": record["source_id"],
                "gold_mos": record["gold_mos"],
                "h_decide": decide_row,
                "h_options": torch.stack(option_rows, dim=0),
                "target": torch.tensor(ordered, dtype=torch.float32),
            }
        )
        del inputs, hidden, captured
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"cached {index}/{len(records)} {record['source_id']}", flush=True)
    return cached


def _fit_head(features: list[dict], *, hidden_size: int, projection: int, epochs: int, lr: float, seed: int) -> tuple[object, list[float]]:
    import torch

    torch.manual_seed(seed)
    head_q = torch.nn.Linear(hidden_size, projection, bias=False)
    head_k = torch.nn.Linear(hidden_size, projection, bias=False)
    torch.nn.init.normal_(head_q.weight, std=0.02)
    torch.nn.init.normal_(head_k.weight, std=0.02)
    scale = projection ** -0.5
    optimizer = torch.optim.AdamW([head_q.weight, head_k.weight], lr=lr)
    losses = []
    for _epoch in range(epochs):
        optimizer.zero_grad()
        batch = []
        for item in features:
            query = head_q(item["h_decide"])
            keys = head_k(item["h_options"])
            logits = (keys * query).sum(dim=-1) * scale
            log_probs = torch.log_softmax(logits, dim=0)
            batch.append(-(item["target"] * log_probs).sum())
        loss = torch.stack(batch).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return (head_q, head_k, scale), losses


def _predict(head, features: list[dict], labels: list[float]) -> list[dict]:
    import torch

    head_q, head_k, scale = head
    rows = []
    with torch.no_grad():
        for item in features:
            query = head_q(item["h_decide"])
            keys = head_k(item["h_options"])
            logits = (keys * query).sum(dim=-1) * scale
            probabilities = torch.softmax(logits, dim=0).tolist()
            errors = distribution_errors(probabilities)
            if errors:
                raise RuntimeError(f"{item['source_id']} invalid distribution: {errors}")
            rows.append(
                {
                    "source_id": item["source_id"],
                    "gold_mos": item["gold_mos"],
                    "probabilities": {format_bin(label): round(prob, 6) for label, prob in zip(labels, probabilities, strict=True)},
                    "expected": round(expected_value(probabilities, labels), 4),
                    "distribution_valid": True,
                }
            )
    return rows


def _load_for_pixels(config: dict, max_pixels: int):
    from image_jev.models import load_model

    profile = dict(config)
    profile["max_pixels"] = max_pixels
    return load_model(profile)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/warmup.spaq.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/warmup/spaq-smoke.json")
    args = parser.parse_args()

    config = read_json(args.config)
    week0 = read_json(ROOT / config["week0"])
    labels = spaq_labels(week0)
    instruction = week0["prompts"]["spaq"]["logits"].strip()
    records = collect_sources(load_local_config())
    chosen = select_train_records(
        records,
        dataset=config["dataset"],
        seed=week0["sampling_seed"],
        train=float(week0["split"]["train"]),
        validation=float(week0["split"]["validation"]),
        n=int(config["n_items"]),
    )
    jsonl_records = [score_record(record, instruction=instruction, labels=labels) for record in chosen]
    jsonl_path = ROOT / "data/exports/warmup-spaq-v0.1/train.jsonl"
    write_jsonl(jsonl_path, jsonl_records)
    print(f"wrote {len(jsonl_records)} train records to {jsonl_path}", flush=True)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. The frozen Qwen3.5-2B forward needs the 4GB GPU.")

    model = None
    try:
        model, processor = _load_for_pixels(config, int(config["max_pixels"]))
        features = _cache_features(model, processor, jsonl_records, labels, instruction, torch.device("cuda:0"))
        used_pixels = int(config["max_pixels"])
    except torch.OutOfMemoryError:
        del model
        model = None
        torch.cuda.empty_cache()
        fallback = int(config["oom_max_pixels"])
        print(f"out of memory at max_pixels={config['max_pixels']}, retrying at {fallback}", flush=True)
        model, processor = _load_for_pixels(config, fallback)
        features = _cache_features(model, processor, jsonl_records, labels, instruction, torch.device("cuda:0"))
        used_pixels = fallback
    del model
    torch.cuda.empty_cache()

    hidden_size = int(features[0]["h_decide"].shape[-1])
    head, losses = _fit_head(
        features,
        hidden_size=hidden_size,
        projection=int(config["projection_dim"]),
        epochs=int(config["epochs"]),
        lr=float(config["head_lr"]),
        seed=int(config["seed"]),
    )
    predictions = _predict(head, features, labels)
    head_q, head_k, _scale = head
    head_path = args.output.with_suffix(".pt")
    head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"q": head_q.state_dict(), "k": head_k.state_dict(), "projection_dim": int(config["projection_dim"])}, head_path)
    report = {
        "config": config["model_id"],
        "base_model": config["base_model"],
        "quantization": "4bit" if config.get("load_in_4bit") else "none",
        "max_pixels": used_pixels,
        "n_items": len(predictions),
        "epochs": int(config["epochs"]),
        "head_lr": float(config["head_lr"]),
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "losses": [round(item, 6) for item in losses],
        "all_distributions_valid": all(row["distribution_valid"] for row in predictions),
        "predictions": predictions,
        "train_jsonl": str(jsonl_path.relative_to(ROOT)),
        "head_path": str(head_path.relative_to(ROOT)),
        "split_seed": week0["sampling_seed"],
    }
    write_json(args.output, report)
    print(json.dumps({"loss_start": report["loss_start"], "loss_end": report["loss_end"], "valid": report["all_distributions_valid"]}, indent=2), flush=True)
    if not report["loss_decreased"] or not report["all_distributions_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
