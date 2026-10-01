#!/usr/bin/env python3
"""Fit a pointer head on SPAQ train photos and score the Week 0 validation split.

The frozen test split is not read. Validation is paired against the Week 0
logits readout on the same images. Reports follow the AlphaHalfBench
per-source layout: Spearman, Pearson, MAE, coverage, and grouped bootstrap CI.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.runtime import apply_runtime_env, load_local_config  # noqa: E402

apply_runtime_env(load_local_config())

from image_jev.bins import expected_value, format_bin  # noqa: E402
from image_jev.compare import compare  # noqa: E402
from image_jev.decision import distribution_errors, find_subsequence  # noqa: E402
from image_jev.io import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from image_jev.prompts import logits_prompt  # noqa: E402
from image_jev.report import render_heldout  # noqa: E402
from image_jev.scoring import continuous, summarize  # noqa: E402
from image_jev.sources import collect_sources  # noqa: E402
from image_jev.warmup_data import (  # noqa: E402
    score_markers,
    score_record,
    score_user_text,
    select_split_records,
    spaq_labels,
)


def _encode_marker(processor, marker: str) -> list[int]:
    from image_jev.models import tokenizer_of

    return list(tokenizer_of(processor).encode(marker, add_special_tokens=False))


def _span_end(input_ids: list[int], needle: list[int], marker: str) -> int:
    hits = find_subsequence(input_ids, needle)
    if len(hits) != 1:
        raise RuntimeError(f"marker {marker} found {len(hits)} times")
    return hits[0] + len(needle) - 1


def _last_language_layer(model):
    import torch

    preferred = []
    fallback = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or not len(module):
            continue
        if name.endswith("layers") and "visual" not in name:
            (preferred if "language" in name else fallback).append(module)
    chosen = preferred[-1] if preferred else (fallback[-1] if fallback else None)
    if chosen is None:
        raise RuntimeError("could not find a language-model layer list")
    return chosen[-1]


def _item_id(source_id: str) -> str:
    return f"spaq:{source_id}:MOS"


def _gold(record: dict) -> dict:
    return {
        "item_id": _item_id(record["source_id"]),
        "source_dataset": "spaq",
        "source_id": record["source_id"],
        "split": "validation",
        "group_id": record["group_id"],
        "task": {
            "dimension": "technical_quality",
            "type": "continuous",
            "prompt_id": "week0-iqa-v0.1:spaq",
            "scale": [0, 100],
            "gold_key": "MOS",
        },
        "gold": {"origin": "source_human", "value": float(record["gold_mos"])},
    }


def _pointer_feature(model, processor, record: dict, labels: list[float], prompt: str, device) -> dict:
    import torch
    from PIL import Image

    from image_jev.inference import _inputs

    options, decide = score_markers(labels)
    option_needles = [_encode_marker(processor, marker) for marker in options]
    decide_needle = _encode_marker(processor, decide)
    layer = _last_language_layer(model)
    with Image.open(record["image"]) as image:
        rgb = image.convert("RGB")
    inputs = _inputs(processor, rgb, prompt, False, device)
    captured: dict = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = output[0] if isinstance(output, tuple) else output

    handle = layer.register_forward_hook(hook)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    hidden = captured["hidden"][0].detach().to(device="cpu", dtype=torch.float32).clone()
    ids = inputs["input_ids"][0].tolist()
    option_rows = [hidden[_span_end(ids, needle, marker)] for needle, marker in zip(option_needles, options, strict=True)]
    target = record["targets"]["mos"]["probabilities"]
    payload = {
        "source_id": record["source_id"],
        "group_id": record["group_id"],
        "gold_mos": float(record["gold_mos"]),
        "latency_ms": latency_ms,
        "h_decide": hidden[_span_end(ids, decide_needle, decide)],
        "h_options": torch.stack(option_rows, dim=0),
        "target": torch.tensor([float(target[format_bin(label)]) for label in labels], dtype=torch.float32),
    }
    del inputs, hidden, captured
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _mean_loss(head_q, head_k, scale, features) -> "object":
    import torch

    losses = []
    for item in features:
        query = head_q(item["h_decide"])
        keys = head_k(item["h_options"])
        logits = (keys * query).sum(dim=-1) * scale
        log_probs = torch.log_softmax(logits, dim=0)
        losses.append(-(item["target"] * log_probs).sum())
    return torch.stack(losses).mean()


def _fit(train_features, val_features, *, hidden_size: int, projection: int, epochs: int, lr: float, seed: int):
    import torch

    torch.manual_seed(seed)
    head_q = torch.nn.Linear(hidden_size, projection, bias=False)
    head_k = torch.nn.Linear(hidden_size, projection, bias=False)
    torch.nn.init.normal_(head_q.weight, std=0.02)
    torch.nn.init.normal_(head_k.weight, std=0.02)
    scale = projection ** -0.5
    optimizer = torch.optim.AdamW([head_q.weight, head_k.weight], lr=lr)

    def snapshot():
        return (
            {key: value.detach().cpu().clone() for key, value in head_q.state_dict().items()},
            {key: value.detach().cpu().clone() for key, value in head_k.state_dict().items()},
        )

    with torch.no_grad():
        train_start = float(_mean_loss(head_q, head_k, scale, train_features))
        val_start = float(_mean_loss(head_q, head_k, scale, val_features))
    best = {
        "epoch": -1,
        "train_loss": train_start,
        "val_loss": val_start,
        "state": snapshot(),
    }
    history = [{"epoch": -1, "train_loss": train_start, "val_loss": val_start}]
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = _mean_loss(head_q, head_k, scale, train_features)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss = float(loss.detach())
            val_loss = float(_mean_loss(head_q, head_k, scale, val_features))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best["val_loss"]:
            best = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "state": snapshot()}
        print(f"epoch {epoch} train {train_loss:.4f} val {val_loss:.4f}", flush=True)
    head_q.load_state_dict(best["state"][0])
    head_k.load_state_dict(best["state"][1])
    return (head_q, head_k, scale), best, history


def _predict(head, features, labels: list[float]) -> list[dict]:
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
                    "item_id": _item_id(item["source_id"]),
                    "prediction": expected_value(probabilities, labels),
                    "error": None,
                    "schema_valid": True,
                    "latency_ms": item["latency_ms"],
                    "mode": "pointer",
                }
            )
    return rows


def _load_map(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return {row["item_id"]: row for row in read_jsonl(path)}


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _cache_path(root: Path, split: str, source_id: str) -> Path:
    return root / split / source_id.replace("\\", "_").replace("/", "_") / "feature.pt"


def _load_feature(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _ensure_features(model, processor, records, labels, prompt, device, cache_root: Path, split: str) -> list[dict]:
    import torch

    features = []
    for index, record in enumerate(records, start=1):
        path = _cache_path(cache_root, split, record["source_id"])
        if path.is_file():
            features.append(_load_feature(path))
            continue
        if model is None:
            raise RuntimeError(f"missing cache {path} and the model is not loaded")
        feature = _pointer_feature(model, processor, record, labels, prompt, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feature, path)
        features.append(feature)
        print(f"cached {split} {index}/{len(records)} {record['source_id']} {feature['latency_ms']} ms", flush=True)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/warmup.spaq.heldout.json")
    args = parser.parse_args()
    config = read_json(args.config)
    week0 = read_json(ROOT / config["week0"])
    labels = spaq_labels(week0)
    instruction = week0["prompts"]["spaq"]["logits"].strip()
    pointer_prompt = score_user_text(instruction, labels)
    letter_prompt = logits_prompt(week0, "spaq")
    raw = [record for record in collect_sources(load_local_config()) if record["source_dataset"] == "spaq"]
    split_kwargs = {
        "dataset": "spaq",
        "seed": week0["sampling_seed"],
        "train": float(week0["split"]["train"]),
        "validation": float(week0["split"]["validation"]),
    }
    train_raw = select_split_records(raw, split="train", n=int(config["n_train"]), **split_kwargs)
    val_raw = select_split_records(raw, split="validation", n=int(config["n_validation"]), **split_kwargs)
    train_ids = {record["source_id"] for record in train_raw}
    val_ids = {record["source_id"] for record in val_raw}
    if train_ids & val_ids:
        raise SystemExit("train and validation selections overlap")
    train_records = [score_record(record, instruction=instruction, labels=labels) for record in train_raw]
    val_records = [score_record(record, instruction=instruction, labels=labels) for record in val_raw]
    export = ROOT / "data/exports" / config["release_id"]
    write_jsonl(export / "train.jsonl", train_records)
    write_jsonl(export / "validation.jsonl", val_records)

    release = config["release_id"]
    run_dir = ROOT / "outputs/runs"
    cache_root = ROOT / "outputs/warmup" / release / "features"
    logits_path = run_dir / f"{release}-logits.jsonl"
    logits_done = _load_map(logits_path)
    missing_features = []
    for split, records in (("train", train_records), ("validation", val_records)):
        for record in records:
            if not _cache_path(cache_root, split, record["source_id"]).is_file():
                missing_features.append((split, record["source_id"]))
    missing_logits = [record for record in val_records if _item_id(record["source_id"]) not in logits_done]
    print(
        f"missing pointer caches {len(missing_features)} / {len(train_records) + len(val_records)}; "
        f"missing logits {len(missing_logits)} / {len(val_records)}",
        flush=True,
    )

    import torch
    from PIL import Image

    from image_jev.inference import score_logits
    from image_jev.models import load_model

    model = None
    processor = None
    used_pixels = int(config["max_pixels"])
    if missing_features or missing_logits:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available.")
        profile = dict(config)
        try:
            model, processor = load_model(profile)
        except torch.OutOfMemoryError:
            used_pixels = int(config["oom_max_pixels"])
            profile["max_pixels"] = used_pixels
            print(f"out of memory, retrying at max_pixels={used_pixels}", flush=True)
            model, processor = load_model(profile)
        device = torch.device("cuda:0")
        try:
            _ensure_features(model, processor, train_records, labels, pointer_prompt, device, cache_root, "train")
            _ensure_features(model, processor, val_records, labels, pointer_prompt, device, cache_root, "validation")
            for index, record in enumerate(val_records, start=1):
                item_id = _item_id(record["source_id"])
                if item_id in logits_done:
                    continue
                with Image.open(record["image"]) as image:
                    rgb = image.convert("RGB")
                result = score_logits(model, processor, rgb, letter_prompt, labels, enable_thinking=False)
                result["item_id"] = item_id
                _append(logits_path, result)
                logits_done[item_id] = result
                print(f"logits {index}/{len(val_records)} {record['source_id']} {result['latency_ms']} ms", flush=True)
        except torch.OutOfMemoryError:
            raise SystemExit(
                f"out of memory at max_pixels={used_pixels}. Delete partial files under {cache_root} "
                f"and rerun with a smaller max_pixels."
            )
        del model
        torch.cuda.empty_cache()

    train_features = [_load_feature(_cache_path(cache_root, "train", record["source_id"])) for record in train_records]
    val_features = [_load_feature(_cache_path(cache_root, "validation", record["source_id"])) for record in val_records]
    hidden_size = int(train_features[0]["h_decide"].shape[-1])
    head, best, history = _fit(
        train_features,
        val_features,
        hidden_size=hidden_size,
        projection=int(config["projection_dim"]),
        epochs=int(config["epochs"]),
        lr=float(config["head_lr"]),
        seed=int(config["seed"]),
    )
    pointer_val = _predict(head, val_features, labels)
    pointer_train = _predict(head, train_features, labels)
    gold_val = [_gold(record) for record in val_records]
    gold_train = [_gold(record) for record in train_records]
    for item in gold_train:
        item["split"] = "train"
    logits_rows = [_load_map(logits_path)[_item_id(record["source_id"])] for record in val_records]
    pointer_score = summarize(gold_val, pointer_val)
    logits_score = summarize(gold_val, logits_rows)
    paired = compare(gold_val, logits_rows, pointer_val)
    train_metrics = continuous([(item, response["prediction"]) for item, response in zip(gold_train, pointer_train, strict=True)])

    pointer_path = run_dir / f"{release}-pointer.jsonl"
    write_jsonl(pointer_path, pointer_val)
    write_jsonl(logits_path, logits_rows)
    pointer_score_path = run_dir / f"{release}-pointer.score.json"
    logits_score_path = run_dir / f"{release}-logits.score.json"
    compare_path = run_dir / f"{release}.compare.json"
    write_json(pointer_score_path, pointer_score)
    write_json(logits_score_path, logits_score)
    write_json(compare_path, paired)
    head_q, head_k, _scale = head
    head_path = ROOT / "outputs/warmup" / release / "pointer-head.pt"
    head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"q": head_q.state_dict(), "k": head_k.state_dict(), "projection_dim": int(config["projection_dim"]), "best_epoch": best["epoch"]},
        head_path,
    )
    run_meta = {
        "release_id": release,
        "display_name": config["display_name"],
        "model_id": config["model_id"],
        "family": config["family"],
        "quantization": "nf4" if config.get("load_in_4bit") else "none",
        "base_model": config["base_model"],
        "max_pixels": used_pixels,
        "n_train": len(train_records),
        "n_validation": len(val_records),
        "head_lr": float(config["head_lr"]),
        "projection_dim": int(config["projection_dim"]),
        "epochs": int(config["epochs"]),
        "best_epoch": best["epoch"],
        "train_loss_start": history[0]["train_loss"],
        "train_loss_best": best["train_loss"],
        "val_loss_start": history[0]["val_loss"],
        "val_loss_best": best["val_loss"],
        "split_seed": week0["sampling_seed"],
        "pointer_responses": str(pointer_path.relative_to(ROOT)),
        "logits_responses": str(logits_path.relative_to(ROOT)),
        "pointer_score": str(pointer_score_path.relative_to(ROOT)),
        "logits_score": str(logits_score_path.relative_to(ROOT)),
        "compare_path": str(compare_path.relative_to(ROOT)),
        "train_spearman_rho": train_metrics["spearman_rho"],
        "train_mae": train_metrics["mae"],
        "history": [{key: (round(value, 6) if isinstance(value, float) else value) for key, value in row.items()} for row in history],
        "conversation_policy": "stateless_per_item",
    }
    meta_path = run_dir / f"{release}.run.json"
    write_json(meta_path, run_meta)
    report_path = ROOT / "outputs/reports" / f"{release}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_heldout(pointer_score, logits_score, paired, run_meta), encoding="utf-8")
    summary = {
        "best_epoch": best["epoch"],
        "val_loss_best": best["val_loss"],
        "pointer_spearman": (pointer_score["by_source"].get("spaq") or {}).get("spearman_rho"),
        "logits_spearman": (logits_score["by_source"].get("spaq") or {}).get("spearman_rho"),
        "delta": (paired["by_source"].get("spaq") or {}).get("delta_right_minus_left"),
        "report": str(report_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
