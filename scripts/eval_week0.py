#!/usr/bin/env python3
"""Run generate / grammar / logits inference on the frozen Week 0 IQA split."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.io import read_json, read_jsonl, recover_jsonl, sha256_file, write_json  # noqa: E402
from image_jev.runtime import apply_runtime_env, load_local_config  # noqa: E402

apply_runtime_env(load_local_config())

from PIL import Image  # noqa: E402

from image_jev.inference import prepare_grammar_tables, run_item  # noqa: E402
from image_jev.models import load_model  # noqa: E402
from image_jev.scoring import summarize  # noqa: E402
from image_jev.report import render_score  # noqa: E402


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def spec_from_release(week0: dict, release: dict) -> dict:
    merged = dict(week0)
    merged["prompts"] = release.get("prompts") or week0["prompts"]
    merged["datasets"] = release.get("datasets") or week0["datasets"]
    merged["release_id"] = release["release_id"]
    return merged


def maybe_score(release_dir: Path, output: Path, run_meta: dict, official: bool) -> None:
    gold_path = release_dir / "benchmark-gold.jsonl"
    gold = read_jsonl(gold_path)
    responses = read_jsonl(output)
    if not official:
        gold = [item for item in gold if item["item_id"] in {row["item_id"] for row in responses}]
    score = summarize(gold, responses)
    score["release"] = run_meta.get("release_id")
    output = output.resolve()
    score_path = output.with_suffix(".score.json")
    report_path = (ROOT / "outputs/reports" / f"{output.stem}.md").resolve()
    write_json(score_path, score)
    run_meta = dict(run_meta)
    try:
        run_meta["score_path"] = str(score_path.relative_to(ROOT.resolve()))
    except ValueError:
        run_meta["score_path"] = str(score_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_score(score, run_meta), encoding="utf-8")
    print(f"scored {output} -> {score_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Model profile JSON")
    parser.add_argument("--week0", type=Path, default=ROOT / "configs/week0.v0.1.json")
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--mode", choices=["generate", "grammar", "logits"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-model", type=Path, help="Optional local HF snapshot")
    args = parser.parse_args()

    week0 = read_json(args.week0)
    release_dir = args.release_dir or (ROOT / "data/processed" / week0["release_id"])
    release = read_json(release_dir / "release.json")
    run_items_path = ROOT / release["artifacts"]["benchmark_run_items"]["path"]
    if sha256_file(run_items_path) != release["artifacts"]["benchmark_run_items"]["sha256"]:
        raise SystemExit("run-item manifest hash differs from frozen release")
    items = read_jsonl(run_items_path)
    if args.max_items is not None:
        items = items[: args.max_items]
    spec = spec_from_release(week0, release)
    model_cfg = read_json(args.config)
    output = (args.output or (
        ROOT / "outputs/runs" / f"{model_cfg['model_id']}-{args.mode}-{release['release_id']}.jsonl"
    )).resolve()

    if args.dry_run:
        for item in items:
            path = ROOT / item["assets"][0]["path"]
            if not path.is_file() or sha256_file(path) != item["assets"][0]["sha256"]:
                raise SystemExit(f"bad asset for {item['item_id']}: {path}")
        print(f"offline dry run passed: {len(items)} items")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".run.json")
    completed = []
    if output.exists() and not args.resume:
        raise SystemExit(f"output already exists: {output}; use a fresh path or --resume")
    if args.resume and output.exists():
        completed = recover_jsonl(output)
        print(f"resuming {output}: {len(completed)} complete records kept", flush=True)
    else:
        write_json(
            metadata_path,
            {
                "release_id": release["release_id"],
                "release_sha256": sha256_file(release_dir / "release.json"),
                "run_items_sha256": sha256_file(run_items_path),
                "model_id": model_cfg["model_id"],
                "display_name": model_cfg.get("display_name"),
                "family": model_cfg["family"],
                "base_model": str(args.local_model or model_cfg["base_model"]),
                "mode": args.mode,
                "week0_config": args.week0.as_posix(),
                "model_config": args.config.as_posix(),
                "started_utc_epoch": time.time(),
                "items_requested": len(items),
            },
        )
    done = {record["item_id"] for record in completed}
    pending = [item for item in items if item["item_id"] not in done]
    if not pending:
        maybe_score(release_dir, output, read_json(metadata_path), official=args.max_items is None)
        print(f"nothing to run; {len(done)}/{len(items)} already complete")
        return

    model, processor = load_model(model_cfg, local_dir=args.local_model)
    eval_cfg = model_cfg.get("evaluation") or {}
    max_new = int(
        eval_cfg["max_new_tokens_grammar"]
        if args.mode == "grammar"
        else eval_cfg.get("max_new_tokens_generate", 32)
    )
    grammar_tables = prepare_grammar_tables(processor) if args.mode == "grammar" else None
    enable_thinking = bool(model_cfg.get("enable_thinking", False))

    with output.open("a", encoding="utf-8", newline="\n") as handle:
        for offset, item in enumerate(pending, 1):
            path = ROOT / item["assets"][0]["path"]
            if sha256_file(path) != item["assets"][0]["sha256"]:
                raise SystemExit(f"asset changed before inference: {path}")
            try:
                record = run_item(
                    model,
                    processor,
                    load_image(path),
                    item,
                    spec,
                    args.mode,
                    max_new_tokens=max_new,
                    enable_thinking=enable_thinking,
                    grammar_tables=grammar_tables,
                )
            except Exception as exc:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                record = {
                    "item_id": item["item_id"],
                    "mode": args.mode,
                    "prediction": None,
                    "error": f"request_error:{type(exc).__name__}:{exc}",
                    "schema_valid": False,
                    "raw_response": None,
                    "latency_ms": None,
                    "n_new_tokens": None,
                    "n_forward_passes": None,
                    "bin_label": None,
                    "bin_probability": None,
                    "probabilities": None,
                }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            finished = len(done) + offset
            if finished % 10 == 0 or offset == 1 or offset == len(pending):
                print(f"{finished}/{len(items)} items", flush=True)
    print(f"completed {len(items)} requests -> {output}")
    maybe_score(release_dir, output, read_json(metadata_path), official=args.max_items is None)


if __name__ == "__main__":
    main()
