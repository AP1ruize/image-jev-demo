#!/usr/bin/env python3
"""Train a language LoRA and the SPAQ pointer head together.

The vision encoder stays frozen. The language model gets the same rank-8
adapter as the JSON QLoRA run, and the pointer head is warm-started from the
frozen-backbone warmup. The loss is the pointer cross-entropy, so inference
stays one forward. The Week 0 test split is scored only after training.
"""
from __future__ import annotations

import argparse
import inspect
import json
import random
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
from image_jev.report import render_pointer_lora  # noqa: E402
from image_jev.resources import cuda_snapshot, latency_summary, reset_cuda_peak  # noqa: E402
from image_jev.scoring import summarize  # noqa: E402
from image_jev.sources import collect_sources  # noqa: E402
from image_jev.warmup_data import score_markers, score_record, score_user_text, select_split_records, spaq_labels  # noqa: E402


LLM_LINEAR = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
}


def _load_base(config: dict):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import Qwen3_5ForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelClass

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    kwargs = {
        "torch_dtype": torch.float16,
        "attn_implementation": config.get("attn_implementation", "eager"),
        "device_map": {"": 0},
        "quantization_config": quant,
    }
    if "attn_implementation" not in inspect.signature(ModelClass.from_pretrained).parameters:
        kwargs.pop("attn_implementation")
    model = ModelClass.from_pretrained(config["base_model"], **kwargs)
    processor = AutoProcessor.from_pretrained(config["base_model"])
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        image_processor.size = {"shortest_edge": int(config["min_pixels"]), "longest_edge": int(config["max_pixels"])}
        if hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = int(config["min_pixels"])
        if hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = int(config["max_pixels"])
    return model, processor


def _lora_targets(model) -> list[str]:
    targets = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        lowered = name.lower()
        if any(token in lowered for token in ("visual", "vision_tower", "vision_model", "vision_encoder")):
            continue
        class_name = module.__class__.__name__
        linear = class_name in {"Linear", "Linear4bit"} or (
            module.__class__.__module__.endswith("bitsandbytes.nn.modules") and class_name.startswith("Linear")
        )
        if linear and leaf in LLM_LINEAR:
            targets.append(name)
    if not targets:
        raise SystemExit("no language LoRA targets found")
    return targets


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


def _encode_marker(processor, marker: str) -> list[int]:
    from image_jev.models import tokenizer_of

    return list(tokenizer_of(processor).encode(marker, add_special_tokens=False))


def _span_end(input_ids: list[int], needle: list[int], marker: str) -> int:
    hits = find_subsequence(input_ids, needle)
    if len(hits) != 1:
        raise RuntimeError(f"marker {marker} found {len(hits)} times")
    return hits[0] + len(needle) - 1


def _splits(config: dict):
    week0 = read_json(ROOT / config["week0"])
    raw = [record for record in collect_sources(load_local_config()) if record["source_dataset"] == "spaq"]
    kwargs = {
        "dataset": "spaq",
        "seed": week0["sampling_seed"],
        "train": float(week0["split"]["train"]),
        "validation": float(week0["split"]["validation"]),
    }
    train_rows = select_split_records(raw, split="train", n=None, **kwargs)
    test_rows = select_split_records(raw, split="test", n=None, allow_test=True, **kwargs)
    if {row["source_id"] for row in train_rows} & {row["source_id"] for row in test_rows}:
        raise SystemExit("train and test overlap")
    labels = spaq_labels(week0)
    instruction = week0["prompts"]["spaq"]["logits"].strip()
    return week0, labels, instruction, train_rows, test_rows


def _examples(records: list[dict], labels: list[float], instruction: str) -> list[dict]:
    examples = []
    for record in records:
        scored = score_record(record, instruction=instruction, labels=labels)
        target = scored["targets"]["mos"]["probabilities"]
        examples.append(
            {
                "source_id": record["source_id"],
                "group_id": record["group_id"],
                "path": record["path"],
                "gold_mos": float(record["labels"]["MOS"]),
                "target": [float(target[format_bin(label)]) for label in labels],
            }
        )
    return examples


def _gold(example: dict) -> dict:
    return {
        "item_id": f"spaq:{example['source_id']}:MOS",
        "source_dataset": "spaq",
        "source_id": example["source_id"],
        "split": "test",
        "group_id": example["group_id"],
        "task": {
            "dimension": "technical_quality",
            "type": "continuous",
            "prompt_id": "week0-iqa-v0.1:spaq",
            "scale": [0, 100],
            "gold_key": "MOS",
        },
        "gold": {"origin": "source_human", "value": float(example["gold_mos"])},
    }


def _hidden_size(model) -> int:
    config = model.config
    if hasattr(config, "text_config") and getattr(config.text_config, "hidden_size", None):
        return int(config.text_config.hidden_size)
    return int(config.hidden_size)


def _build_model(config: dict):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    model, processor = _load_base(config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=_lora_targets(model),
        ),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    hidden = _hidden_size(model)
    projection = int(config["projection_dim"])
    head_q = torch.nn.Linear(hidden, projection, bias=False)
    head_k = torch.nn.Linear(hidden, projection, bias=False)
    init_path = ROOT / config["head_init"]
    if not init_path.is_file():
        raise SystemExit(f"missing pointer head init: {init_path}")
    blob = torch.load(init_path, map_location="cpu", weights_only=False)
    head_q.load_state_dict(blob["q"])
    head_k.load_state_dict(blob["k"])
    device = next(model.parameters()).device
    head_q.to(device=device, dtype=torch.float32)
    head_k.to(device=device, dtype=torch.float32)
    return model, processor, head_q, head_k, projection ** -0.5


def _marker_positions(processor, input_ids: list[int], labels: list[float]) -> list[int]:
    options, decide = score_markers(labels)
    positions = [_span_end(input_ids, _encode_marker(processor, marker), marker) for marker in options]
    positions.append(_span_end(input_ids, _encode_marker(processor, decide), decide))
    return positions


def _install_capture(layer):
    """Read the last language layer outside its checkpoint wrapper.

    A forward hook runs inside gradient checkpointing and sees a tensor with no
    grad. Wrapping the already-checkpointed ``forward`` sees the tensor that
    backward will recompute.
    """
    holder = getattr(layer, "_pointer_capture", None)
    if holder is not None:
        return holder
    holder = {"index": None, "rows": None}
    inner = layer.forward

    def outer(*args, **kwargs):
        output = inner(*args, **kwargs)
        hidden = output[0] if isinstance(output, tuple) else output
        index = holder["index"]
        if index is not None:
            holder["rows"] = hidden[0].index_select(0, index)
        return output

    layer.forward = outer
    layer._pointer_capture = holder
    return holder


def _pointer_logits(model, processor, layer, head_q, head_k, scale, example: dict, labels: list[float], prompt: str, *, train: bool):
    import torch
    from PIL import Image

    from image_jev.inference import _inputs

    device = next(model.parameters()).device
    with Image.open(example["path"]) as image:
        rgb = image.convert("RGB")
    inputs = _inputs(processor, rgb, prompt, False, device)
    positions = _marker_positions(processor, inputs["input_ids"][0].tolist(), labels)
    holder = _install_capture(layer)
    holder["index"] = torch.tensor(positions, device=device)
    holder["rows"] = None
    if train:
        model(**inputs)
    else:
        with torch.no_grad():
            model(**inputs)
    rows = holder["rows"]
    if rows is None:
        raise RuntimeError("language-layer forward did not capture hidden states")
    if train and not rows.requires_grad:
        raise RuntimeError("pointer hidden states do not require grad; LoRA would not learn")
    query = head_q(rows[-1].float())
    keys = head_k(rows[:-1].float())
    return (keys * query).sum(dim=-1) * scale


def _loss_from_logits(logits, target: list[float]):
    import torch

    gold = torch.tensor(target, device=logits.device, dtype=torch.float32)
    return -(gold * torch.log_softmax(logits.float(), dim=0)).sum()


def _save(output_dir: Path, model, processor, head_q, head_k, state: dict, optimizer=None) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    torch.save({"q": head_q.state_dict(), "k": head_k.state_dict()}, output_dir / "pointer-head.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
    write_json(output_dir / "trainer-state.json", state)


def _train(config: dict, examples: list[dict], labels: list[float], prompt: str, output_dir: Path, max_steps: int | None) -> dict:
    import torch
    from peft import PeftModel

    state_path = output_dir / "trainer-state.json"
    resume = state_path.is_file()
    if resume:
        state = read_json(state_path)
        model, processor = _load_base(config)
        model = PeftModel.from_pretrained(model, output_dir, is_trainable=True)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        hidden = _hidden_size(model)
        projection = int(config["projection_dim"])
        head_q = torch.nn.Linear(hidden, projection, bias=False)
        head_k = torch.nn.Linear(hidden, projection, bias=False)
        blob = torch.load(output_dir / "pointer-head.pt", map_location="cpu", weights_only=False)
        head_q.load_state_dict(blob["q"])
        head_k.load_state_dict(blob["k"])
        device = next(model.parameters()).device
        head_q.to(device=device, dtype=torch.float32)
        head_k.to(device=device, dtype=torch.float32)
        scale = projection ** -0.5
    else:
        model, processor, head_q, head_k, scale = _build_model(config)
        order = list(range(len(examples)))
        random.Random(int(config["seed"])).shuffle(order)
        state = {"optimizer_step": 0, "seen": 0, "order": order, "losses": [], "loss_start": None}
    order = state["order"]
    accum = int(config["gradient_accumulation_steps"])
    total_steps = (len(order) + accum - 1) // accum
    limit = total_steps if max_steps is None else min(total_steps, int(max_steps))
    if state["optimizer_step"] >= limit:
        print(f"training already at step {state['optimizer_step']} of {limit}", flush=True)
        return state
    lora_params = [param for param in model.parameters() if param.requires_grad]
    head_lr = float(config.get("head_learning_rate", config["learning_rate"]))
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": float(config["learning_rate"])},
            {"params": [head_q.weight, head_k.weight], "lr": head_lr},
        ]
    )
    params = lora_params + [head_q.weight, head_k.weight]
    optimizer_path = output_dir / "optimizer.pt"
    if resume and optimizer_path.is_file():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=False))
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    layer = _last_language_layer(model)
    model.train()
    head_q.train()
    head_k.train()
    optimizer.zero_grad(set_to_none=True)
    micro_losses = []
    print(f"cuda before step: {cuda_snapshot()}", flush=True)
    for example_index in range(state["seen"], len(order)):
        example = examples[order[example_index]]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = _pointer_logits(
                model, processor, layer, head_q, head_k, scale, example, labels, prompt, train=True
            )
        loss = _loss_from_logits(logits, example["target"])
        if state["loss_start"] is None:
            state["loss_start"] = float(loss.detach())
            print(f"loss_start {state['loss_start']:.4f}", flush=True)
        scaler.scale(loss / accum).backward()
        micro_losses.append(float(loss.detach()))
        finished_window = len(micro_losses) == accum or example_index + 1 == len(order)
        if not finished_window:
            continue
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, float(config["max_grad_norm"])))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        state["optimizer_step"] += 1
        state["seen"] = example_index + 1
        mean_loss = sum(micro_losses) / len(micro_losses)
        micro_losses = []
        state["losses"].append({"step": state["optimizer_step"], "loss": round(mean_loss, 6), "grad_norm": round(grad_norm, 4)})
        if state["optimizer_step"] % int(config["logging_steps"]) == 0 or state["optimizer_step"] <= 5 or state["optimizer_step"] == limit:
            recent = state["losses"][-20:]
            running = sum(item["loss"] for item in recent) / len(recent)
            print(
                f"step {state['optimizer_step']}/{limit} loss {mean_loss:.4f} recent {running:.4f} grad {grad_norm:.3f}",
                flush=True,
            )
        if state["optimizer_step"] % int(config["save_steps"]) == 0 or state["optimizer_step"] == limit:
            _save(output_dir, model, processor, head_q, head_k, state, optimizer)
        if state["optimizer_step"] >= limit:
            break
    state["loss_end"] = state["losses"][-1]["loss"] if state["losses"] else state["loss_start"]
    _save(output_dir, model, processor, head_q, head_k, state, optimizer)
    del model, head_q, head_k
    torch.cuda.empty_cache()
    return state


def _evaluate(config: dict, examples: list[dict], labels: list[float], prompt: str, output_dir: Path) -> tuple[list[dict], dict]:
    import torch
    from peft import PeftModel

    model, processor = _load_base(config)
    model = PeftModel.from_pretrained(model, output_dir)
    model.eval()
    model.config.use_cache = False
    hidden = _hidden_size(model)
    projection = int(config["projection_dim"])
    head_q = torch.nn.Linear(hidden, projection, bias=False)
    head_k = torch.nn.Linear(hidden, projection, bias=False)
    blob = torch.load(output_dir / "pointer-head.pt", map_location="cpu", weights_only=False)
    head_q.load_state_dict(blob["q"])
    head_k.load_state_dict(blob["k"])
    device = next(model.parameters()).device
    head_q.to(device=device, dtype=torch.float32).eval()
    head_k.to(device=device, dtype=torch.float32).eval()
    scale = projection ** -0.5
    layer = _last_language_layer(model)
    resident = cuda_snapshot()
    reset_cuda_peak()
    rows = []
    for index, example in enumerate(examples, start=1):
        started = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = _pointer_logits(
                model, processor, layer, head_q, head_k, scale, example, labels, prompt, train=False
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        probabilities = torch.softmax(logits.float(), dim=0).tolist()
        errors = distribution_errors(probabilities)
        rows.append(
            {
                "item_id": f"spaq:{example['source_id']}:MOS",
                "prediction": None if errors else expected_value(probabilities, labels),
                "error": None if not errors else ",".join(errors),
                "schema_valid": not errors,
                "latency_ms": latency_ms,
                "mode": "pointer-lora",
            }
        )
        if index == 1 or index % 50 == 0 or index == len(examples):
            print(f"score {index}/{len(examples)} {example['source_id']} {latency_ms} ms", flush=True)
    resources = {"resident_after_load": resident, **latency_summary(row["latency_ms"] for row in rows), **cuda_snapshot()}
    return rows, resources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pointer-lora.spaq.qwen35-2b.json")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    _week0, labels, instruction, train_raw, test_raw = _splits(config)
    prompt = score_user_text(instruction, labels)
    train_examples = _examples(train_raw, labels, instruction)
    test_examples = _examples(test_raw, labels, instruction)
    output_dir = ROOT / "outputs/qlora" / config["release_id"]
    if args.max_steps is not None and not args.eval_only:
        output_dir = output_dir.with_name(output_dir.name + "-smoke")
    print(f"train={len(train_examples)} test={len(test_examples)} output={output_dir}", flush=True)
    state = {"loss_start": None, "loss_end": None, "optimizer_step": None}
    if not args.eval_only:
        state = _train(config, train_examples, labels, prompt, output_dir, args.max_steps)
    if args.skip_eval:
        print(json.dumps({"loss_start": state.get("loss_start"), "loss_end": state.get("loss_end"), "step": state.get("optimizer_step")}), flush=True)
        return
    responses, resources = _evaluate(config, test_examples, labels, prompt, output_dir)
    gold = [_gold(example) for example in test_examples]
    score = summarize(gold, responses)
    release = config["release_id"] if args.max_steps is None else config["release_id"] + "-smoke"
    run_dir = ROOT / "outputs/runs"
    response_path = run_dir / f"{release}.jsonl"
    score_path = run_dir / f"{release}.score.json"
    write_jsonl(response_path, responses)
    write_json(score_path, score)
    run_meta = {
        "release_id": release,
        "display_name": config["display_name"],
        "model_id": config["model_id"],
        "family": config["family"],
        "quantization": "nf4",
        "base_model": config["base_model"],
        "epochs": config["num_train_epochs"],
        "lora_r": config["lora_r"],
        "max_pixels": config["max_pixels"],
        "n_train": len(train_examples),
        "n_test": len(test_examples),
        "head_init": config["head_init"],
        "loss_start": state.get("loss_start"),
        "loss_end": state.get("loss_end"),
        "optimizer_step": state.get("optimizer_step"),
        "responses_path": str(response_path.relative_to(ROOT)),
        "score_path": str(score_path.relative_to(ROOT)),
        "inference_resources": resources,
    }
    write_json(run_dir / f"{release}.run.json", run_meta)
    frozen_meta = read_json(run_dir / "spaq-pointer-full-v0.1.run.json")
    frozen_score = read_json(run_dir / "spaq-pointer-full-v0.1-pointer.score.json")
    qlora_meta = read_json(run_dir / "spaq-qwen35-2b-4bit-full-v0.1.run.json")
    qlora_score = read_json(run_dir / "spaq-qwen35-2b-4bit-full-v0.1.score.json")
    versus_frozen = compare(gold, read_jsonl(run_dir / "spaq-pointer-full-v0.1-pointer.jsonl"), responses)
    versus_qlora = compare(gold, read_jsonl(run_dir / "spaq-qwen35-2b-4bit-full-v0.1.jsonl"), responses)
    report = render_pointer_lora(
        frozen_meta, frozen_score, qlora_meta, qlora_score, run_meta, score, versus_frozen, versus_qlora
    )
    report_path = ROOT / "outputs/reports" / f"{release}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    write_json(run_dir / f"{release}.compare.json", {"versus_frozen": versus_frozen, "versus_qlora": versus_qlora})
    spaq = score["by_source"]["spaq"]
    print(
        json.dumps(
            {
                "spearman": spaq.get("spearman_rho"),
                "mae": spaq.get("mae"),
                "delta_vs_frozen": versus_frozen["by_source"]["spaq"]["delta_right_minus_left"],
                "delta_vs_qlora": versus_qlora["by_source"]["spaq"]["delta_right_minus_left"],
                "median_latency_ms": resources.get("median_latency_ms"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
