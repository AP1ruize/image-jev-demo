#!/usr/bin/env python3
"""Train a language LoRA and the SPAQ pointer head together.

The vision encoder stays frozen. The language model gets the same rank-8
adapter as the JSON QLoRA run, and the pointer head is warm-started from the
frozen-backbone warmup. The loss is the pointer cross-entropy, so inference
stays one forward.

When the config sets select_by_validation, checkpoints are ranked by that same
cross-entropy on a fixed validation subset. The chosen checkpoint is confirmed
on the full validation split and only then scored on the Week 0 test split.
"""
from __future__ import annotations

import argparse
import inspect
import json
import random
import subprocess
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
from image_jev.io import read_json, read_jsonl, recover_jsonl, write_json, write_jsonl  # noqa: E402
from image_jev.report import render_pointer_lora, render_pointer_lora_valselect  # noqa: E402
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
    val_rows = select_split_records(raw, split="validation", n=None, **kwargs)
    test_rows = select_split_records(raw, split="test", n=None, allow_test=True, **kwargs)
    train_ids = {row["source_id"] for row in train_rows}
    val_ids = {row["source_id"] for row in val_rows}
    test_ids = {row["source_id"] for row in test_rows}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise SystemExit("train, validation, and test overlap")
    labels = spaq_labels(week0)
    instruction = week0["prompts"]["spaq"]["logits"].strip()
    return week0, labels, instruction, train_rows, val_rows, test_rows


def _check_validation_export(config: dict, val_rows: list[dict]) -> None:
    path = ROOT / config["val_export"]
    exported = read_jsonl(path)
    export_ids = [row["source_id"] for row in exported]
    split_ids = [row["source_id"] for row in val_rows]
    if len(export_ids) != len(set(export_ids)) or set(export_ids) != set(split_ids):
        raise SystemExit(f"validation export {path} does not match select_split_records")
    if len(val_rows) != 1660:
        raise SystemExit(f"expected 1660 validation rows, found {len(val_rows)}")


def _assert_same_training_order(order: list[int]) -> None:
    if not PREVIOUS_ORDER.is_file():
        return
    previous = read_json(PREVIOUS_ORDER).get("order")
    if previous != order:
        raise SystemExit("training order does not match spaq-pointer-lora-v0.1; refusing to start")


def _output_dir(config: dict, max_steps: int | None, eval_only: bool) -> Path:
    if config.get("output_dir"):
        output_dir = ROOT / config["output_dir"]
    else:
        output_dir = ROOT / "outputs/qlora" / config["release_id"]
    if max_steps is not None and not eval_only:
        output_dir = output_dir.with_name(output_dir.name + "-smoke")
    return output_dir


def _assert_gpu_idle(limit_mb: float = 1024) -> None:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    used_text, util_text = [part.strip() for part in text.strip().splitlines()[0].split(",")]
    used = float(used_text)
    print(f"gpu before work: used {used:.0f} MB, util {util_text}%", flush=True)
    if used > limit_mb:
        raise SystemExit(f"GPU is not idle: {used:.0f} MB already used")


def _assert_visual_frozen(model) -> None:
    visual = [name for name, param in model.named_parameters() if param.requires_grad and "visual" in name.lower()]
    if visual:
        raise RuntimeError("visual parameters are trainable: " + ", ".join(visual[:5]))


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


PROTECTED_CHECKPOINT = ROOT / "outputs/qlora/spaq-pointer-lora-v0.1"
PREVIOUS_ORDER = PROTECTED_CHECKPOINT / "trainer-state.json"
MERGER_HEADROOM_MB = 512.0
LAST_STEP_RELEASE = "spaq-pointer-lora-v0.1"
QLORA_RELEASE = "spaq-qwen35-2b-4bit-full-v0.1"


def monitor_indices(n: int, size: int | None, seed: int) -> list[int]:
    """Fixed-seed validation subset. The training shuffle uses its own Random."""
    if n <= 0:
        raise ValueError("validation set is empty")
    if size is None or int(size) >= n:
        return list(range(n))
    size = int(size)
    if size <= 0:
        raise ValueError("val monitor size must be positive")
    rng = random.Random(int(seed))
    picked = rng.sample(range(n), size)
    picked.sort()
    return picked


def keeps_lower_validation_loss(best: dict | None, loss: float) -> bool:
    if best is None:
        return True
    return float(loss) < float(best["monitor_loss"])


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


def _read_state(path: Path) -> dict | None:
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _nvidia_memory() -> tuple[float | None, float | None]:
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    used_text, total_text = [part.strip() for part in text.strip().splitlines()[0].split(",")]
    return float(used_text), float(total_text)


def _fp16_linear_from_4bit(module):
    """NF4 Linear4bit weights cannot require grad. Materialize a trainable fp16 copy."""
    import torch
    import bitsandbytes.functional as bnb_functional

    weight = bnb_functional.dequantize_4bit(module.weight.data, module.weight.quant_state).to(dtype=torch.float16)
    linear = torch.nn.Linear(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        device=weight.device,
        dtype=torch.float16,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
        if module.bias is not None:
            linear.bias.copy_(module.bias.detach().to(dtype=torch.float16))
    linear.requires_grad_(True)
    return linear


def _enable_merger_training(mergers) -> str:
    import torch
    from bitsandbytes.nn import Linear4bit

    replaced = False
    for _, merger in mergers:
        for child_name, child in list(merger.named_modules()):
            if child_name == "":
                continue
            if isinstance(child, Linear4bit):
                parent = merger
                parts = child_name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], _fp16_linear_from_4bit(child))
                replaced = True
        for param in merger.parameters():
            if not (param.dtype.is_floating_point or param.dtype.is_complex):
                raise RuntimeError(f"merger parameter {tuple(param.shape)} dtype {param.dtype} cannot require grad")
            param.requires_grad = True
    return "nf4-dequantized-to-fp16" if replaced else "floating"


def _vision_mergers(model) -> list[tuple[str, object]]:
    found = []
    seen = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Qwen3_5VisionPatchMerger":
            continue
        if id(module) in seen:
            continue
        seen.add(id(module))
        found.append((name, module))
    if not found:
        raise RuntimeError("visual merger not found")
    return found


def _save(output_dir: Path, model, processor, head_q, head_k, state: dict, optimizer=None) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    torch.save({"q": head_q.state_dict(), "k": head_k.state_dict()}, output_dir / "pointer-head.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
    write_json(output_dir / "trainer-state.json", state)


def _mean_pointer_ce(model, processor, layer, head_q, head_k, scale, examples, labels, prompt, *, tag: str) -> float:
    """Mean pointer cross-entropy. The forward does not update weights."""
    import torch

    if not examples:
        raise ValueError("no examples to score")
    training = model.training
    model.eval()
    head_q.eval()
    head_k.eval()
    total = 0.0
    try:
        with torch.no_grad():
            for index, example in enumerate(examples, start=1):
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = _pointer_logits(
                        model, processor, layer, head_q, head_k, scale, example, labels, prompt, train=False
                    )
                total += float(_loss_from_logits(logits, example["target"]).detach())
                if index == 1 or index % 50 == 0 or index == len(examples):
                    print(f"{tag} {index}/{len(examples)} running {total / index:.4f}", flush=True)
    finally:
        if training:
            model.train()
            head_q.train()
            head_k.train()
    return total / len(examples)


def _load_trainable(config: dict, directory: Path):
    import torch
    from peft import PeftModel

    model, processor = _load_base(config)
    model = PeftModel.from_pretrained(model, directory, is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    hidden = _hidden_size(model)
    projection = int(config["projection_dim"])
    head_q = torch.nn.Linear(hidden, projection, bias=False)
    head_k = torch.nn.Linear(hidden, projection, bias=False)
    blob = torch.load(directory / "pointer-head.pt", map_location="cpu", weights_only=False)
    head_q.load_state_dict(blob["q"])
    head_k.load_state_dict(blob["k"])
    device = next(model.parameters()).device
    head_q.to(device=device, dtype=torch.float32)
    head_k.to(device=device, dtype=torch.float32)
    _assert_visual_frozen(model)
    return model, processor, head_q, head_k, projection ** -0.5


def _try_load_trainable(config: dict, directory: Path):
    import gc
    import torch

    try:
        return _load_trainable(config, directory)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"checkpoint load failed at {directory}: {exc}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def _rewind_to_best(state: dict) -> dict:
    best = state.get("best")
    if not best:
        raise SystemExit("weights lost and no best validation checkpoint is recorded")
    step = int(best["step"])
    state["optimizer_step"] = step
    state["seen"] = int(best["seen"])
    state["losses"] = [item for item in state.get("losses") or [] if int(item["step"]) <= step]
    state["val_monitor"] = [item for item in state.get("val_monitor") or [] if int(item["step"]) <= step]
    state["pending_val_step"] = None
    state["loss_end"] = state["losses"][-1]["loss"] if state["losses"] else state.get("loss_start")
    print(f"resuming from best validation checkpoint at step {step}", flush=True)
    return state


def _val_due(step: int, limit: int, every: int) -> bool:
    return step > 0 and (step % every == 0 or step == limit)


def _monitor_recorded(state: dict, step: int) -> bool:
    return any(int(item["step"]) == int(step) for item in state.get("val_monitor") or [])


def _remember_best(state: dict, entry: dict, best_dir: Path, model, processor, head_q, head_k, optimizer) -> None:
    if not keeps_lower_validation_loss(state.get("best"), entry["loss"]):
        return
    state["best"] = {
        "step": entry["step"],
        "monitor_loss": entry["loss"],
        "train_loss": entry["train_loss"],
        "n": entry["n"],
        "seen": entry["seen"],
    }
    _save(best_dir, model, processor, head_q, head_k, state, optimizer)
    write_json(best_dir / "selection.json", state["best"])
    print(f"saved best checkpoint at step {entry['step']} val {entry['loss']:.4f}", flush=True)


def _record_monitor(state, model, processor, layer, head_q, head_k, scale, monitor, labels, prompt, train_loss, best_dir, optimizer) -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    started = time.perf_counter()
    loss = _mean_pointer_ce(
        model,
        processor,
        layer,
        head_q,
        head_k,
        scale,
        monitor,
        labels,
        prompt,
        tag=f"val-monitor step {state['optimizer_step']}",
    )
    entry = {
        "step": state["optimizer_step"],
        "loss": round(float(loss), 6),
        "train_loss": round(float(train_loss), 6),
        "n": len(monitor),
        "seen": state["seen"],
        "seconds": round(time.perf_counter() - started, 1),
    }
    state.setdefault("val_monitor", []).append(entry)
    print(
        f"val step {entry['step']} loss {entry['loss']:.4f} train {entry['train_loss']:.4f} n {entry['n']} ({entry['seconds']:.0f}s)",
        flush=True,
    )
    _remember_best(state, entry, best_dir, model, processor, head_q, head_k, optimizer)


def _load_optimizer(optimizer, path: Path) -> None:
    import torch

    if not path.is_file():
        if optimizer is not None:
            print(f"optimizer state missing at {path}; Adam moments start over", flush=True)
        return
    try:
        optimizer.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"optimizer state unreadable ({exc}); Adam moments start over", flush=True)


def _train(
    config: dict,
    examples: list[dict],
    labels: list[float],
    prompt: str,
    output_dir: Path,
    max_steps: int | None,
    val_examples: list[dict] | None = None,
) -> dict:
    import gc
    import torch

    selecting = val_examples is not None
    best_dir = output_dir / "best"
    output_dir.mkdir(parents=True, exist_ok=True)
    rolling_weights = (output_dir / "adapter_model.safetensors").is_file() and (output_dir / "pointer-head.pt").is_file()
    best_weights = (best_dir / "adapter_model.safetensors").is_file() and (best_dir / "pointer-head.pt").is_file()
    state = _read_state(output_dir / "trainer-state.json")
    if state is None and rolling_weights:
        raise SystemExit(f"trainer-state.json is unreadable while {output_dir} has weights; refusing to restart")
    loaded = None
    rewound = False
    if state is not None and rolling_weights:
        loaded = _try_load_trainable(config, output_dir)
    if loaded is None and selecting and (state is not None or best_weights):
        if state is None:
            state = _read_state(best_dir / "trainer-state.json")
        if state is None or not best_weights:
            raise SystemExit(f"cannot resume {output_dir}; the best validation checkpoint is missing too")
        state = _rewind_to_best(state)
        loaded = _try_load_trainable(config, best_dir)
        rewound = True
        if loaded is None:
            raise SystemExit("best validation checkpoint could not be loaded")
    model = head_q = head_k = None
    try:
        if loaded is None:
            model, processor, head_q, head_k, scale = _build_model(config)
            _assert_visual_frozen(model)
            order = list(range(len(examples)))
            random.Random(int(config["seed"])).shuffle(order)
            state = {"optimizer_step": 0, "seen": 0, "order": order, "losses": [], "loss_start": None}
        else:
            model, processor, head_q, head_k, scale = loaded
            loaded = None
        if selecting:
            state.setdefault("val_monitor", [])
            state.setdefault("best", None)
            state.setdefault("pending_val_step", None)
            state.setdefault("val_full", None)
            if "monitor_indices" not in state:
                state["monitor_indices"] = monitor_indices(
                    len(val_examples),
                    config.get("val_monitor_size"),
                    int(config.get("val_monitor_seed", config["seed"])),
                )
                state["monitor_seed"] = int(config.get("val_monitor_seed", config["seed"]))
                state["monitor_size"] = len(state["monitor_indices"])
            monitor = [val_examples[index] for index in state["monitor_indices"]]
            _assert_same_training_order(state["order"])
            eval_every = int(config.get("eval_steps", config["save_steps"]))
        else:
            monitor = None
            eval_every = None
        order = state["order"]
        print(f"train order head: {order[:5]}", flush=True)
        accum = int(config["gradient_accumulation_steps"])
        total_steps = (len(order) + accum - 1) // accum
        limit = total_steps if max_steps is None else min(total_steps, int(max_steps))
        layer = _last_language_layer(model)

        def resolve_pending(optimizer) -> None:
            pending = state.get("pending_val_step")
            if not selecting or not pending or int(pending) != int(state["optimizer_step"]):
                return
            if not _monitor_recorded(state, int(pending)):
                train_loss = state["losses"][-1]["loss"]
                _record_monitor(
                    state, model, processor, layer, head_q, head_k, scale, monitor, labels, prompt, train_loss, best_dir, optimizer
                )
            state["pending_val_step"] = None
            write_json(output_dir / "trainer-state.json", state)

        finished = state["optimizer_step"] >= limit
        pending_now = bool(selecting and state.get("pending_val_step") == state["optimizer_step"] and not _monitor_recorded(state, state["optimizer_step"]))
        optimizer = None
        if not finished or pending_now:
            lora_params = [param for param in model.parameters() if param.requires_grad]
            head_lr = float(config.get("head_learning_rate", config["learning_rate"]))
            optimizer = torch.optim.AdamW(
                [
                    {"params": lora_params, "lr": float(config["learning_rate"])},
                    {"params": [head_q.weight, head_k.weight], "lr": head_lr},
                ]
            )
            params = lora_params + [head_q.weight, head_k.weight]
            if state["optimizer_step"] > 0:
                _load_optimizer(optimizer, (best_dir if rewound else output_dir) / "optimizer.pt")
            if pending_now:
                model.train()
                head_q.train()
                head_k.train()
                resolve_pending(optimizer)
            if state["optimizer_step"] >= limit:
                state["loss_end"] = state["losses"][-1]["loss"] if state.get("losses") else state.get("loss_start")
                write_json(output_dir / "trainer-state.json", state)
                print(f"training already at step {state['optimizer_step']} of {limit}", flush=True)
                return state
        else:
            state["loss_end"] = state["losses"][-1]["loss"] if state.get("losses") else state.get("loss_start")
            print(f"training already at step {state['optimizer_step']} of {limit}", flush=True)
            return state
        scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
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
            state["losses"].append(
                {"step": state["optimizer_step"], "loss": round(mean_loss, 6), "grad_norm": round(grad_norm, 4)}
            )
            if state["optimizer_step"] % int(config["logging_steps"]) == 0 or state["optimizer_step"] <= 5 or state["optimizer_step"] == limit:
                recent = state["losses"][-20:]
                running = sum(item["loss"] for item in recent) / len(recent)
                print(
                    f"step {state['optimizer_step']}/{limit} loss {mean_loss:.4f} recent {running:.4f} grad {grad_norm:.3f}",
                    flush=True,
                )
            if selecting and _val_due(state["optimizer_step"], limit, eval_every):
                state["pending_val_step"] = state["optimizer_step"]
                _save(output_dir, model, processor, head_q, head_k, state, optimizer)
                resolve_pending(optimizer)
            elif not selecting and (
                state["optimizer_step"] % int(config["save_steps"]) == 0 or state["optimizer_step"] == limit
            ):
                _save(output_dir, model, processor, head_q, head_k, state, optimizer)
            if state["optimizer_step"] >= limit:
                break
        state["loss_end"] = state["losses"][-1]["loss"] if state["losses"] else state["loss_start"]
        if selecting:
            write_json(output_dir / "trainer-state.json", state)
        else:
            _save(output_dir, model, processor, head_q, head_k, state, optimizer)
        return state
    finally:
        loaded = None
        del model, head_q, head_k
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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


def _load_for_eval(config: dict, checkpoint: Path):
    import torch
    from peft import PeftModel

    model, processor = _load_base(config)
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    model.config.use_cache = False
    hidden = _hidden_size(model)
    projection = int(config["projection_dim"])
    head_q = torch.nn.Linear(hidden, projection, bias=False)
    head_k = torch.nn.Linear(hidden, projection, bias=False)
    blob = torch.load(checkpoint / "pointer-head.pt", map_location="cpu", weights_only=False)
    head_q.load_state_dict(blob["q"])
    head_k.load_state_dict(blob["k"])
    device = next(model.parameters()).device
    head_q.to(device=device, dtype=torch.float32).eval()
    head_k.to(device=device, dtype=torch.float32).eval()
    return model, processor, head_q, head_k, projection ** -0.5, _last_language_layer(model), cuda_snapshot()


def _score_loaded(model, processor, layer, head_q, head_k, scale, examples, labels, prompt, *, resident, stream_path: Path | None = None):
    import torch

    done = recover_jsonl(stream_path) if stream_path is not None and stream_path.is_file() else []
    done_ids = {row["item_id"] for row in done}
    handle = None
    fresh = []
    try:
        if stream_path is not None:
            stream_path.parent.mkdir(parents=True, exist_ok=True)
            handle = stream_path.open("a", encoding="utf-8", newline="\n")
        reset_cuda_peak()
        pending = [example for example in examples if f"spaq:{example['source_id']}:MOS" not in done_ids]
        finished = len(done_ids)
        total = len(examples)
        for example in pending:
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
            row = {
                "item_id": f"spaq:{example['source_id']}:MOS",
                "prediction": None if errors else expected_value(probabilities, labels),
                "error": None if not errors else ",".join(errors),
                "schema_valid": not errors,
                "latency_ms": latency_ms,
                "mode": "pointer-lora",
            }
            fresh.append(row)
            finished += 1
            if handle is not None:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
            if finished == 1 or finished % 50 == 0 or finished == total:
                print(f"score {finished}/{total} {example['source_id']} {latency_ms} ms", flush=True)
    finally:
        if handle is not None:
            handle.close()
    by_id = {row["item_id"]: row for row in done}
    by_id.update({row["item_id"]: row for row in fresh})
    ordered = []
    for example in examples:
        item_id = f"spaq:{example['source_id']}:MOS"
        if item_id in by_id:
            ordered.append(by_id[item_id])
    resources = {
        "resident_after_load": resident,
        **latency_summary(row["latency_ms"] for row in ordered),
        **cuda_snapshot(),
    }
    return ordered, resources


def _probe_visual_merger(config: dict, example: dict, labels: list[float], prompt: str) -> dict:
    import gc
    import torch

    result = {
        "kind": "visual-merger-one-image",
        "images": 1,
        "source_id": example["source_id"],
        "gradient_checkpointing": True,
        "use_reentrant": False,
        "quantization": "nf4",
        "oom": False,
        "grad_ok": False,
        "fits": False,
        "headroom_threshold_mb": MERGER_HEADROOM_MB,
    }
    model = head_q = head_k = None
    try:
        _assert_gpu_idle()
        model, processor, head_q, head_k, scale = _build_model(config)
        _assert_visual_frozen(model)
        mergers = _vision_mergers(model)
        names = [name for name, _module in mergers]
        storage = _enable_merger_training(mergers)
        merger_ids = {id(param) for _, module in mergers for param in module.parameters()}
        for name, param in model.named_parameters():
            if "visual" in name.lower() and id(param) not in merger_ids:
                param.requires_grad = False
        still_frozen = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and "visual" in name.lower() and id(param) not in merger_ids
        ]
        if still_frozen:
            raise RuntimeError("non-merger visual parameters became trainable")
        result["modules"] = names
        result["n_parameters"] = sum(param.numel() for _, module in mergers for param in module.parameters())
        result["merger_storage"] = storage
        layer = _last_language_layer(model)
        model.train()
        head_q.train()
        head_k.train()
        reset_cuda_peak()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = _pointer_logits(
                model, processor, layer, head_q, head_k, scale, example, labels, prompt, train=True
            )
        loss = _loss_from_logits(logits, example["target"])
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        grad_ok = True
        for _, module in mergers:
            grads = [param.grad for param in module.parameters() if param.requires_grad]
            if not grads or any(item is None for item in grads):
                grad_ok = False
        result["grad_ok"] = grad_ok
        result["loss"] = round(float(loss.detach()), 6)
        snap = cuda_snapshot()
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        nvidia_used, nvidia_total = _nvidia_memory()
        peak = max(float(snap.get("gpu_max_allocated_mb") or 0), float(snap.get("gpu_max_reserved_mb") or 0))
        headroom_torch = float(total_mb) - peak
        headroom_nvidia = None if nvidia_used is None or nvidia_total is None else float(nvidia_total) - float(nvidia_used)
        headroom = headroom_torch if headroom_nvidia is None else min(headroom_torch, headroom_nvidia)
        result.update(
            {
                "gpu_max_allocated_mb": snap.get("gpu_max_allocated_mb"),
                "gpu_max_reserved_mb": snap.get("gpu_max_reserved_mb"),
                "gpu_name": snap.get("gpu_name"),
                "gpu_total_mb": round(total_mb, 2),
                "nvidia_used_mb": nvidia_used,
                "nvidia_total_mb": nvidia_total,
                "headroom_mb": round(headroom, 2),
                "fits": bool(grad_ok and headroom >= MERGER_HEADROOM_MB),
            }
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        result["oom"] = True
        result["fits"] = False
        result["error"] = str(exc).splitlines()[-1][:500]
        print(f"merger probe OOM: {result['error']}", flush=True)
    finally:
        del model, head_q, head_k
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _wait_gpu_idle(limit_mb: float = 1024, attempts: int = 6, pause: float = 5) -> None:
    for attempt in range(1, attempts + 1):
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
        used_text, util_text = [part.strip() for part in text.strip().splitlines()[0].split(",")]
        used = float(used_text)
        print(f"gpu before work: used {used:.0f} MB, util {util_text}%", flush=True)
        if used <= limit_mb:
            return
        if attempt == attempts:
            raise SystemExit(f"GPU is not idle: {used:.0f} MB already used")
        time.sleep(pause)


def _shuffled_order(n: int, seed: int) -> list[int]:
    order = list(range(n))
    random.Random(int(seed)).shuffle(order)
    return order


def _finish_valselect(config, state, output_dir, val_examples, test_examples, labels, prompt, release, run_dir, probe_path) -> None:
    import gc
    import torch

    best_dir = output_dir / "best"
    if not (best_dir / "adapter_model.safetensors").is_file() or not (best_dir / "pointer-head.pt").is_file():
        raise SystemExit(f"missing best checkpoint in {best_dir}")
    best = state.get("best") or (_read_state(best_dir / "selection.json") or {})
    if not best or best.get("step") is None:
        raise SystemExit("best checkpoint has no selection record")
    state["best"] = best
    full = state.get("val_full") or {}
    need_full = int(full.get("step") or -1) != int(best["step"]) or int(full.get("n") or -1) != len(val_examples)
    response_path = run_dir / f"{release}.jsonl"
    existing = recover_jsonl(response_path) if response_path.is_file() else []
    needed_ids = {f"spaq:{example['source_id']}:MOS" for example in test_examples}
    need_test = {row["item_id"] for row in existing} != needed_ids
    resources = None
    responses = existing
    if need_full or need_test:
        _wait_gpu_idle()
        model, processor, head_q, head_k, scale, layer, resident = _load_for_eval(config, best_dir)
        try:
            if need_full:
                started = time.perf_counter()
                loss = _mean_pointer_ce(
                    model, processor, layer, head_q, head_k, scale, val_examples, labels, prompt, tag="val-full"
                )
                state["val_full"] = {
                    "step": int(best["step"]),
                    "loss": round(float(loss), 6),
                    "n": len(val_examples),
                    "seconds": round(time.perf_counter() - started, 1),
                }
                write_json(output_dir / "trainer-state.json", state)
                write_json(best_dir / "selection.json", {**best, "val_full": state["val_full"]})
                print(f"full validation loss {state['val_full']['loss']:.4f} n {len(val_examples)}", flush=True)
            if need_test:
                responses, resources = _score_loaded(
                    model,
                    processor,
                    layer,
                    head_q,
                    head_k,
                    scale,
                    test_examples,
                    labels,
                    prompt,
                    resident=resident,
                    stream_path=response_path,
                )
        finally:
            del model, head_q, head_k
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if resources is None:
        previous = _read_state(run_dir / f"{release}.run.json") or {}
        resources = previous.get("inference_resources") or latency_summary(row.get("latency_ms") or 0 for row in responses)
    if len(responses) != len(test_examples):
        raise SystemExit(f"test responses {len(responses)} / {len(test_examples)}")
    gold = [_gold(example) for example in test_examples]
    score = summarize(gold, responses)
    probe = _read_state(probe_path) or {}
    last_meta = read_json(run_dir / f"{LAST_STEP_RELEASE}.run.json")
    last_score = read_json(run_dir / f"{LAST_STEP_RELEASE}.score.json")
    qlora_meta = read_json(run_dir / f"{QLORA_RELEASE}.run.json")
    qlora_score = read_json(run_dir / f"{QLORA_RELEASE}.score.json")
    versus_last = compare(gold, read_jsonl(run_dir / f"{LAST_STEP_RELEASE}.jsonl"), responses)
    versus_qlora = compare(gold, read_jsonl(run_dir / f"{QLORA_RELEASE}.jsonl"), responses)
    order_head = ",".join(str(index) for index in (state.get("order") or [])[:5])
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
        "seed": config["seed"],
        "n_train": state.get("seen") if int(state.get("optimizer_step") or 0) else None,
        "n_validation": len(val_examples),
        "n_test": len(test_examples),
        "head_init": config["head_init"],
        "checkpoint_dir": best_dir.relative_to(ROOT).as_posix(),
        "order_head": order_head,
        "eval_steps": int(config.get("eval_steps", config["save_steps"])),
        "monitor_size": (state.get("best") or {}).get("n"),
        "monitor_seed": state.get("monitor_seed", config.get("val_monitor_seed")),
        "selected_step": (state.get("best") or {}).get("step"),
        "monitor_loss": (state.get("best") or {}).get("monitor_loss"),
        "train_loss_selected": (state.get("best") or {}).get("train_loss"),
        "val_full_loss": (state.get("val_full") or {}).get("loss"),
        "val_full_n": (state.get("val_full") or {}).get("n"),
        "val_monitor": state.get("val_monitor") or [],
        "loss_start": state.get("loss_start"),
        "loss_end": state.get("loss_end"),
        "optimizer_step": state.get("optimizer_step"),
        "responses_path": response_path.relative_to(ROOT).as_posix(),
        "score_path": (run_dir / f"{release}.score.json").relative_to(ROOT).as_posix(),
        "inference_resources": resources,
        "merger_probe": probe,
    }
    # n_train is the completed training split, not the resume cursor.
    run_meta["n_train"] = len(state.get("order") or [])
    write_jsonl(response_path, responses)
    write_json(run_dir / f"{release}.score.json", score)
    write_json(run_dir / f"{release}.run.json", run_meta)
    report = render_pointer_lora_valselect(
        last_meta, last_score, qlora_meta, qlora_score, run_meta, score, versus_last, versus_qlora
    )
    report_path = ROOT / "outputs/reports" / f"{release}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    write_json(
        run_dir / f"{release}.compare.json",
        {"versus_last_step": versus_last, "versus_qlora": versus_qlora},
    )
    spaq = score["by_source"]["spaq"]
    qlora_ci = versus_qlora["by_source"]["spaq"].get("ci95")
    print(
        json.dumps(
            {
                "selected_step": run_meta["selected_step"],
                "monitor_loss": run_meta["monitor_loss"],
                "val_full_loss": run_meta["val_full_loss"],
                "spearman": spaq.get("spearman_rho"),
                "mae": spaq.get("mae"),
                "delta_vs_last": versus_last["by_source"]["spaq"]["delta_right_minus_left"],
                "delta_vs_last_ci": versus_last["by_source"]["spaq"].get("ci95"),
                "delta_vs_qlora": versus_qlora["by_source"]["spaq"]["delta_right_minus_left"],
                "delta_vs_qlora_ci": qlora_ci,
                "median_latency_ms": resources.get("median_latency_ms"),
                "gpu_max_allocated_mb": resources.get("gpu_max_allocated_mb"),
                "merger_fits": probe.get("fits"),
                "merger_oom": probe.get("oom"),
                "merger_headroom_mb": probe.get("headroom_mb"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pointer-lora.spaq.qwen35-2b.json")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    selecting = bool(config.get("select_by_validation"))
    _week0, labels, instruction, train_raw, val_raw, test_raw = _splits(config)
    prompt = score_user_text(instruction, labels)
    train_examples = _examples(train_raw, labels, instruction)
    val_examples = _examples(val_raw, labels, instruction) if selecting else []
    test_examples = _examples(test_raw, labels, instruction)
    if selecting:
        _check_validation_export(config, val_raw)
        if (len(train_examples), len(val_examples), len(test_examples)) != (7807, 1660, 1658):
            raise SystemExit(
                f"unexpected split sizes train={len(train_examples)} validation={len(val_examples)} test={len(test_examples)}"
            )
        _assert_same_training_order(_shuffled_order(len(train_examples), int(config["seed"])))
    output_dir = _output_dir(config, args.max_steps, args.eval_only or args.probe_only or args.check_only)
    release = config["release_id"] if args.max_steps is None else config["release_id"] + "-smoke"
    run_dir = ROOT / "outputs/runs"
    probe_path = run_dir / f"{release}.merger-probe.json"
    print(
        f"train={len(train_examples)} validation={len(val_examples)} test={len(test_examples)} output={output_dir}",
        flush=True,
    )
    if args.check_only:
        print("split check ok", flush=True)
        return
    if args.probe_only:
        if not selecting:
            raise SystemExit("--probe-only requires select_by_validation")
        order = _shuffled_order(len(train_examples), int(config["seed"]))
        probe = _probe_visual_merger(config, train_examples[order[0]], labels, prompt)
        write_json(probe_path, probe)
        print(json.dumps({key: probe.get(key) for key in ("oom", "grad_ok", "fits", "gpu_max_allocated_mb", "gpu_max_reserved_mb", "headroom_mb", "n_parameters")}, ensure_ascii=False), flush=True)
        return
    if selecting and not args.skip_probe and not probe_path.is_file():
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config), "--probe-only"]
        if args.max_steps is not None:
            command.extend(["--max-steps", str(args.max_steps)])
        print("running merger probe in a child process", flush=True)
        completed = subprocess.run(command, cwd=str(ROOT))
        if completed.returncode != 0 or not probe_path.is_file():
            raise SystemExit(f"merger probe failed with status {completed.returncode}")
        _wait_gpu_idle()
    if not args.eval_only and output_dir.resolve() == PROTECTED_CHECKPOINT.resolve():
        raise SystemExit(f"refusing to overwrite {PROTECTED_CHECKPOINT}")
    state = {"loss_start": None, "loss_end": None, "optimizer_step": None}
    if not args.eval_only:
        _wait_gpu_idle()
        state = _train(
            config,
            train_examples,
            labels,
            prompt,
            output_dir,
            args.max_steps,
            val_examples if selecting else None,
        )
    elif selecting:
        state = _read_state(output_dir / "trainer-state.json") or {}
    if args.skip_eval or args.check_only:
        print(json.dumps({"loss_start": state.get("loss_start"), "loss_end": state.get("loss_end"), "step": state.get("optimizer_step")}), flush=True)
        return
    if selecting:
        _finish_valselect(config, state, output_dir, val_examples, test_examples, labels, prompt, release, run_dir, probe_path)
        return
    responses, resources = _evaluate(config, test_examples, labels, prompt, output_dir)
    gold = [_gold(example) for example in test_examples]
    score = summarize(gold, responses)
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
        "responses_path": response_path.relative_to(ROOT).as_posix(),
        "score_path": score_path.relative_to(ROOT).as_posix(),
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
