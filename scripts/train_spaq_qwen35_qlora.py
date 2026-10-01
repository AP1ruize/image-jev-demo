#!/usr/bin/env python3
"""QLoRA-tune frozen Qwen3.5-2B 4-bit on the full SPAQ train split, then score test.

The vision encoder stays frozen. The language model gets a rank-8 adapter and
learns to emit {"value": number}. Inference time and GPU memory are measured
on the full Week 0 test split, for comparison with the pointer head.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.runtime import apply_runtime_env, load_local_config  # noqa: E402

apply_runtime_env(load_local_config())

from image_jev.compare import compare  # noqa: E402
from image_jev.io import read_json, write_json, write_jsonl  # noqa: E402
from image_jev.prompts import generate_prompt  # noqa: E402
from image_jev.report import render_full_pair  # noqa: E402
from image_jev.resources import cuda_snapshot, latency_summary, reset_cuda_peak  # noqa: E402
from image_jev.scoring import summarize  # noqa: E402
from image_jev.sources import collect_sources  # noqa: E402
from image_jev.warmup_data import select_split_records  # noqa: E402


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


def _gold(record: dict) -> dict:
    return {
        "item_id": f"spaq:{record['source_id']}:MOS",
        "source_dataset": "spaq",
        "source_id": record["source_id"],
        "split": "test",
        "group_id": record["group_id"],
        "task": {
            "dimension": "technical_quality",
            "type": "continuous",
            "prompt_id": "week0-iqa-v0.1:spaq",
            "scale": [0, 100],
            "gold_key": "MOS",
        },
        "gold": {"origin": "source_human", "value": float(record["labels"]["MOS"])},
    }


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
    train_ids = {row["source_id"] for row in train_rows}
    if train_ids & {row["source_id"] for row in test_rows}:
        raise SystemExit("train and test overlap")
    return week0, train_rows, test_rows


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


def _collate(examples, processor, min_pixels: int, max_pixels: int):
    import torch

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    texts, prefixes, images = [], [], []
    for example in examples:
        user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["prompt"]}]}]
        full = user + [{"role": "assistant", "content": [{"type": "text", "text": example["response"]}]}]
        texts.append(processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False))
        prefixes.append(processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True))
        images.append(example["image"])
    del min_pixels, max_pixels
    batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
    prefix = processor(text=prefixes, images=images, padding=True, return_tensors="pt")
    labels = batch["input_ids"].clone()
    for row in range(labels.size(0)):
        prefix_len = min(int(prefix["attention_mask"][row].sum().item()), int(batch["attention_mask"][row].sum().item()))
        labels[row, :prefix_len] = -100
        labels[row, batch["attention_mask"][row] == 0] = -100
        if pad_id is not None:
            labels[row, batch["input_ids"][row] == pad_id] = -100
    batch["labels"] = labels
    return batch


def _train(config: dict, train_rows: list[dict], prompt: str, output_dir: Path) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainingArguments
    from PIL import Image

    class Rows(Dataset):
        def __len__(self):
            return len(train_rows)

        def __getitem__(self, index):
            record = train_rows[index]
            mos = round(float(record["labels"]["MOS"]), 2)
            with Image.open(record["path"]) as image:
                rgb = image.convert("RGB")
            return {"prompt": prompt, "response": json.dumps({"value": mos}, ensure_ascii=False), "image": rgb}

    model, processor = _load_base(config)
    model.gradient_checkpointing_enable()
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
    model.print_trainable_parameters()
    torch.cuda.empty_cache()
    print(f"cuda after adapter: {cuda_snapshot()}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=float(config["num_train_epochs"]),
            per_device_train_batch_size=int(config["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
            learning_rate=float(config["learning_rate"]),
            fp16=True,
            bf16=False,
            gradient_checkpointing=True,
            logging_steps=int(config["logging_steps"]),
            save_steps=int(config["save_steps"]),
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            report_to="none",
            seed=int(config["seed"]),
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            label_names=["labels"],
        ),
        train_dataset=Rows(),
        data_collator=lambda examples: _collate(examples, processor, int(config["min_pixels"]), int(config["max_pixels"])),
    )
    resume = sorted(output_dir.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=str(resume[-1]) if resume else None)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(output_dir)
    del trainer, model
    torch.cuda.empty_cache()


def _evaluate(config: dict, test_rows: list[dict], prompt: str, adapter_dir: Path) -> tuple[list[dict], dict]:
    import torch
    from PIL import Image
    from peft import PeftModel

    from image_jev.inference import generate_json

    base, processor = _load_base(config)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    resident = cuda_snapshot()
    reset_cuda_peak()
    responses = []
    for index, record in enumerate(test_rows, start=1):
        with Image.open(record["path"]) as image:
            rgb = image.convert("RGB")
        result = generate_json(
            model,
            processor,
            rgb,
            prompt,
            max_new_tokens=int(config["max_new_tokens"]),
            enable_thinking=False,
        )
        result["item_id"] = f"spaq:{record['source_id']}:MOS"
        responses.append(result)
        if index == 1 or index % 50 == 0 or index == len(test_rows):
            print(f"generate {index}/{len(test_rows)} {record['source_id']} {result.get('latency_ms')} ms", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    resources = {
        "resident_after_load": resident,
        **latency_summary(row["latency_ms"] for row in responses if isinstance(row.get("latency_ms"), (int, float))),
        **cuda_snapshot(),
    }
    return responses, resources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qlora.spaq.qwen35-2b.json")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    week0, train_rows, test_rows = _splits(config)
    prompt = generate_prompt(week0, "spaq")
    adapter_dir = ROOT / "outputs/qlora" / config["release_id"]
    print(f"train={len(train_rows)} test={len(test_rows)}", flush=True)
    if not args.eval_only:
        _train(config, train_rows, prompt, adapter_dir)
    responses, resources = _evaluate(config, test_rows, prompt, adapter_dir)
    gold = [_gold(record) for record in test_rows]
    score = summarize(gold, responses)
    release = config["release_id"]
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
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "responses_path": str(response_path.relative_to(ROOT)),
        "score_path": str(score_path.relative_to(ROOT)),
        "inference_resources": resources,
    }
    write_json(run_dir / f"{release}.run.json", run_meta)
    pointer_meta_path = run_dir / "spaq-pointer-full-v0.1.run.json"
    pointer_score_path = run_dir / "spaq-pointer-full-v0.1-pointer.score.json"
    if pointer_meta_path.is_file() and pointer_score_path.is_file():
        pointer_meta = read_json(pointer_meta_path)
        pointer_score = read_json(pointer_score_path)
        pointer_rows = read_json_lines(run_dir / "spaq-pointer-full-v0.1-pointer.jsonl")
        paired = compare(gold, responses, pointer_rows)
        report = render_full_pair(pointer_meta, pointer_score, run_meta, score, paired)
        report_path = ROOT / "outputs/reports" / "spaq-full-pointer-vs-qwen35-4bit.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        write_json(run_dir / "spaq-full-pointer-vs-qwen35-4bit.compare.json", paired)
        print(f"wrote {report_path}", flush=True)
    spaq = score["by_source"]["spaq"]
    print(
        json.dumps(
            {
                "spearman": spaq.get("spearman_rho"),
                "mae": spaq.get("mae"),
                "median_latency_ms": resources.get("median_latency_ms"),
                "gpu_max_allocated_mb": resources.get("gpu_max_allocated_mb"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def read_json_lines(path: Path) -> list[dict]:
    from image_jev.io import read_jsonl

    return read_jsonl(path)


if __name__ == "__main__":
    main()
