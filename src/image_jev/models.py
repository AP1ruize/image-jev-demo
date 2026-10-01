from __future__ import annotations

from pathlib import Path
from typing import Any


def configure_pixels(processor, family: str, min_pixels: int, max_pixels: int) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    if family == "qwen35":
        image_processor.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = min_pixels
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels


def model_class(family: str):
    if family == "qwen3vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass
        return ModelClass
    if family == "qwen35":
        try:
            from transformers import Qwen3_5ForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass
        return ModelClass
    raise ValueError(f"unsupported family: {family}")


def load_model(config: dict[str, Any], *, local_dir: Path | None = None):
    import inspect
    import torch
    from transformers import AutoProcessor

    family = config["family"]
    source = str(local_dir or config["base_model"])
    dtype = torch.float16 if config.get("torch_dtype", "float16") == "float16" else torch.float32
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "attn_implementation": config.get("attn_implementation", "eager"),
        "device_map": {"": 0} if torch.cuda.is_available() else None,
    }
    if config.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        skip_modules = ["visual", "lm_head"]
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
            llm_int8_skip_modules=skip_modules,
        )
        if hasattr(quant, "modules_to_not_convert"):
            quant.modules_to_not_convert = skip_modules
        kwargs["quantization_config"] = quant
    signature = inspect.signature(model_class(family).from_pretrained)
    if "attn_implementation" not in signature.parameters:
        kwargs.pop("attn_implementation")
    model = model_class(family).from_pretrained(source, **kwargs)
    processor = AutoProcessor.from_pretrained(source if Path(source).exists() else config["base_model"])
    configure_pixels(processor, family, int(config["min_pixels"]), int(config["max_pixels"]))
    if torch.cuda.is_available() and not config.get("load_in_4bit"):
        model.to(torch.device("cuda:0"))
        print(f"using CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    elif torch.cuda.is_available():
        print(f"using CUDA device: {torch.cuda.get_device_name(0)} (4-bit)", flush=True)
    model.eval()
    return model, processor


def tokenizer_of(processor):
    return getattr(processor, "tokenizer", processor)
