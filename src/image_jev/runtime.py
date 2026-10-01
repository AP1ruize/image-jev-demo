from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def load_local_config(path: Path | None = None) -> dict[str, Any]:
    target = path or (ROOT / "configs/local.json")
    if not target.is_file():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def apply_runtime_env(local: dict[str, Any] | None = None) -> None:
    """Set CUDA / Hub env vars before importing torch when possible."""
    import os

    cfg = local if local is not None else load_local_config()
    os.environ.setdefault("CUDA_DEVICE_ORDER", str(cfg.get("cuda_device_order") or "PCI_BUS_ID"))
    visible = cfg.get("cuda_visible_devices")
    if visible and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible)
    if cfg.get("hf_hub_disable_xet", True):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
