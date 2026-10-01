from __future__ import annotations

import json
import math
import re
from typing import Any


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_value(content: Any) -> tuple[Any, str | None]:
    """Parse a model string that should be JSON `{\"value\": number}`."""
    if not isinstance(content, str):
        return None, "assistant_content_not_text"
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return _recover_number(text)
    if not isinstance(result, dict) or "value" not in result:
        return None, "missing_value"
    value = result["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None, "invalid_value_type"
    return float(value), None


def _recover_number(text: str) -> tuple[Any, str | None]:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None, "invalid_json"
    try:
        value = float(match.group(0))
    except ValueError:
        return None, "invalid_json"
    if not math.isfinite(value):
        return None, "invalid_value_type"
    return value, "recovered_number"


def in_scale(value: Any, scale: list[float] | tuple[float, float]) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(float(value)):
        return False
    return float(scale[0]) <= float(value) <= float(scale[1])
