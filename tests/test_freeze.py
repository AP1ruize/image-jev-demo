from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.freeze import build_item, select_test  # noqa: E402
from image_jev.io import read_json  # noqa: E402
from image_jev.prompts import generate_prompt, logits_prompt  # noqa: E402
from image_jev.sources import split_name  # noqa: E402


class FreezeTests(unittest.TestCase):
    def test_split_is_deterministic(self):
        a = split_name("spaq:00001.jpg", seed="image-jev-week0-v0.1", train=0.7, validation=0.15)
        b = split_name("spaq:00001.jpg", seed="image-jev-week0-v0.1", train=0.7, validation=0.15)
        self.assertEqual(a, b)

    def test_select_test_respects_quota_and_seed(self):
        records = [
            {"source_dataset": "spaq", "source_id": f"{i:05d}.jpg", "group_id": f"spaq:{i:05d}.jpg", "labels": {"MOS": i}}
            for i in range(40)
        ]
        spec = {
            "sampling_seed": "unit",
            "split": {"train": 0.5, "validation": 0.2},
            "benchmark_samples": {"spaq": 5},
        }
        first = select_test(records, spec)
        second = select_test(records, spec)
        self.assertEqual([item["source_id"] for item in first], [item["source_id"] for item in second])
        self.assertEqual(len(first), 5)

    def test_build_item_omits_gold_from_run_shape_fields(self):
        spec = read_json(ROOT / "configs/week0.v0.1.json")
        record = {
            "source_dataset": "koniq10k",
            "source_id": "x.jpg",
            "group_id": "koniq10k:x.jpg",
            "labels": {"MOS": 3.4},
        }
        item = build_item(record, {"asset_id": "koniq10k:x.jpg", "path": "a.jpg", "sha256": "0" * 64}, spec)
        self.assertEqual(item["gold"]["value"], 3.4)
        self.assertEqual(item["task"]["scale"], [1, 5])
        self.assertNotIn("KEEP", json_blob(item))

    def test_prompts_are_quality_not_aesthetics(self):
        spec = read_json(ROOT / "configs/week0.v0.1.json")
        gen = generate_prompt(spec, "koniq10k")
        logits = logits_prompt(spec, "spaq")
        self.assertIn("JSON", gen)
        self.assertIn("A. 0", logits)
        self.assertIn("K. 100", logits)
        self.assertNotRegex(gen.lower(), r"lut|filter|keep|reject")


def json_blob(payload) -> str:
    import json

    return json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
