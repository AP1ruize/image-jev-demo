from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.report import merger_decision, render_pointer_lora_valselect  # noqa: E402


def _load_trainer():
    path = ROOT / "scripts" / "train_spaq_pointer_lora.py"
    spec = importlib.util.spec_from_file_location("train_spaq_pointer_lora", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PointerLoraValselectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = _load_trainer()

    def test_monitor_indices_are_a_stable_subset(self):
        first = self.trainer.monitor_indices(1660, 256, 42)
        second = self.trainer.monitor_indices(1660, 256, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 256)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(set(first)), 256)
        self.assertEqual(self.trainer.monitor_indices(10, 10, 42), list(range(10)))

    def test_lower_validation_loss_keeps_the_earlier_tie(self):
        self.assertTrue(self.trainer.keeps_lower_validation_loss(None, 1.2))
        current = {"monitor_loss": 1.2}
        self.assertTrue(self.trainer.keeps_lower_validation_loss(current, 1.1))
        self.assertFalse(self.trainer.keeps_lower_validation_loss(current, 1.2))
        self.assertFalse(self.trainer.keeps_lower_validation_loss(current, 1.3))

    def test_report_names_the_selected_step_and_both_deltas(self):
        score = {
            "by_source": {
                "spaq": {
                    "n_total": 1658,
                    "n_valid": 1658,
                    "coverage": 1,
                    "schema_valid": 1,
                    "spearman_rho": 0.9,
                    "primary_metric_ci95_group_bootstrap": [0.89, 0.91],
                    "pearson_r": 0.91,
                    "mae": 7,
                    "median_latency_ms": 400,
                }
            }
        }
        paired = {"by_source": {"spaq": {"delta_right_minus_left": -0.01, "ci95": [-0.02, -0.005], "n_paired_valid": 1658, "n_total": 1658}}}
        meta = {
            "release_id": "spaq-pointer-lora-valselect-v0.1",
            "model_id": "qwen3.5-2b-4bit-pointer-lora-valselect",
            "epochs": 1,
            "lora_r": 8,
            "head_init": "outputs/warmup/spaq-pointer-full-v0.1/pointer-head.pt",
            "checkpoint_dir": "outputs/qlora/spaq-pointer-lora-v0.1-valselect/best",
            "n_train": 7807,
            "n_validation": 1660,
            "n_test": 1658,
            "max_pixels": 65536,
            "seed": 42,
            "order_head": "1,2,3,4,5",
            "eval_steps": 100,
            "monitor_size": 256,
            "monitor_seed": 42,
            "selected_step": 200,
            "monitor_loss": 1.1,
            "val_full_loss": 1.12,
            "val_full_n": 1660,
            "loss_start": 1.3,
            "train_loss_selected": 1.2,
            "loss_end": 1.5,
            "val_monitor": [{"step": 100, "train_loss": 1.25, "loss": 1.2}, {"step": 200, "train_loss": 1.2, "loss": 1.1}],
            "inference_resources": {"median_latency_ms": 400, "p95_latency_ms": 500, "gpu_max_allocated_mb": 2900, "gpu_max_reserved_mb": 3000},
            "merger_probe": {
                "modules": ["model.visual.merger"],
                "n_parameters": 10,
                "oom": False,
                "grad_ok": True,
                "fits": True,
                "gpu_max_allocated_mb": 3000,
                "gpu_max_reserved_mb": 3200,
                "gpu_total_mb": 4096,
                "nvidia_used_mb": 3400,
                "nvidia_total_mb": 4096,
                "headroom_mb": 696,
            },
        }
        text = render_pointer_lora_valselect(meta, score, meta, score, meta, score, paired, paired)
        self.assertIn("选中 step：200", text)
        self.assertIn("监控子集：256 / 1660", text)
        self.assertIn("全量验证交叉熵：1.1200（n=1660）", text)
        self.assertIn("相对最后一步语言 LoRA + 指针头", text)
        self.assertIn("相对 JSON QLoRA", text)
        self.assertIn("本次不启动那个训练", text)

    def test_merger_decision_stops_when_memory_or_interval_does_not_qualify(self):
        self.assertIn("OOM", merger_decision({"oom": True}, [-0.02, -0.01]))
        self.assertIn("不到 512MB", merger_decision({"oom": False, "grad_ok": True, "fits": False}, [-0.02, -0.01]))
        self.assertIn("没有整段低于零", merger_decision({"oom": False, "grad_ok": True, "fits": True}, [-0.02, 0.01]))
        self.assertIn("短程", merger_decision({"oom": False, "grad_ok": True, "fits": True}, [-0.02, -0.01]))


if __name__ == "__main__":
    unittest.main()
