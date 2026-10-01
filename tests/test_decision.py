from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.decision import (  # noqa: E402
    distribution_errors,
    hybrid_rows,
    nearest_bin_distribution,
    restarted_position_ids,
    train_pointer_synthetic,
    tree_attention_mask,
)
from image_jev.sources import split_name  # noqa: E402
from image_jev.report import render_heldout  # noqa: E402
from image_jev.warmup_data import score_user_text, select_split_records, select_train_records  # noqa: E402


class DecisionTests(unittest.TestCase):
    def test_nearest_bin_is_a_distribution(self):
        labels = [0, 10, 20, 30]
        target = nearest_bin_distribution(14, labels)
        self.assertEqual(target, [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(distribution_errors(target), [])

    def test_tree_mask_isolates_questions(self):
        mask = tree_attention_mask(3, [2, 2])
        # Layout: state 0..2, branch A 3..4, branch B 5..6.
        self.assertEqual(mask[4][3], 1)
        self.assertEqual(mask[4][0], 1)
        self.assertEqual(mask[4][5], 0)
        self.assertEqual(mask[5][3], 0)
        self.assertEqual(mask[5][4], 0)
        self.assertEqual(mask[1][3], 0)
        self.assertEqual(mask[1][2], 0)

    def test_positions_restart_after_state(self):
        self.assertEqual(restarted_position_ids(3, [2, 2]), [0, 1, 2, 3, 4, 3, 4])

    def test_qwen35_rows_do_not_pack_questions(self):
        rows = hybrid_rows([1, 2, 3], [[9, 9], [8]])
        self.assertEqual(rows, [[1, 2, 3, 9, 9], [1, 2, 3, 8]])
        self.assertNotIn(8, rows[0])
        self.assertNotIn(9, rows[1])

    def test_synthetic_pointer_loss_falls_and_stays_valid(self):
        result = train_pointer_synthetic()
        self.assertLess(result["losses"][-1], result["losses"][0])
        for probabilities in result["probabilities"]:
            self.assertEqual(distribution_errors(probabilities), [])

    def test_train_slice_skips_week0_test_split(self):
        records = [
            {"source_dataset": "spaq", "source_id": f"{index}.jpg", "group_id": f"spaq:{index}.jpg", "path": f"{index}.jpg", "labels": {"MOS": 50}}
            for index in range(30)
        ]
        chosen = select_train_records(
            records,
            dataset="spaq",
            seed="image-jev-week0-v0.1",
            train=0.70,
            validation=0.15,
            n=5,
        )
        self.assertEqual(len(chosen), 5)
        for record in chosen:
            self.assertEqual(
                split_name(record["group_id"], seed="image-jev-week0-v0.1", train=0.70, validation=0.15),
                "train",
            )
        text = score_user_text("Rate technical quality.", [0, 10])
        self.assertIn("||OPT00:0||", text)
        self.assertIn("||DECIDE||", text)

    def test_validation_slice_is_disjoint_from_train_and_test(self):
        records = [
            {
                "source_dataset": "spaq",
                "source_id": f"{index}.jpg",
                "group_id": f"spaq:{index}.jpg",
                "path": f"{index}.jpg",
                "labels": {"MOS": 50},
            }
            for index in range(80)
        ]
        kwargs = {
            "dataset": "spaq",
            "seed": "image-jev-week0-v0.1",
            "train": 0.70,
            "validation": 0.15,
        }
        train = select_split_records(records, split="train", n=10, **kwargs)
        validation = select_split_records(records, split="validation", n=5, **kwargs)
        train_ids = {record["source_id"] for record in train}
        self.assertTrue(train_ids.isdisjoint({record["source_id"] for record in validation}))
        for record in validation:
            self.assertEqual(split_name(record["group_id"], seed=kwargs["seed"], train=0.70, validation=0.15), "validation")
        with self.assertRaises(ValueError):
            select_split_records(records, split="test", n=1, **kwargs)
        meta = {
            "release_id": "heldout",
            "display_name": "pointer",
            "model_id": "m",
            "family": "qwen35",
            "quantization": "4bit",
            "base_model": "Qwen/Qwen3.5-2B",
            "max_pixels": 1,
            "n_train": 1,
            "n_validation": 1,
            "head_lr": 0.0002,
            "projection_dim": 8,
            "epochs": 1,
            "best_epoch": 0,
            "train_loss_start": 1.0,
            "train_loss_best": 1.0,
            "val_loss_start": 1.0,
            "val_loss_best": 1.0,
            "split_seed": "s",
            "pointer_responses": "a",
            "logits_responses": "b",
            "pointer_score": "c",
            "logits_score": "d",
            "compare_path": "e",
            "train_spearman_rho": 0.5,
            "train_mae": 1,
        }
        report = render_heldout(
            {"n_expected": 1, "n_responses": 1, "errors": {}, "by_source": {"spaq": {"n_total": 1, "n_valid": 1, "coverage": 1, "n_groups": 1, "schema_valid": 1, "median_latency_ms": 1, "spearman_rho": 0.1, "pearson_r": 0.1, "mae": 1, "primary_metric_ci95_group_bootstrap": [0.0, 0.2]}}},
            {"n_expected": 1, "n_responses": 1, "errors": {}, "by_source": {"spaq": {"n_total": 1, "n_valid": 1, "coverage": 1, "n_groups": 1, "schema_valid": 1, "median_latency_ms": 2, "spearman_rho": 0.2, "pearson_r": 0.2, "mae": 2, "primary_metric_ci95_group_bootstrap": [0.0, 0.3]}}},
            {"by_source": {"spaq": {"n_total": 1, "n_paired_valid": 1, "left_spearman_rho": 0.2, "right_spearman_rho": 0.1, "delta_right_minus_left": -0.1, "ci95": [-0.2, 0.0], "left_schema_valid": 1, "right_schema_valid": 1, "left_median_latency_ms": 2, "right_median_latency_ms": 1, "latency_ratio_right_over_left": 0.5}}},
            meta,
        )
        self.assertIn("## 分来源原生指标", report)
        self.assertIn("## 配对比较", report)
        self.assertIn("validation", report)


if __name__ == "__main__":
    unittest.main()
