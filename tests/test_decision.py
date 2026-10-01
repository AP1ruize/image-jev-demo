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
from image_jev.warmup_data import score_user_text, select_train_records  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
