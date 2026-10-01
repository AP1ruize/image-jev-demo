from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.scoring import continuous, ranks, summarize  # noqa: E402
from image_jev.compare import compare  # noqa: E402
from image_jev.tokens import resolve_candidates, softmax  # noqa: E402


def item(item_id: str, source: str, gold: float, scale=(1, 5)):
    return {
        "item_id": item_id,
        "source_dataset": source,
        "group_id": item_id,
        "task": {"type": "continuous", "scale": list(scale)},
        "gold": {"value": gold},
    }


class ScoringTests(unittest.TestCase):
    def test_tie_aware_ranks(self):
        self.assertEqual(ranks([3, 1, 1, 2]), [4.0, 1.5, 1.5, 3.0])

    def test_perfect_spearman(self):
        rows = [(item("a", "koniq10k", 1.0), 1.0), (item("b", "koniq10k", 2.0), 2.0), (item("c", "koniq10k", 3.0), 3.0)]
        self.assertAlmostEqual(continuous(rows)["spearman_rho"], 1.0)

    def test_out_of_range_lowers_coverage(self):
        gold = [item("a", "koniq10k", 2.0), item("b", "koniq10k", 3.0)]
        result = summarize(
            gold,
            [
                {"item_id": "a", "prediction": 2.0, "error": None, "schema_valid": True},
                {"item_id": "b", "prediction": 9.0, "error": None, "schema_valid": True},
            ],
        )
        self.assertEqual(result["by_source"]["koniq10k"]["coverage"], 0.5)
        self.assertEqual(result["by_source"]["koniq10k"]["schema_valid"], 1.0)

    def test_missing_response_counted(self):
        gold = [item("a", "spaq", 50, (0, 100)), item("b", "spaq", 60, (0, 100))]
        result = summarize(gold, [{"item_id": "a", "prediction": 50, "error": None, "schema_valid": True}])
        self.assertEqual(result["errors"]["missing_response"], 1)
        self.assertEqual(result["by_source"]["spaq"]["n_valid"], 1)

    def test_paired_compare_requires_complete_runs(self):
        gold = [item("a", "spaq", 10, (0, 100)), item("b", "spaq", 20, (0, 100))]
        left = [{"item_id": "a", "prediction": 10, "error": None, "schema_valid": True, "latency_ms": 100}]
        right = [
            {"item_id": "a", "prediction": 10, "error": None, "schema_valid": True, "latency_ms": 40},
            {"item_id": "b", "prediction": 20, "error": None, "schema_valid": True, "latency_ms": 40},
        ]
        with self.assertRaisesRegex(ValueError, "every frozen item"):
            compare(gold, left, right, resamples=10)

    def test_paired_delta_and_faster_logits(self):
        gold = [item("a", "spaq", 10, (0, 100)), item("b", "spaq", 90, (0, 100))]
        left = [
            {"item_id": "a", "prediction": 20, "error": None, "schema_valid": False, "latency_ms": 200},
            {"item_id": "b", "prediction": 80, "error": None, "schema_valid": True, "latency_ms": 220},
        ]
        right = [
            {"item_id": "a", "prediction": 12, "error": None, "schema_valid": True, "latency_ms": 50},
            {"item_id": "b", "prediction": 88, "error": None, "schema_valid": True, "latency_ms": 60},
        ]
        result = compare(gold, left, right, resamples=20)
        row = result["by_source"]["spaq"]
        self.assertEqual(row["n_paired_valid"], 2)
        self.assertEqual(row["right_schema_valid"], 1.0)
        self.assertLess(row["latency_ratio_right_over_left"], 1.0)
        self.assertGreaterEqual(row["right_spearman_rho"], row["left_spearman_rho"])


class TokenTests(unittest.TestCase):
    def test_softmax(self):
        probs = softmax([0.0, 0.0])
        self.assertAlmostEqual(probs[0], 0.5)
        self.assertAlmostEqual(sum(probs), 1.0)

    def test_letter_fallback_when_numbers_are_multi_token(self):
        def encode(text: str):
            if text.strip() in {"A", "B", "C"}:
                return [{"A": 1, "B": 2, "C": 3}[text.strip()]]
            return [100, 101]

        candidates = resolve_candidates(encode, {}, [1, 2, 3])
        self.assertEqual([c.token_id for c in candidates], [1, 2, 3])
        self.assertEqual(candidates[0].letter, "A")

    def test_single_token_numbers(self):
        def encode(text: str):
            table = {"1": [11], "2": [12], "3": [13], " 1": [11], " 2": [12], " 3": [13]}
            return table.get(text, [99, 98])

        candidates = resolve_candidates(encode, {11: "1", 12: "2", 13: "3"}, [1, 2, 3])
        self.assertEqual([c.token_id for c in candidates], [11, 12, 13])


if __name__ == "__main__":
    unittest.main()
