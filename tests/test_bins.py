from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.bins import argmax_label, bin_labels, expected_value, format_bin, letter_for_index  # noqa: E402


class BinTests(unittest.TestCase):
    def test_koniq_integer_bins(self):
        self.assertEqual(bin_labels(1, 5, 1), [1, 2, 3, 4, 5])

    def test_spaq_decade_bins(self):
        labels = bin_labels(0, 100, 10)
        self.assertEqual(labels[0], 0)
        self.assertEqual(labels[-1], 100)
        self.assertEqual(len(labels), 11)

    def test_half_step_bins(self):
        self.assertEqual(bin_labels(1, 2, 0.5), [1, 1.5, 2])

    def test_expected_value(self):
        self.assertAlmostEqual(expected_value([0.1, 0.7, 0.2], [1, 2, 3]), 2.1)

    def test_argmax_stable(self):
        self.assertEqual(argmax_label([0.2, 0.4, 0.4], [1, 2, 3]), 2)

    def test_letters_and_format(self):
        self.assertEqual(letter_for_index(0), "A")
        self.assertEqual(format_bin(10.0), "10")
        self.assertEqual(format_bin(1.5), "1.5")


if __name__ == "__main__":
    unittest.main()
