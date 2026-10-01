from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.parse import in_scale, parse_value  # noqa: E402


class ParseTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_value('{"value":4.2}'), (4.2, None))

    def test_fenced_json(self):
        value, error = parse_value("```json\n{\"value\": 71}\n```")
        self.assertEqual(value, 71.0)
        self.assertIsNone(error)

    def test_missing_key(self):
        self.assertEqual(parse_value('{"score":1}'), (None, "missing_value"))

    def test_rejects_bool(self):
        self.assertEqual(parse_value('{"value": true}'), (None, "invalid_value_type"))

    def test_scale(self):
        self.assertTrue(in_scale(3.2, [1, 5]))
        self.assertFalse(in_scale(9, [1, 5]))
        self.assertFalse(in_scale(True, [0, 1]))


if __name__ == "__main__":
    unittest.main()
