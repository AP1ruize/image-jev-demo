from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_jev.grammar import (  # noqa: E402
    allowed_token_ids,
    compile_number_tokens,
    is_complete_number,
    is_number_prefix,
    json_number_allowed,
    make_prefix_allowed_fn,
)


class GrammarTests(unittest.TestCase):
    def test_prefix_and_complete(self):
        self.assertTrue(is_number_prefix(""))
        self.assertTrue(is_number_prefix("72."))
        self.assertTrue(is_complete_number("72.5"))
        self.assertFalse(is_complete_number("72."))
        self.assertFalse(is_number_prefix("72.}"))

    def test_token_continuation(self):
        self.assertTrue(json_number_allowed("7", "2"))
        self.assertTrue(json_number_allowed("72", "}"))
        self.assertFalse(json_number_allowed("72.", "}"))
        self.assertFalse(json_number_allowed("", "}"))

    def test_allowed_ids_from_toy_vocab(self):
        id_to_text = {0: "7", 1: "2", 2: ".", 3: "}", 4: "abc", 5: " "}
        groups = compile_number_tokens(id_to_text)
        allowed = set(allowed_token_ids("7", groups, id_to_text))
        self.assertIn(1, allowed)
        self.assertIn(2, allowed)
        self.assertIn(3, allowed)
        self.assertNotIn(4, allowed)

    def test_prefix_fn_stops_after_brace(self):
        id_to_text = {10: "4", 11: "}", 12: "eos"}
        groups = compile_number_tokens(id_to_text)
        fn = make_prefix_allowed_fn(2, lambda ids: "".join(id_to_text[i] for i in ids), groups, id_to_text, 12)
        self.assertEqual(fn(0, [1, 2, 10, 11]), [12])
        self.assertIn(10, fn(0, [1, 2]))


if __name__ == "__main__":
    unittest.main()
