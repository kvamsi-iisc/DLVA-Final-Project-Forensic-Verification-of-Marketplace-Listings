import unittest

from listing_verifier.demo.verifier import extract_claims, guess_noun


class ClaimExtractionTests(unittest.TestCase):
    def test_extracts(self):
        c = extract_claims("Brown leather sofa, set of 2")
        self.assertEqual(c, {"colour": "brown", "material": "leather", "count": 2})
        self.assertEqual(guess_noun("Brown leather sofa"), "sofa")

    def test_ambiguous_skipped(self):
        self.assertNotIn("colour", extract_claims("black and white striped mug"))
        self.assertNotIn("count", extract_claims("mug"))  # count only when stated

    def test_negation(self):
        self.assertNotIn("material", extract_claims("not leather chair"))
        self.assertEqual(extract_claims("chair, no glass, red")["colour"], "red")
        self.assertNotIn("colour", extract_claims("gold frame"))   # metallic shades are not checkable

    def test_single(self):
        self.assertEqual(extract_claims("a single item: red mug")["count"], 1)


if __name__ == "__main__":
    unittest.main()
