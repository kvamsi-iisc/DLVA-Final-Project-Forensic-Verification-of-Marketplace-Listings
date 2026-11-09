import unittest

import pandas as pd

from listing_verifier.data.splits import assign_splits, leakage_report, split_for_key


class SplitTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(split_for_key("rivet|modern sofa"), split_for_key("rivet|modern sofa"))

    def test_fractions_roughly_respected(self):
        keys = [f"brand{i % 50}|family{i}" for i in range(5000)]
        splits = pd.Series([split_for_key(k) for k in keys]).value_counts(normalize=True)
        self.assertAlmostEqual(splits["train"], 0.70, delta=0.03)
        self.assertAlmostEqual(splits["test"], 0.20, delta=0.03)

    def test_no_family_leaks(self):
        df = pd.DataFrame({"family_key": [f"f{i % 30}" for i in range(300)], "brand": [f"b{i % 7}" for i in range(300)]})
        df = assign_splits(df)
        rep = leakage_report(df)
        self.assertEqual(sum(rep["split_sizes"].values()), 300)
        self.assertIn("n_brands_in_both_train_and_test", rep)


if __name__ == "__main__":
    unittest.main()
