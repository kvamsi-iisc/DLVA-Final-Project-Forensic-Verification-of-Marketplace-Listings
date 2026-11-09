import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class EmptyPairsTests(unittest.TestCase):
    def test_featurise_handles_attribute_without_pairs(self):
        from listing_verifier.data.mismatch import make_pairs
        mod = load_script("10_train_eval_consistency")
        idx = pd.DataFrame({"image_key": ["a", "b"], "split": ["train", "train"], "label": [0, 0], "product_noun": ["mug", "mug"],
                            "colour": ["red", None], "material": [None, None], "count": [None, None], "count_verifiable": [False, False]})
        empty = make_pairs(idx, "material", "train")
        self.assertEqual(len(empty), 0)
        X, keep = mod.featurise(empty, {}, True)      # used to raise KeyError: 'image_key'
        self.assertEqual(X.shape, (0, 7))
        self.assertEqual(keep.shape, (0,))
        colour = make_pairs(idx, "colour", "train")
        self.assertEqual(len(colour), 0)                # only one colour value exists: no negative possible
        X, keep = mod.featurise(colour, {}, True)
        self.assertEqual(len(keep), 0)


if __name__ == "__main__":
    unittest.main()
