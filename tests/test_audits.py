import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from listing_verifier.eval.metrics import operating_point, summarise_detector
from listing_verifier.eval.nuisance import nuisance_audit
from listing_verifier.features.device import is_cuda, pick_dtype
from listing_verifier.features.extract import FeatureStore, extract


class OperatingPointTests(unittest.TestCase):
    def test_counts(self):
        y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.9, 0.8, 0.2])
        op = operating_point(y, s, 0.5)
        self.assertEqual((op["tp"], op["fp"], op["fn"], op["tn"]), (1, 1, 1, 1))
        self.assertIn("fpr_at_threshold", summarise_detector(y, s, 0.5))


class DeviceTests(unittest.TestCase):
    def test_indexed_cuda_is_cuda(self):
        self.assertTrue(is_cuda("cuda:1"))
        self.assertFalse(is_cuda("cpu"))
        import torch
        self.assertEqual(pick_dtype("cpu", "bf16"), torch.float32)


class NuisanceAuditTests(unittest.TestCase):
    def _make(self, d, square_fakes):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(160):
            fake = i % 2 == 1
            size = (256, 256) if (fake and square_fakes) else (256, 192)
            p = d / f"{i}.jpg"
            Image.fromarray((rng.random((size[1], size[0], 3)) * 255).astype("uint8")).save(p, quality=92)
            rows.append({"image_key": f"k{i}", "path": str(p), "domain": "catalogue", "label": int(fake),
                         "generator": "g" if fake else "real", "split": "train" if i < 100 else "test"})
        return pd.DataFrame(rows)

    def test_detects_square_shortcut_and_passes_when_matched(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            leaky = nuisance_audit(self._make(d, square_fakes=True))
            self.assertGreater(leaky["nuisance_only_test_auroc"], 0.9)
            self.assertTrue(leaky["verdict"].startswith("WARNING"))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            clean = nuisance_audit(self._make(d, square_fakes=False))
            self.assertLess(clean["nuisance_only_test_auroc"], 0.75)


class FeatureStoreEdgeTests(unittest.TestCase):
    def test_all_missing_keys(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); p = d / "a.jpg"; Image.new("RGB", (40, 30)).save(p)
            extract(pd.DataFrame({"image_key": ["k0"], "path": [str(p)]}), None, 0, d, verbose=False)
            g = FeatureStore(d).get(["missing"])
            self.assertEqual(g["clip"].shape, (0, 768))
            self.assertFalse(g["present"].any())


if __name__ == "__main__":
    unittest.main()


class CacheIntegrityTests(unittest.TestCase):
    def test_fingerprint_mismatch_is_refused_and_no_tmp_left(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); p = d / "a.jpg"; Image.new("RGB", (40, 30)).save(p)
            extract(pd.DataFrame({"image_key": ["k0"], "path": [str(p)]}), None, 0, d, verbose=False)
            self.assertFalse(list(d.glob("*.tmp.npz")))
            z = dict(np.load(d / "features_q0.npz"))
            z["fingerprint"] = np.array("clip=other;fmt=0")
            np.savez(d / "features_q0.npz", **z)
            with self.assertRaises(RuntimeError):
                FeatureStore(d).load(0)
            with self.assertRaises(RuntimeError):
                extract(pd.DataFrame({"image_key": ["k1"], "path": [str(p)]}), None, 0, d, verbose=False)

    def test_inconsistent_cache_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); p = d / "a.jpg"; Image.new("RGB", (40, 30)).save(p)
            extract(pd.DataFrame({"image_key": ["k0"], "path": [str(p)]}), None, 0, d, verbose=False)
            z = dict(np.load(d / "features_q0.npz"))
            z["clip"] = np.zeros((2, 768), np.float32)      # one more row than keys
            np.savez(d / "features_q0.npz", **z)
            with self.assertRaises(RuntimeError):
                FeatureStore(d).load(0)


class FakeIndexDedupeTests(unittest.TestCase):
    def test_duplicate_keys_across_chunks_kept_once(self):
        from listing_verifier.data.index import fakes_from_chunks
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for sub in ("runA", "runB"):
                (d / sub).mkdir()
                img = d / sub / "x.jpg"; Image.new("RGB", (8, 8)).save(img)
                pd.DataFrame([{"image_key": "fake:g:t2i:0.00:item:1", "item_id": "item", "path": str(img), "generator": "g",
                               "mode": "t2i", "strength": 0.0}]).to_csv(d / sub / "chunk_0000.csv", index=False)
            df = fakes_from_chunks(d, None)
            self.assertEqual(len(df), 1)
