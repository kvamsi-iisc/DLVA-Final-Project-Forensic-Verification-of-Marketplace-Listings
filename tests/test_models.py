import tempfile
import unittest
from pathlib import Path

import numpy as np

from listing_verifier.eval.metrics import auroc, expected_calibration_error, threshold_at_fpr, tpr_at_fpr
from listing_verifier.models.consistency import ClaimCalibrator, claim_features, option_index, prompts_for
from listing_verifier.models.fusion import fit_fusion
from listing_verifier.models.probe import LinearProbe, train_probe
from listing_verifier.models.spectrum_head import load_spectrum_head, save_spectrum_head, train_spectrum_head


class MetricTests(unittest.TestCase):
    def test_perfect_and_random(self):
        y = np.r_[np.zeros(50), np.ones(50)]
        self.assertEqual(auroc(y, y), 1.0)
        self.assertAlmostEqual(tpr_at_fpr(y, y, 0.05), 1.0)
        self.assertTrue(np.isnan(auroc(np.zeros(10), np.random.rand(10))))

    def test_threshold_at_fpr(self):
        y = np.r_[np.zeros(1000), np.ones(10)]
        s = np.r_[np.linspace(0, 1, 1000), np.ones(10)]
        th = threshold_at_fpr(y, s, 0.05)
        self.assertAlmostEqual(np.mean(s[y == 0] >= th), 0.05, delta=0.01)

    def test_ece_perfect_calibration_small(self):
        rng = np.random.default_rng(0)
        p = rng.random(20000)
        y = (rng.random(20000) < p).astype(int)
        self.assertLess(expected_calibration_error(y, p), 0.02)


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 400
        self.y = np.arange(n) % 2
        self.X = rng.normal(size=(n, 768)).astype(np.float32)
        self.X[self.y == 1, :10] += 1.0
        self.maps = rng.normal(size=(n, 64, 64)).astype(np.float32)
        self.maps[self.y == 1, 48:, 48:] += 0.8
        self.rad = rng.normal(size=(n, 128)).astype(np.float32)

    def test_probe_roundtrip(self):
        p = train_probe(self.X[:300], self.y[:300], self.X[300:], self.y[300:])
        self.assertGreater(p.meta["val_auroc"], 0.8)
        with tempfile.TemporaryDirectory() as d:
            p.save(Path(d) / "probe.npz")
            q = LinearProbe.load(Path(d) / "probe.npz")
            self.assertTrue(np.allclose(p.logits(self.X[:5]), q.logits(self.X[:5])))

    def test_spectrum_head_roundtrip(self):
        r = train_spectrum_head(self.maps[:300], self.rad[:300], self.y[:300], self.maps[300:], self.rad[300:],
                                self.y[300:], epochs=4, device="cpu", verbose=False)
        with tempfile.TemporaryDirectory() as d:
            save_spectrum_head(r["model"], Path(d) / "s.pt")
            m = load_spectrum_head(Path(d) / "s.pt")
            a = r["model"].predict_logits(self.maps[:8], self.rad[:8])
            b = m.predict_logits(self.maps[:8], self.rad[:8])
            self.assertTrue(np.allclose(a, b, atol=1e-5))

    def test_fusion(self):
        f = fit_fusion(np.r_[np.zeros(50), np.ones(50)] * 4 - 2, np.random.randn(100), np.r_[np.zeros(50), np.ones(50)])
        self.assertGreater(f.meta["weight_probe"], f.meta["weight_spectrum"])

    def test_consistency_bits(self):
        self.assertEqual(option_index("colour", "red"), 5)
        self.assertEqual(option_index("count", 3), 2)
        self.assertEqual(len(prompts_for("colour", "chair")), 12)
        f = claim_features(np.array([0.7, 0.2, 0.1]), None, 0)
        self.assertEqual(f.shape, (7,))
        cal = ClaimCalibrator()
        X = np.random.randn(100, 7).astype(np.float32)
        cal.fit("colour", X, (X[:, 0] > 0).astype(int))
        with tempfile.TemporaryDirectory() as d:
            cal.save(Path(d) / "c.npz")
            c2 = ClaimCalibrator.load(Path(d) / "c.npz")
            self.assertTrue(np.allclose(cal.predict_proba("colour", X[:3]), c2.predict_proba("colour", X[:3])))


if __name__ == "__main__":
    unittest.main()
