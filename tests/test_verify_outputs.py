"""The artifact validator must accept every real checkpoint type and reject empties."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


class VerifyOutputsTests(unittest.TestCase):
    def test_real_artifact_types(self):
        from listing_verifier.features.extract import extract
        from listing_verifier.models.consistency import ClaimCalibrator
        from listing_verifier.models.fusion import LateFusion
        from listing_verifier.models.probe import LinearProbe
        from listing_verifier.models.spectrum_head import SpectrumHead, save_spectrum_head
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            LinearProbe(np.ones(768, np.float32), 0.1).save(d / "probe.npz")
            cal = ClaimCalibrator()
            cal.fit("colour", np.random.randn(50, 7), (np.random.rand(50) > .5).astype(int))
            cal.save(d / "calibrator.npz")
            LateFusion(np.ones(2), 0.0, np.zeros(2), np.ones(2)).save(d / "fusion.npz")
            save_spectrum_head(SpectrumHead(), d / "spectrum.pt")
            img = d / "a.jpg"
            Image.new("RGB", (40, 30)).save(img)
            extract(pd.DataFrame({"image_key": ["k0"], "path": [str(img)]}), None, 0, d, verbose=False)
            (d / "ok.json").write_text("{}")
            (d / "rows.csv").write_text("a,b\n1,2\n")
            good = [d / n for n in ("probe.npz", "calibrator.npz", "fusion.npz", "spectrum.pt", "features_q0.npz", "ok.json", "rows.csv")] + [d]
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_outputs.py"), *map(str, good)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            (d / "empty.csv").write_text("a,b\n")
            np.savez(d / "features_q9.npz", clip=np.zeros((1, 2)))   # feature cache without image_key
            (d / "bad.json").write_text("{")
            np.savez(d / "probe_badshape.npz", w=np.ones(10, np.float32), b=np.float32(0), normalise=True)
            np.savez(d / "fusion_nan.npz", w=np.array([np.nan, 1.0]), b=np.float32(0), mean=np.zeros(2), std=np.ones(2))
            np.savez(d / "calibrator_broken.npz", colour__w=np.ones(3))
            (d / "thresholds.json").write_text('{"probe": "nope"}')
            bad = [d / "empty.csv", d / "features_q9.npz", d / "bad.json", d / "missing.pt", d / "probe_badshape.npz",
                   d / "fusion_nan.npz", d / "calibrator_broken.npz", d / "thresholds.json"]
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_outputs.py"), *map(str, bad)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertEqual(r.stdout.count("PROBLEM"), len(bad), r.stdout)


if __name__ == "__main__":
    unittest.main()
