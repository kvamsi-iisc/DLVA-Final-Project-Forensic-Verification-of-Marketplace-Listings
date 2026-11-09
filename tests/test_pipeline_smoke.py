"""End-to-end smoke test of the feature cache -> train -> evaluate scripts on
synthetic images, with no CLIP download (backbone=None gives zero CLIP
features, so only the DCT branch can learn; that is enough to exercise every
code path, including the JPEG curve and the result files)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def make_image(path: Path, fake: bool, rng):
    base = rng.random((256, 256, 3)) * 255
    if fake:  # inject a periodic high-frequency pattern (a caricature of a decoder fingerprint)
        yy, xx = np.mgrid[0:256, 0:256]
        base += 40 * np.sin(xx * 3.1)[..., None]
    Image.fromarray(np.clip(base, 0, 255).astype("uint8")).save(path, quality=92)


class PipelineSmokeTest(unittest.TestCase):
    def test_train_eval_roundtrip(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            img_dir = d / "img"
            img_dir.mkdir()
            rows = []
            n = 0
            for split, count in [("train", 120), ("val", 80), ("test", 60)]:
                for i in range(count):
                    fake = i % 2 == 1
                    p = img_dir / f"{split}_{i}.jpg"
                    make_image(p, fake, rng)
                    rows.append({"image_key": f"k{n}", "path": str(p), "domain": "catalogue", "source": "fake" if fake else "abo",
                                 "label": int(fake), "generator": "sdxl-turbo" if fake else "real", "mode": "t2i" if fake else "none",
                                 "strength": 0.0, "split": split, "item_id": f"it{n}", "family_key": f"f{n}", "product_noun": "mug",
                                 "colour": "red", "material": None, "count": 1, "count_verifiable": True, "item_name": "mug"})
                    n += 1
            idx = pd.DataFrame(rows)
            idx.to_csv(d / "index.csv", index=False)
            sys.path.insert(0, str(ROOT))
            from listing_verifier.features.extract import extract
            for q, splits in ((0, None), (50, ["test"]), (50, ["train"])):      # second q=50 call appends the train split
                sub = idx if splits is None else idx[idx["split"].isin(splits)]
                extract(sub.reset_index(drop=True), None, quality=q, features_dir=d / "feat", chunk_size=100, batch_size=16, verbose=False)
            env = {**os.environ, "LV_OUTPUT_ROOT": str(d / "out"), "LV_DATA_ROOT": str(d)}
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "07_train_authenticity.py"), "--tag", "smoke", "--index", str(d / "index.csv"),
                                "--features-dir", str(d / "feat"), "--epochs", "5", "--device", "cpu", "--augment-qualities", "50", "--train-generators", "sdxl-turbo"], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            ckpt = d / "out" / "checkpoints" / "smoke"
            for f in ["probe.npz", "spectrum.pt", "fusion.npz", "thresholds.json", "train_summary.json"]:
                self.assertTrue((ckpt / f).exists(), f)
            summary = json.loads((ckpt / "train_summary.json").read_text())
            self.assertGreater(summary["spectrum_val_auroc"], 0.9)   # the DCT head must find the planted pattern
            self.assertEqual(summary["augmentation"], {"50": 120})         # JPEG-50 covers every one of the 120 train images
            self.assertEqual(summary["augmentation"]["50"], summary["n_train"])
            self.assertEqual(summary["n_train_with_augmentation"], 240)
            self.assertIn(summary["recommended_mode"], ("fused", "probe"))
            self.assertTrue((ckpt / "train_keys.txt").exists())
            self.assertGreater(summary["n_val_select"], 10)
            self.assertGreater(summary["n_val_calib"], 10)
            # strict data requirements: a generator with no train fakes, or a missing augmentation cache, is exit 3
            base = [sys.executable, str(ROOT / "scripts" / "07_train_authenticity.py"), "--index", str(d / "index.csv"),
                    "--features-dir", str(d / "feat"), "--epochs", "1", "--device", "cpu"]
            r = subprocess.run(base + ["--tag", "x", "--train-generators", "sdxl-turbo,kandinsky22"], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 3, r.stdout[-500:])
            r = subprocess.run(base + ["--tag", "x", "--train-generators", "sdxl-turbo", "--augment-qualities", "90"], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 3, r.stdout[-500:])            # cache absent entirely
            train_rows = idx[idx["split"] == "train"]
            extract(train_rows.head(60).reset_index(drop=True), None, quality=70, features_dir=d / "feat", chunk_size=100, batch_size=16, verbose=False)
            r = subprocess.run(base + ["--tag", "x", "--train-generators", "sdxl-turbo", "--augment-qualities", "70"], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 3, r.stdout[-500:])            # cache covers only 60/120 train images: refused
            self.assertIn("60/120", r.stdout)
            extract(train_rows.reset_index(drop=True), None, quality=70, features_dir=d / "feat", chunk_size=100, batch_size=16, verbose=False)
            r = subprocess.run(base + ["--tag", "x", "--train-generators", "sdxl-turbo", "--augment-qualities", "70"], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])   # appended to full coverage: accepted
            self.assertEqual(json.loads((d / "out" / "checkpoints" / "x" / "train_summary.json").read_text())["augmentation"], {"70": 120})
            self.assertTrue((d / "out" / "checkpoints" / "x" / "invocation.json").exists())
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "08_eval_authenticity.py"), "--tag", "smoke", "--index", str(d / "index.csv"),
                                "--features-dir", str(d / "feat"), "--device", "cpu"], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            res_dir = d / "out" / "results" / "smoke"
            self.assertTrue((res_dir / "summary.csv").exists())
            self.assertTrue((res_dir / "jpeg_curve.png").exists())
            curve = pd.read_csv(res_dir / "jpeg_curve.csv", index_col=0)
            self.assertEqual(sorted(curve.index.tolist()), [0, 50])      # q70 holds train images only: excluded from the curve
            self.assertIn("quality 70", r.stdout)
            preds = pd.read_csv(res_dir / "predictions.csv")
            self.assertTrue({"image_key", "label", "test_set", "quality", "probe", "spectrum", "fused"} <= set(preds.columns))
            self.assertGreater(len(preds), 100)
            results = json.loads((res_dir / "results.json").read_text())
            self.assertIn("catalogue/all_seen", results)
            self.assertGreater(results["catalogue/all_seen"]["spectrum"]["auroc"], 0.9)
            self.assertIn("tpr_at_threshold", results["catalogue/all_seen"]["spectrum"])
            self.assertIn("fp", results["catalogue/all_seen"]["spectrum"])
            # probe-only loading must work without touching spectrum/fusion
            from listing_verifier.eval.authenticity import load_models, score_features
            m = load_models(ckpt, require_fusion=False)
            self.assertIn("fusion", m)
            import shutil
            shutil.copy(ckpt / "probe.npz", d / "probe.npz")
            probe_only = load_models(d, require_fusion=False)
            self.assertNotIn("fusion", probe_only)
            self.assertEqual(set(score_features(probe_only, np.zeros((2, 768), np.float32), None, None)), {"probe_logit", "probe"})
            with self.assertRaises(FileNotFoundError):
                load_models(d, require_fusion=True)


if __name__ == "__main__":
    unittest.main()
