"""End-to-end run of the data-preparation scripts on a synthetic miniature ABO
tree: 02_prepare_abo -> 05_build_index -> 11_make_handlabel_sheet -> verify_outputs.
These scripts were never executed before this test; the Kaggle notebooks call
exactly these commands."""
import gzip
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
COLOURS = ["Black", "Navy Blue", "Red", "Walnut Brown", "White", "Multicolor", "Teal", "Olive Green"]
MATERIALS = ["Faux Leather", "Solid Wood", "Stainless Steel", "Polyester", None, "Wood and Metal"]
TYPES = ["SOFA", "CHAIR", "MUG", "LAMP", "RUG"]


def make_abo(raw: Path, n=240):
    rng = np.random.default_rng(0)
    (raw / "listings" / "metadata").mkdir(parents=True)
    (raw / "images" / "metadata").mkdir(parents=True)
    recs, imgs = [], []
    for i in range(n):
        brand = f"Brand{i % 12}"
        family = i // 3                                     # three colour variants per family
        colour = COLOURS[i % len(COLOURS)]
        name = f"{brand} Modern {TYPES[family % len(TYPES)].title()} Model{family}, {colour}"
        if i % 7 == 0:
            name = f"Set of 4 {name}"
        image_id = f"img{i:05d}"
        rel = f"{i % 10:02d}/{image_id}.jpg"
        p = raw / "images" / "small" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        w, h = (256, int(rng.integers(150, 256)))
        Image.fromarray((rng.random((h, w, 3)) * 255).astype("uint8")).save(p, quality=90)
        imgs.append({"image_id": image_id, "height": h, "width": w, "path": rel})
        desc = "Content coming soon" if i % 11 == 0 else "A sturdy, well-made item for everyday use at home."
        recs.append({"item_id": f"B{i:07d}", "main_image_id": image_id, "other_image_id": [],
                     "brand": [{"language_tag": "en_US", "value": brand}],
                     "item_name": [{"language_tag": "en_US", "value": name}],
                     "product_type": [{"value": TYPES[family % len(TYPES)]}],
                     "color": [{"language_tag": "en_US", "value": colour, "standardized_values": [colour.split()[-1]]}],
                     "material": ([{"language_tag": "en_US", "value": MATERIALS[i % len(MATERIALS)]}] if MATERIALS[i % len(MATERIALS)] else []),
                     "model_number": [{"value": f"M{family}"}],
                     "product_description": [{"language_tag": "en_US", "value": desc}],
                     "country": "US", "domain_name": "amazon.com"})
    with gzip.open(raw / "listings" / "metadata" / "listings_0.json.gz", "wt") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    pd.DataFrame(imgs).to_csv(raw / "images" / "metadata" / "images.csv.gz", index=False, compression="gzip")


class PreparePipelineTests(unittest.TestCase):
    def test_prepare_index_handlabel(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            raw = d / "raw" / "abo"
            make_abo(raw)
            env = {**os.environ, "LV_DATA_ROOT": str(d), "LV_OUTPUT_ROOT": str(d / "out")}

            def run(*args):
                r = subprocess.run([sys.executable, str(ROOT / "scripts" / args[0]), *map(str, args[1:])], env=env, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, f"{args[0]}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
                return r.stdout

            out = run("02_prepare_abo.py", "--raw-dir", raw, "--out-dir", d / "processed", "--train", 100, "--val", 30, "--test", 60, "--cap-per-type", 50, "--copy-images")
            clean = pd.read_csv(d / "processed" / "abo_clean.csv")
            rep = json.loads((d / "processed" / "cleaning_report.json").read_text())
            self.assertGreater(rep["drop_reasons"].get("placeholder_description", 0), 0)
            self.assertEqual(set(clean["split"]), {"train", "val", "test"})
            self.assertTrue(clean["colour"].notna().mean() > 0.5)
            self.assertFalse(clean["colour"].isin(["Teal", "Multicolor"]).any())
            self.assertTrue((d / "processed" / "images").exists())
            # family-based split: all variants of a family share a split
            per_family = clean.groupby("family_key")["split"].nunique()
            self.assertTrue((per_family == 1).all())

            out = run("05_build_index.py", "--no-audit")   # no fakes yet, like the first pilot step
            idx = pd.read_csv(d / "processed" / "index.csv")
            self.assertEqual(len(idx), len(clean))
            self.assertTrue(idx["path"].map(os.path.exists).all())
            cov = json.loads((d / "processed" / "coverage.json").read_text())
            self.assertIn("train/real", cov["by_split_generator"])

            run("11_make_handlabel_sheet.py", "--n", 20)
            sheet = pd.read_csv(d / "hand_labels" / "hand_labels_todo.csv")
            self.assertTrue({"image_key", "attribute", "claimed", "human_label", "visible_count"} <= set(sheet.columns))
            html = (d / "hand_labels" / "contact_sheet.html").read_text()
            self.assertIn("data:image/jpeg;base64", html)
            self.assertNotIn("file://", html)

            run("verify_outputs.py", d / "processed" / "abo_clean.csv", d / "processed" / "index.csv", d / "processed" / "images")


if __name__ == "__main__":
    unittest.main()
