"""Fakes must be stored like their paired reals: same JPEG quantisation tables and
chroma subsampling. The nuisance audit found AUROC 0.999 from the quantisation
table alone when fakes were saved at a fixed quality 92 (2026-11-10)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from listing_verifier.data.fakes import generate_chunk, jpeg_params_like, save_fake, storage_tag


def _noise(w=256, h=200, seed=0):
    return Image.fromarray((np.random.default_rng(seed).random((h, w, 3)) * 255).astype("uint8"))


class _StubGenerator:
    generator = "sdxl-turbo"      # effective_steps() knows only real generator names
    licence = "none"

    def text_to_image(self, prompt, seed):
        return _noise(512, 512, seed)

    def image_to_image(self, prompt, init, strength, seed):
        return _noise(512, 512, seed)


class FakeStorageTests(unittest.TestCase):
    def test_save_fake_copies_real_tables_and_subsampling(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            real = d / "real.jpg"
            _noise().save(real, format="JPEG", quality=60, subsampling=2)
            fake = d / "fake.jpg"
            params = save_fake(_noise(512, 512, 1), fake, aspect=(256, 200), like=real)
            with Image.open(real) as r, Image.open(fake) as f:
                self.assertEqual({k: list(v) for k, v in r.quantization.items()}, {k: list(v) for k, v in f.quantization.items()})
                self.assertEqual(f.size, (256, 200))
            self.assertTrue(storage_tag(params).startswith("matched:"))
            # a quality-92 save would differ from the quality-60 real
            fake92 = d / "fake92.jpg"
            save_fake(_noise(512, 512, 1), fake92, aspect=(256, 200), like=None)
            with Image.open(real) as r, Image.open(fake92) as f:
                self.assertNotEqual(list(r.quantization[0]), list(f.quantization[0]))

    def test_fallback_when_source_is_not_jpeg(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            png = d / "phone.png"
            _noise().save(png, format="PNG")
            self.assertEqual(storage_tag(jpeg_params_like(png)), "q92")
            self.assertEqual(storage_tag(jpeg_params_like(None)), "q92")
            self.assertEqual(storage_tag(jpeg_params_like(d / "missing.jpg")), "q92")

    def test_generate_chunk_records_storage(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = []
            for i in range(3):
                src = d / f"src{i}.jpg"
                _noise(256, 180 + i, i).save(src, format="JPEG", quality=65 + 5 * i)
                rows.append({"item_id": f"B{i}", "item_name": f"item {i}", "colour": "red", "material": None,
                             "product_noun": "mug", "source_path": str(src), "aspect_w": 256, "aspect_h": 180 + i})
            csv = generate_chunk(_StubGenerator(), pd.DataFrame(rows), "t2i", 0.0, d / "out", 0, keep_png_every=2)
            df = pd.read_csv(csv)
            self.assertEqual(len(df), 3)
            self.assertTrue(df["storage"].str.startswith("matched:").all())
            self.assertEqual(df["storage"].nunique(), 3)       # one set of tables per source quality
            for _, r in df.iterrows():
                with Image.open(r["source_path"]) as s, Image.open(r["path"]) as f:
                    self.assertEqual(list(s.quantization[0]), list(f.quantization[0]))
                    self.assertEqual(f.size, s.size)
            # resuming describes what is on disk instead of regenerating
            df2 = pd.read_csv(generate_chunk(_StubGenerator(), pd.DataFrame(rows), "t2i", 0.0, d / "out", 0, keep_png_every=2))
            self.assertEqual(list(df2["storage"]), list(df["storage"]))


if __name__ == "__main__":
    unittest.main()
