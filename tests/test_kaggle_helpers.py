import subprocess
import tempfile
import unittest
from pathlib import Path

from listing_verifier.kaggle import attach, dataset_root, find_source, init, input_file, is_attached, list_attached, merge_fakes, optional_attach, run


class AttachTests(unittest.TestCase):
    def _dataset(self, root: Path, slug: str, layout: str):
        base = root / slug
        if layout == "expected":
            p = base / "data" / "processed"
        elif layout == "flat":
            p = base / "processed"
        elif layout == "nested":
            p = base / slug / "data" / "processed"
        else:
            p = base
        p.mkdir(parents=True)
        (p / "abo_clean.csv").write_text("a\n1\n")
        (p / "images").mkdir()
        return base

    def test_attach_finds_marker_in_any_layout(self):
        for layout in ("expected", "flat", "nested", "root"):
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                self._dataset(d / "input", "abo-prepared", layout)
                dest = attach("abo-prepared", "data/processed", d / "work" / "data" / "processed", "abo_clean.csv", input_root=d / "input")
                self.assertTrue((dest / "abo_clean.csv").exists(), layout)
                self.assertTrue((dest / "images").exists(), layout)

    def test_new_kaggle_layout_datasets_owner_slug(self):
        """Kaggle (late 2025) mounts inputs at /kaggle/input/datasets/<owner>/<slug>."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            inp = d / "input"
            self._dataset(inp / "datasets" / "someuser", "abo-prepared", "expected")
            self.assertEqual(dataset_root("abo-prepared", inp), inp / "datasets" / "someuser" / "abo-prepared")
            self.assertTrue(is_attached("abo-prepared", inp))
            self.assertFalse(is_attached("abo-fakes-turbo-t2i", inp))
            self.assertEqual(list_attached(inp), ["datasets/someuser/abo-prepared"])
            dest = attach("abo-prepared", "data/processed", d / "work" / "data" / "processed", "abo_clean.csv", input_root=inp)
            self.assertTrue((dest / "abo_clean.csv").exists())
            self.assertIsNone(optional_attach("phone-set", "phone", d / "work" / "data" / "phone", "labels.csv", input_root=inp))
            # a single file inside a dataset, direct path or found by name under extra nesting
            w = inp / "datasets" / "someuser" / "univfd-weights" / "extra"
            w.mkdir(parents=True)
            (w / "fc_weights.pth").write_bytes(b"x")
            self.assertEqual(input_file("univfd-weights", "fc_weights.pth", inp), w / "fc_weights.pth")
            self.assertIsNone(input_file("univfd-weights", "missing.pth", inp))
            self.assertIsNone(input_file("not-attached", "fc_weights.pth", inp))
            # merge_fakes under the new layout
            f = inp / "datasets" / "someuser" / "abo-fakes-pilot" / "data" / "fakes" / "sdxl-turbo" / "t2i" / "run1"
            f.mkdir(parents=True)
            (f / "chunk_0000.csv").write_text("image_key\nk\n")
            self.assertEqual(merge_fakes(["abo-fakes-pilot"], d / "work" / "data" / "fakes", input_root=inp), 1)

    def test_missing_dataset_and_marker_messages(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "input" / "other").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError) as cm:
                find_source("abo-prepared", "abo_clean.csv", input_root=d / "input")
            self.assertIn("not attached", str(cm.exception))
            (d / "input" / "abo-prepared" / "x").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError) as cm:
                find_source("abo-prepared", "abo_clean.csv", input_root=d / "input")
            self.assertIn("layout", str(cm.exception))

    def test_stale_dest_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self._dataset(d / "input", "abo-prepared", "expected")
            stale = d / "work" / "data" / "processed"
            stale.mkdir(parents=True)
            (stale / "junk.txt").write_text("x")
            attach("abo-prepared", "data/processed", stale, "abo_clean.csv", input_root=d / "input")
            self.assertFalse((stale / "junk.txt").exists())
            self.assertTrue((stale / "abo_clean.csv").exists())

    def test_merge_fakes_layouts(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for slug, prefix, runf in (("abo-fakes-a", Path("data") / "fakes", "run1"), ("abo-fakes-b", Path("fakes"), "run2")):
                p = d / "input" / slug / prefix / "sdxl-turbo" / "t2i" / runf
                p.mkdir(parents=True)
                (p / "chunk_0000.csv").write_text("image_key\nk\n")
            n = merge_fakes(["abo-fakes-a", "abo-fakes-b", "abo-fakes-missing"], d / "work" / "data" / "fakes", input_root=d / "input")
            self.assertEqual(n, 2)

    def test_run_logs_and_raises(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            init(d)
            script = d / "s.py"
            script.write_text("import sys; print('hello'); sys.exit(int(sys.argv[1]))\n")
            run(script, "0")
            with self.assertRaises(subprocess.CalledProcessError):
                run(script, "3")
            logs = list((d / "outputs" / "logs").glob("s_*.log"))
            self.assertEqual(len(logs), 2)
            self.assertEqual(len((d / "outputs" / "logs" / "completed.jsonl").read_text().strip().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
