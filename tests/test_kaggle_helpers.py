import subprocess
import tempfile
import unittest
from pathlib import Path

from listing_verifier.kaggle import attach, find_source, init, merge_fakes, run


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
