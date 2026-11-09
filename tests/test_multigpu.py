"""Sharding and launcher logic for Kaggle T4 x2 runs (no GPU needed to test)."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import _common  # noqa: E402

from listing_verifier.features.extract import FeatureStore, extract, merge_parts, n_parts_for, pending_index  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LauncherHelpersTests(unittest.TestCase):
    def test_strip_args(self):
        argv = ["--generator", "sdxl-turbo", "--gpus", "0,1", "--persist-cmd=echo hi", "--mode", "t2i", "--device", "cuda"]
        self.assertEqual(_common.strip_args(argv, ["--gpus", "--persist-cmd", "--device"]), ["--generator", "sdxl-turbo", "--mode", "t2i"])

    def test_resolve_and_parse(self):
        self.assertEqual(_common.resolve_gpus(None), [])
        self.assertEqual(_common.resolve_gpus("0,1"), [0, 1])
        self.assertEqual(_common.parse_shard(None), (0, 1))
        self.assertEqual(_common.parse_shard("1,2"), (1, 2))

    def test_spawn_per_gpu_streams_and_reports_failures(self):
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "worker.py"
            script.write_text("import sys\nprint('args', sys.argv[1:])\nsys.exit(1 if '--shard' in sys.argv and sys.argv[sys.argv.index('--shard')+1]=='1,2' and '--fail' in sys.argv else 0)\n")
            self.assertEqual(_common.spawn_per_gpu(str(script), ["--x", "1", "--gpus", "0,1"], [0, 1], "shard", strip=[]), 0)
            self.assertEqual(_common.spawn_per_gpu(str(script), ["--fail"], [0, 1], "shard", strip=[]), 1)

    def test_chunk_modulo_split_covers_all_chunks_once(self):
        chunks = list(range(7))
        w0 = [c for c in chunks if c % 2 == 0]
        w1 = [c for c in chunks if c % 2 == 1]
        self.assertEqual(sorted(w0 + w1), chunks)
        self.assertFalse(set(w0) & set(w1))


class ShardedExtractionTests(unittest.TestCase):
    def test_parts_then_merge_equals_single_pass(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = []
            for i in range(11):
                p = d / f"{i}.jpg"
                Image.fromarray((rng.random((64, 48, 3)) * 255).astype("uint8")).save(p)
                rows.append({"image_key": f"k{i}", "path": str(p)})
            idx = pd.DataFrame(rows)
            extract(idx, None, quality=0, features_dir=d / "single", chunk_size=3, batch_size=4, verbose=False)
            # two "workers" each do every other part, nobody merges, then the parent merges
            for shard in [(0, 2), (1, 2)]:
                r = extract(idx, None, quality=0, features_dir=d / "multi", chunk_size=3, batch_size=4, verbose=False,
                            part_modulo=shard, merge=False)
                self.assertIsNone(r)
            merge_parts(0, n_parts_for(len(idx), 3), d / "multi", verbose=False)
            a, b = FeatureStore(d / "single").load(0), FeatureStore(d / "multi").load(0)
            self.assertEqual(a["pos"], b["pos"])
            self.assertTrue(np.array_equal(a["dct_radial"], b["dct_radial"]))
            self.assertFalse(list((d / "multi").glob("*.part*")))

    def test_rerun_appends_only_new_images(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = []
            for i in range(6):
                p = d / f"{i}.jpg"
                Image.fromarray((rng.random((40, 40, 3)) * 255).astype("uint8")).save(p)
                rows.append({"image_key": f"k{i}", "path": str(p)})
            idx = pd.DataFrame(rows)
            extract(idx.iloc[:4], None, quality=70, features_dir=d, chunk_size=3, batch_size=2, verbose=False)
            first = FeatureStore(d).load(70)["dct_radial"].copy()
            self.assertEqual(len(pending_index(idx, 70, d)), 2)
            extract(idx, None, quality=70, features_dir=d, chunk_size=3, batch_size=2, verbose=False)   # appends k4, k5
            z = FeatureStore(d).load(70)
            self.assertEqual(sorted(z["pos"]), [f"k{i}" for i in range(6)])
            self.assertTrue(np.array_equal(z["dct_radial"][:4], first))      # earlier rows untouched
            self.assertEqual(len(pending_index(idx, 70, d)), 0)
            self.assertIsNotNone(extract(idx, None, quality=70, features_dir=d, verbose=False))  # no-op, returns path

    def test_merge_missing_part_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                merge_parts(0, 2, Path(d), verbose=False)


class ConsistencyShardMergeTests(unittest.TestCase):
    def test_merge_shards_dedupes_and_cleans(self):
        mod = load_script("09_run_consistency")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "consistency_probs.csv"
            pd.DataFrame([{"image_key": "a", "attribute": "colour", "clip_p": "1 0", "vlm_p": ""}]).to_csv(out, index=False)
            pd.DataFrame([{"image_key": "b", "attribute": "colour", "clip_p": "0 1", "vlm_p": ""}]).to_csv(out.with_name("consistency_probs.shard0.csv"), index=False)
            pd.DataFrame([{"image_key": "a", "attribute": "colour", "clip_p": "0.5 0.5", "vlm_p": ""},
                          {"image_key": "c", "attribute": "count", "clip_p": "1", "vlm_p": ""}]).to_csv(out.with_name("consistency_probs.shard1.csv"), index=False)
            df = mod.merge_shards(out)
            self.assertEqual(len(df), 3)
            self.assertEqual(df.set_index(["image_key", "attribute"]).loc[("a", "colour"), "clip_p"], "0.5 0.5")
            self.assertFalse(list(Path(d).glob("*.shard*")))
            self.assertEqual(len(pd.read_csv(out)), 3)


if __name__ == "__main__":
    unittest.main()
