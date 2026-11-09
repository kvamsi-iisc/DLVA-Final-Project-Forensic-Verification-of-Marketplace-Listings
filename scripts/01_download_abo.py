"""Download and extract ABO listings (83 MB) and optionally small images (3 GB).

Result layout:
  data/raw/abo/listings/metadata/listings_*.json.gz
  data/raw/abo/images/metadata/images.csv.gz
  data/raw/abo/images/small/<ab>/<image_id>.jpg
On Kaggle, point --raw-dir at /kaggle/working/data/raw (or attach an existing
ABO Kaggle dataset and pass --raw-dir to it; both tars are public, CC BY 4.0).
"""
from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

from _common import ROOT
from listing_verifier.config import ABO_IMAGES_SMALL_URL, ABO_LISTINGS_URL, RAW_DIR


def download(url: str, dest: Path):
    if dest.exists():
        print(f"[skip] {dest} exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}")

    def hook(count, block, total):
        done = count * block
        if total > 0 and count % 200 == 0:
            print(f"  {done / 2**20:8.0f} / {total / 2**20:.0f} MB", end="\r")

    tmp = dest.with_suffix(dest.suffix + ".part")      # download to a temp name, rename when complete
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    tmp.rename(dest)


def extract(tar_path: Path, out_dir: Path, marker: str):
    if (out_dir / marker).exists():
        print(f"[skip] {out_dir / marker} exists")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {tar_path} -> {out_dir}")
    with tarfile.open(tar_path) as tf:
        members = []
        for m in tf.getmembers():   # reject path traversal, absolute paths and links (Python 3.9 has no 'data' filter)
            if m.name.startswith("/") or ".." in Path(m.name).parts or m.issym() or m.islnk():
                raise RuntimeError(f"refusing unsafe archive member: {m.name}")
            members.append(m)
        tf.extractall(out_dir, members=members)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--images", action="store_true", help="also fetch abo-images-small.tar (~3 GB)")
    ap.add_argument("--keep-tars", action="store_true")
    args = ap.parse_args()
    raw = Path(args.raw_dir)
    abo = raw / "abo"
    lt = raw / "abo-listings.tar"
    download(ABO_LISTINGS_URL, lt)
    extract(lt, abo, "listings/metadata")
    if args.images:
        it = raw / "abo-images-small.tar"
        download(ABO_IMAGES_SMALL_URL, it)
        extract(it, abo, "images/metadata/images.csv.gz")
        if not args.keep_tars:
            it.unlink(missing_ok=True)
    if not args.keep_tars:
        lt.unlink(missing_ok=True)
    print("done:", abo)


if __name__ == "__main__":
    main()
