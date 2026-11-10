"""Project-wide constants. Everything that a report has to state lives here.

Student note: keeping every "magic number" in one file means the report can
quote it, and the examiner can check it, without reading the pipeline code.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. Override with environment variables on Kaggle / Colab.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("LV_DATA_ROOT", PROJECT_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("LV_OUTPUT_ROOT", PROJECT_ROOT / "outputs"))

RAW_DIR = DATA_ROOT / "raw"              # abo-listings.tar, abo-images-small.tar
PROCESSED_DIR = DATA_ROOT / "processed"  # clean metadata + split tables
FAKES_DIR = DATA_ROOT / "fakes"          # our generated fakes
FORENSICS_DIR = DATA_ROOT / "forensics"  # streamed CommunityForensics subset
PHONE_DIR = DATA_ROOT / "phone"          # own phone photos (+ fakes made from them)
FEATURES_DIR = DATA_ROOT / "features"    # cached CLIP / DCT features
HAND_LABELS_DIR = DATA_ROOT / "hand_labels"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
RESULTS_DIR = OUTPUT_ROOT / "results"
DEMO_BUNDLE_DIR = OUTPUT_ROOT / "demo_bundle"

# ---------------------------------------------------------------------------
# ABO download locations (public S3 bucket, CC BY 4.0).
# ---------------------------------------------------------------------------
ABO_LISTINGS_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar"
ABO_IMAGES_SMALL_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar"

# ---------------------------------------------------------------------------
# Image canonicalisation. EVERY image (real, fake, phone, streamed) goes
# through the same path before any feature is computed:
#   resize longest side to CANON_MAX_SIDE  ->  JPEG encode at CANON_JPEG_QUALITY
# Why: ABO "small" images are already <=256 px JPEGs. If fakes were kept as
# 512 px PNGs the detector could learn "PNG vs JPEG" or "512 vs 256" instead of
# "real vs fake". Same treatment for both sides removes that shortcut.
# Storage of fakes on disk is a separate matter: ABO reals are NOT q92 JPEGs
# (luma tables ~17-30, varying per image), so data/fakes.py saves each fake with
# its paired real's quantisation tables; CANON_JPEG_QUALITY is only the fallback
# there and the in-memory canonicalisation quality here.
# ---------------------------------------------------------------------------
CANON_MAX_SIDE = 256
CANON_JPEG_QUALITY = 92

# JPEG qualities for the robustness curve (quality 90 -> 30).
JPEG_QUALITIES = [90, 70, 50, 30]

# DCT feature geometry.
DCT_SIZE = 256        # grayscale image is resized to DCT_SIZE x DCT_SIZE
DCT_MAP_SIZE = 64     # log-DCT map is average-pooled to this size for storage
DCT_RADIAL_BINS = 128

# ---------------------------------------------------------------------------
# Models.
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"
CLIP_FEATURE_DIM = 768

VLM_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

GENERATORS = {
    # key -> (HF model id, licence)
    "sdxl-turbo": ("stabilityai/sdxl-turbo", "Stability AI Community License"),
    "sd15": ("stable-diffusion-v1-5/stable-diffusion-v1-5", "CreativeML OpenRAIL-M"),
    # Third family: UNet + MoVQ decoder (not the SD VAE), image-prior conditioning.
    # Chosen because its decoder fingerprint differs from both SD models; fits a T4 in fp16.
    "kandinsky22": ("kandinsky-community/kandinsky-2-2-decoder", "Apache 2.0"),
}
SEEN_GENERATORS = ["sdxl-turbo", "sd15"]             # the `main` tag trains on these
ALL_GENERATORS = ["sdxl-turbo", "sd15", "kandinsky22"]  # the `all3` tag trains on these; kandinsky22 is unseen for `main`

# Verified against the dataset card on 2026-11-09: split 'train', image bytes in
# 'image_data', generator in 'model_name', label 1 = fake / 0 = real, CC BY-NC-SA 4.0.
# (The full 'OwensLab/CommunityForensics' has different splits; do not use it.)
COMMUNITY_FORENSICS_ID = "OwensLab/CommunityForensics-Small"
COMMUNITY_FORENSICS_LICENCE = "CC BY-NC-SA 4.0"

# ---------------------------------------------------------------------------
# Attribute vocabularies for the consistency branch.
# A claim is only checked if its value maps into one of these lists.
# ---------------------------------------------------------------------------
COLOUR_VOCAB = [
    "black", "white", "grey", "brown", "beige", "red",
    "orange", "yellow", "green", "blue", "purple", "pink",
]

MATERIAL_VOCAB = [
    "wood", "metal", "plastic", "glass", "fabric", "leather", "ceramic", "stone",
]

# Item counts we try to verify. Above 6, a 500M VLM cannot count reliably and
# neither can CLIP; those listings keep their count in the table but are
# flagged count_verifiable = False.
COUNT_MIN, COUNT_MAX = 1, 6

# Fixed split fractions (by product family, see data/splits.py).
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.10, "test": 0.20}
SPLIT_SEED = 2026

# Fake generation defaults.
FAKE_CHUNK_SIZE = 500
FAKE_IMAGE_SIZE = 512
# Every N-th fake also keeps a lossless 256 px PNG twin (scripts/14_png_diagnostic.py
# measures how much the JPEG storage step costs the frequency cue). 0 disables.
FAKE_KEEP_PNG_EVERY = 10
