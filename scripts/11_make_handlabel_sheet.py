"""Create the 100-listing hand-labelling sheet + an HTML contact sheet.

Fill `human_label` (1 = the metadata claim matches the photo, 0 = it does not,
blank = cannot tell) and, for count rows, `visible_count` (the number of units
you actually see) in data/hand_labels/hand_labels_todo.csv; save it as
hand_labels_filled.csv. The sheet asks about the METADATA value, so it measures
two things at once: how noisy ABO labels are, and how the pipeline does on
human-verified claims. The contact sheet embeds base64 thumbnails, so it opens
on any laptop after download (no Kaggle paths).

These 100 listings are a held-out audit. Do not look at detector scores for
them and do not tune any rule on them.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
from pathlib import Path

from PIL import Image

import pandas as pd

from _common import ROOT
from listing_verifier.config import HAND_LABELS_DIR, PROCESSED_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(PROCESSED_DIR / "index.csv"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    HAND_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(args.index, low_memory=False)
    pool = idx[(idx["split"] == "test") & (idx["label"] == 0) & (idx["domain"] == "catalogue")]
    pool = pool[pool["colour"].notna() | pool["material"].notna() | pool["count_verifiable"].astype(bool)]
    sample = pool.sample(min(args.n, len(pool)), random_state=args.seed)
    rows, cards = [], []
    for _, r in sample.iterrows():
        claims = []
        if pd.notna(r["colour"]):
            claims.append(("colour", r["colour"]))
        if pd.notna(r["material"]):
            claims.append(("material", r["material"]))
        if bool(r["count_verifiable"]):
            claims.append(("count", int(r["count"])))
        for a, v in claims:
            rows.append({"image_key": r["image_key"], "attribute": a, "claimed": v, "human_label": "",
                         "visible_count": "" if a == "count" else "n/a", "item_name": r["item_name"], "path": r["path"]})
        cl = "<br>".join(f"<b>{a}</b>: {html.escape(str(v))}" for a, v in claims)
        buf = io.BytesIO()
        Image.open(r["path"]).convert("RGB").resize((224, 224)).save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        cards.append(f'<div class="c"><img src="data:image/jpeg;base64,{b64}"><div><small>{html.escape(str(r["image_key"]))}</small>'
                     f'<br>{html.escape(str(r["item_name"])[:80])}<br>{cl}</div></div>')
    pd.DataFrame(rows).to_csv(HAND_LABELS_DIR / "hand_labels_todo.csv", index=False)
    (HAND_LABELS_DIR / "contact_sheet.html").write_text(
        "<html><head><style>body{font-family:sans-serif}.c{display:inline-block;width:260px;margin:8px;vertical-align:top;"
        "border:1px solid #ddd;padding:6px;border-radius:8px}img{width:100%;height:200px;object-fit:contain}</style></head><body>"
        + "".join(cards) + "</body></html>")
    print(f"[handlabel] {len(rows)} claim rows for {len(sample)} listings -> {HAND_LABELS_DIR}")


if __name__ == "__main__":
    main()
