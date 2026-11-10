"""Load and clean ABO listing metadata.

ABO ships 16 gzipped JSON-lines files (listings_0.json.gz ... listings_f.json.gz)
plus images/metadata/images.csv.gz that maps image_id -> relative path.

Ground truth for the consistency branch comes ONLY from structured fields:
    color, material, item_name ("Pack of 2" style count pattern).
We never parse product_description for labels. The description text is only
used to drop placeholder listings ("Content coming soon").

Student intuition for the cleaning rules: if a human looking at the raw string
would hesitate ("Multicolor Yellow Accent", "Teal", "Gold"), we do not guess;
the listing simply gets no label for that attribute and is not used to
evaluate it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..config import COLOUR_VOCAB, COUNT_MAX, COUNT_MIN, MATERIAL_VOCAB

# ---------------------------------------------------------------------------
# Colour normalisation
# ---------------------------------------------------------------------------
# token (lower-case, single word) -> vocab colour. Anything not here is ignored.
# Ambiguous shades (teal, turquoise, gold, silver, multicolour) are deliberately
# ABSENT so they produce no label.
COLOUR_TOKEN_MAP = {
    "black": "black", "ebony": "black", "onyx": "black", "charcoal": "black", "jet": "black",
    "white": "white", "ivory": "white", "cream": "white", "offwhite": "white", "snow": "white",
    "grey": "grey", "gray": "grey", "graphite": "grey", "slate": "grey", "pewter": "grey",
    "ash": "grey", "gunmetal": "grey",
    "brown": "brown", "chocolate": "brown", "espresso": "brown", "walnut": "brown",
    "mocha": "brown", "coffee": "brown", "cognac": "brown", "chestnut": "brown",
    "tan": "brown", "caramel": "brown", "umber": "brown", "mahogany": "brown",
    "beige": "beige", "sand": "beige", "khaki": "beige", "taupe": "beige", "oatmeal": "beige",
    "camel": "beige", "natural": "beige", "linen": "beige", "wheat": "beige", "nude": "beige",
    "red": "red", "crimson": "red", "burgundy": "red", "maroon": "red", "scarlet": "red",
    "wine": "red", "cherry": "red", "ruby": "red", "brick": "red",
    "orange": "orange", "rust": "orange", "tangerine": "orange", "coral": "orange",
    "apricot": "orange", "peach": "orange", "terracotta": "orange",
    "yellow": "yellow", "mustard": "yellow", "lemon": "yellow", "amber": "yellow",
    "green": "green", "olive": "green", "sage": "green", "mint": "green", "lime": "green",
    "forest": "green", "emerald": "green", "jade": "green", "moss": "green", "hunter": "green",
    "blue": "blue", "navy": "blue", "cobalt": "blue", "indigo": "blue", "denim": "blue",
    "sky": "blue", "royal": "blue", "azure": "blue", "sapphire": "blue", "cerulean": "blue",
    "purple": "purple", "violet": "purple", "lavender": "purple", "plum": "purple",
    "lilac": "purple", "eggplant": "purple", "mauve": "purple", "aubergine": "purple",
    "pink": "pink", "rose": "pink", "blush": "pink", "fuchsia": "pink", "magenta": "pink",
    "salmon": "pink", "hotpink": "pink",
}
# These tokens mean "there is no single colour": the whole string is rejected.
COLOUR_REJECT_TOKENS = {"gold", "golden", "silver", "metallic", "teal", "turquoise", "aqua",
                        "multi", "multicolor", "multicolour", "multicolored", "multicoloured",
                        "assorted", "various", "mixed", "rainbow", "clear", "transparent",
                        "print", "printed", "pattern", "patterned", "floral", "striped", "stripe",
                        "plaid", "camo", "camouflage", "marble", "marbled", "ombre", "tie", "dye"}

MATERIAL_TOKEN_MAP = {
    "wood": "wood", "wooden": "wood", "oak": "wood", "walnut": "wood", "pine": "wood",
    "bamboo": "wood", "mdf": "wood", "plywood": "wood", "rubberwood": "wood", "teak": "wood",
    "birch": "wood", "maple": "wood", "acacia": "wood", "mango": "wood", "hardwood": "wood",
    "veneer": "wood", "particleboard": "wood", "rattan": "wood", "wicker": "wood",
    "metal": "metal", "steel": "metal", "stainless": "metal", "aluminum": "metal",
    "aluminium": "metal", "iron": "metal", "brass": "metal", "copper": "metal",
    "alloy": "metal", "chrome": "metal", "tin": "metal", "zinc": "metal", "titanium": "metal",
    "plastic": "plastic", "silicone": "plastic", "acrylic": "plastic", "polypropylene": "plastic",
    "abs": "plastic", "pvc": "plastic", "polycarbonate": "plastic", "resin": "plastic",
    "nylon": "plastic", "pp": "plastic", "pe": "plastic", "tpu": "plastic", "vinyl": "plastic",
    "glass": "glass", "crystal": "glass",
    "fabric": "fabric", "cotton": "fabric", "polyester": "fabric", "wool": "fabric",
    "velvet": "fabric", "textile": "fabric", "microfiber": "fabric", "microfibre": "fabric",
    "fleece": "fabric", "canvas": "fabric", "jute": "fabric", "chenille": "fabric",
    "cloth": "fabric", "spandex": "fabric", "rayon": "fabric", "viscose": "fabric",
    "satin": "fabric", "silk": "fabric", "denim": "fabric", "upholstered": "fabric",
    "leather": "leather", "suede": "leather", "leatherette": "leather",
    "ceramic": "ceramic", "porcelain": "ceramic", "stoneware": "ceramic", "earthenware": "ceramic",
    "terracotta": "ceramic", "bone": "ceramic",
    "stone": "stone", "marble": "stone", "granite": "stone", "concrete": "stone",
    "slate": "stone", "quartz": "stone", "cement": "stone",
}
# 'linen' is a colour AND a fabric; in the material field it means fabric.
MATERIAL_TOKEN_MAP["linen"] = "fabric"

PLACEHOLDER_PATTERNS = [
    r"content\s+coming\s+soon", r"description\s+coming\s+soon", r"coming\s+soon",
    r"^\s*n/?a\s*$", r"^\s*none\s*$", r"^\s*-\s*$",
]
_PLACEHOLDER_RE = re.compile("|".join(PLACEHOLDER_PATTERNS), re.IGNORECASE)

_TOKEN_RE = re.compile(r"[a-z]+")


def _tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower().replace("-", " "))


def normalise_colour(raw: Optional[str], standardized: Optional[Iterable[str]] = None) -> Optional[str]:
    """Map a raw ABO colour string to the 12-colour vocab, or None if ambiguous.

    Rule: collect every vocab colour mentioned; accept only if exactly ONE
    distinct vocab colour appears and no reject token is present.
    'Navy Blue' -> blue (both tokens map to blue). 'Black/White' -> None.
    """
    candidates = []
    if standardized:
        candidates.extend(str(s) for s in standardized if s)
    if raw:
        candidates.append(str(raw))
    found: set[str] = set()
    for cand in candidates:
        toks = _tokens(cand)
        if any(t in COLOUR_REJECT_TOKENS for t in toks):
            return None
        for t in toks:
            if t in COLOUR_TOKEN_MAP:
                found.add(COLOUR_TOKEN_MAP[t])
    if len(found) == 1:
        return next(iter(found))
    return None


def normalise_material(raw: Optional[str]) -> Optional[str]:
    """Same single-answer rule for material. 'Faux Leather' -> leather.
    'Wood and Metal' -> None (two materials: not a checkable single claim)."""
    if not raw:
        return None
    found = {MATERIAL_TOKEN_MAP[t] for t in _tokens(str(raw)) if t in MATERIAL_TOKEN_MAP}
    if len(found) == 1:
        return next(iter(found))
    return None


# Count patterns are applied to item_name only (semi-structured title, not the
# free-text description). Only explicit "N of something" phrasing is trusted.
_COUNT_PATTERNS = [
    re.compile(r"\b(?:set|pack|box|case|bundle)\s+of\s+(\d{1,2})\b", re.I),
    re.compile(r"\b(\d{1,2})\s*[- ]?(?:pack|pk|pcs|pc|pieces|piece|count|ct)\b", re.I),
]
_SET_WORDS = {"set", "pack", "pair", "pairs", "pieces", "piece", "bundle", "kit",
              "pcs", "assortment", "collection", "combo", "lot"}


def extract_count(item_name: Optional[str]):
    """Return (count, source). source in {'regex', 'inferred_single', None}.

    - 'regex': an explicit 'Set of 4' / '2-Pack' phrase (n in 2..12).
    - 'inferred_single': no set/pack/pair word in the title -> assume 1 item.
      This is an inference, not a field. The 100 hand-labelled listings exist
      partly to measure how often it is wrong.
    - None: title mentions a set/pair/kit but gives no number.
    """
    if not item_name:
        return None, None
    name = str(item_name)
    for pat in _COUNT_PATTERNS:
        m = pat.search(name)
        if m:
            n = int(m.group(1))
            if 2 <= n <= 12:
                return n, "regex"
            return None, None
    toks = set(_tokens(name))
    if toks & _SET_WORDS:
        return None, None
    return 1, "inferred_single"


def is_placeholder_description(desc: Optional[str]) -> bool:
    """True only for an ACTUAL placeholder string ("Content coming soon").

    A missing description is not a placeholder: most ABO listings have no
    product_description at all (they use bullet points), and we never read the
    description for labels. Dropping those lost ~85% of the catalogue in the
    first Kaggle run (2,001 listings kept instead of ~10,000)."""
    if desc is None or not str(desc).strip():
        return False
    return bool(_PLACEHOLDER_RE.search(str(desc)))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _first_value(field, prefer_lang: str = "en"):
    """ABO stores most fields as [{'language_tag': 'en_US', 'value': ...}, ...]."""
    if not field:
        return None
    if isinstance(field, (str, int, float)):
        return field
    if isinstance(field, dict):
        return field.get("value")
    for entry in field:
        if isinstance(entry, dict) and str(entry.get("language_tag", "")).lower().startswith(prefer_lang):
            return entry.get("value")
    entry = field[0]
    return entry.get("value") if isinstance(entry, dict) else entry


def _first_standardized(field):
    if not field or not isinstance(field, list):
        return None
    for entry in field:
        if isinstance(entry, dict) and entry.get("standardized_values"):
            return entry["standardized_values"]
    return None


def _has_english(field) -> bool:
    if not field or not isinstance(field, list):
        return False
    return any(str(e.get("language_tag", "")).lower().startswith("en") for e in field if isinstance(e, dict))


def iter_listing_records(listings_dir: Path):
    files = sorted(Path(listings_dir).glob("listings_*.json.gz"))
    if not files:
        raise FileNotFoundError(f"no listings_*.json.gz under {listings_dir}")
    for fp in files:
        with gzip.open(fp, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def record_to_row(rec: dict) -> dict:
    return {
        "item_id": rec.get("item_id"),
        "main_image_id": rec.get("main_image_id"),
        "n_other_images": len(rec.get("other_image_id") or []),
        "brand": _first_value(rec.get("brand")),
        "item_name": _first_value(rec.get("item_name")),
        "item_name_is_en": _has_english(rec.get("item_name")),
        "product_type": _first_value(rec.get("product_type")),
        "colour_raw": _first_value(rec.get("color")),
        "colour_std": "|".join(_first_standardized(rec.get("color")) or []),
        "material_raw": _first_value(rec.get("material")),
        "model_name": _first_value(rec.get("model_name")),
        "model_number": _first_value(rec.get("model_number")),
        "description": _first_value(rec.get("product_description")),
        "country": rec.get("country"),
        "domain_name": rec.get("domain_name"),
    }


def load_listings(listings_dir: Path, limit: Optional[int] = None) -> pd.DataFrame:
    rows = []
    for i, rec in enumerate(iter_listing_records(listings_dir)):
        rows.append(record_to_row(rec))
        if limit and len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def load_image_index(images_csv_gz: Path) -> pd.DataFrame:
    df = pd.read_csv(images_csv_gz)
    return df.rename(columns={"path": "image_rel_path"})[["image_id", "height", "width", "image_rel_path"]]


# ---------------------------------------------------------------------------
# Family key (used for leakage-free splitting)
# ---------------------------------------------------------------------------
_SIZE_RE = re.compile(r"\b\d+(\.\d+)?\s*(\"|''|in|inch|inches|cm|mm|ft|x|w|h|d|l)?\b", re.I)


def family_key(brand, item_name, model_number=None) -> str:
    """Group variants of one product (colour, size, pack) under one key.

    Key = normalised brand + first 4 alphabetic tokens of the item name after
    removing colour words, numbers and sizes. Example:
      'Rivet Modern Leather Sofa, 80"W, Black'  -> 'rivet|rivet modern leather sofa'
      'Rivet Modern Leather Sofa, 80"W, Brown'  -> same key -> same split.
    Coarse on purpose: over-grouping costs a little data, under-grouping leaks.
    """
    b = re.sub(r"[^a-z0-9]+", " ", str(brand or "unknown").lower()).strip()
    name = _SIZE_RE.sub(" ", str(item_name or ""))
    toks = [t for t in _tokens(name) if t not in COLOUR_TOKEN_MAP and t not in COLOUR_REJECT_TOKENS
            and t not in {"of", "the", "and", "with", "for", "in", "a", "an"}]
    head = " ".join(toks[:4]) if toks else (str(model_number or "") or "unnamed")
    return f"{b}|{head}"


def product_noun(product_type: Optional[str]) -> str:
    """'CELLULAR_PHONE_CASE' -> 'cellular phone case' (used in CLIP / VLM prompts)."""
    if not product_type:
        return "item"
    return str(product_type).lower().replace("_", " ").strip() or "item"


# ---------------------------------------------------------------------------
# Cleaning pipeline
# ---------------------------------------------------------------------------
def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised label columns and a keep flag with the reason for drops."""
    out = df.copy()
    out["colour"] = [normalise_colour(r, s.split("|") if isinstance(s, str) and s else None)
                     for r, s in zip(out["colour_raw"], out["colour_std"])]
    out["material"] = out["material_raw"].map(normalise_material)
    counts = out["item_name"].map(extract_count)
    out["count"] = [c for c, _ in counts]
    out["count_source"] = [s for _, s in counts]
    out["count_verifiable"] = out["count"].map(lambda c: c is not None and COUNT_MIN <= c <= COUNT_MAX)
    out["placeholder_desc"] = out["description"].map(is_placeholder_description)
    out["has_description"] = out["description"].map(lambda d: d is not None and bool(str(d).strip()))
    out["family_key"] = [family_key(b, n, m) for b, n, m in zip(out["brand"], out["item_name"], out["model_number"])]
    out["product_noun"] = out["product_type"].map(product_noun)

    reason = pd.Series([""] * len(out), index=out.index)
    reason[~preferred_marketplace_row(out)] = "duplicate_item_id"
    reason[(reason == "") & out["main_image_id"].isna()] = "no_main_image"
    reason[(reason == "") & (~out["item_name_is_en"])] = "non_english_name"
    reason[(reason == "") & out["placeholder_desc"]] = "placeholder_description"
    reason[(reason == "") & out["item_name"].isna()] = "no_item_name"
    out["drop_reason"] = reason
    out["keep"] = reason == ""
    return out


def preferred_marketplace_row(df: pd.DataFrame) -> pd.Series:
    """ABO lists the same item_id once per marketplace (US, GB, CA ...), often with a
    different title, image and family key. Keeping more than one row per item breaks the
    unique image key in the index and can put one product in two splits (the family key
    is title-based). True for the one row to keep per item_id: English name first, then a
    colour label, a real description, the amazon.com listing, then file order."""
    has_desc = df["description"].map(lambda d: d is not None and bool(str(d).strip())) & ~df["description"].map(is_placeholder_description)
    rank = pd.DataFrame({
        "en": df["item_name_is_en"].astype(bool).values,
        "colour": df["colour"].notna().values,
        "desc": has_desc.values,
        "com": (df["domain_name"] == "amazon.com").values,
        "pos": range(len(df)),
    }, index=df.index)
    order = rank.sort_values(["en", "colour", "desc", "com", "pos"], ascending=[False, False, False, False, True]).index
    first = ~df.loc[order, "item_id"].duplicated(keep="first")
    return first.reindex(df.index)


def cleaning_report(df: pd.DataFrame) -> dict:
    kept = df[df["keep"]]
    return {
        "n_listings": int(len(df)),
        "n_kept": int(len(kept)),
        "drop_reasons": df["drop_reason"].value_counts().to_dict(),
        "n_with_colour": int(kept["colour"].notna().sum()),
        "n_with_material": int(kept["material"].notna().sum()),
        "n_with_count_regex": int((kept["count_source"] == "regex").sum()),
        "n_with_count_inferred": int((kept["count_source"] == "inferred_single").sum()),
        "n_with_description": int(kept["has_description"].sum()),
        "n_families": int(kept["family_key"].nunique()),
        "n_brands": int(kept["brand"].nunique()),
        "colour_distribution": kept["colour"].value_counts().to_dict(),
        "material_distribution": kept["material"].value_counts().to_dict(),
    }


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)
