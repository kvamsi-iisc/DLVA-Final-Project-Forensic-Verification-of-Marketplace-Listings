"""ListingVerifier: everything the demo needs, CPU-only, offline.

    v = ListingVerifier("outputs/demo_bundle")
    v.verify(pil_image, description="brown leather sofa, set of 2", product_noun="sofa", jpeg_quality=50)

Returns plain dicts so the FastAPI layer can serialise them directly.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from ..config import COLOUR_VOCAB, COUNT_MAX, COUNT_MIN, MATERIAL_VOCAB
from ..data.abo_metadata import COLOUR_REJECT_TOKENS, COLOUR_TOKEN_MAP, MATERIAL_TOKEN_MAP, extract_count
from ..eval.authenticity import load_models, score_features
from ..features.dct import spectrum_features
from ..features.image_ops import canonicalise, degrade, load_rgb, to_square_gray
from ..models.consistency import (ATTRIBUTE_OPTIONS, VLM_QUESTIONS, ClaimCalibrator, ClipZeroShot, claim_features,
                                  option_index, vlm_options)

_WORD = re.compile(r"[a-z]+")
_NEGATIONS = {"not", "no", "non", "without", "never", "isn", "isnt", "except"}


def extract_claims(description: str) -> Dict[str, object]:
    """Pull checkable claims out of a free-text description with the same
    vocabulary rules used for the ABO labels. Ambiguous -> no claim."""
    text = (description or "").lower().replace("-", " ")
    toks = _WORD.findall(text)
    claims: Dict[str, object] = {}

    def negated(i: int) -> bool:   # 'not leather', 'no red', 'non wood', 'without glass'
        return i > 0 and toks[i - 1] in _NEGATIONS

    colours = {COLOUR_TOKEN_MAP[t] for i, t in enumerate(toks) if t in COLOUR_TOKEN_MAP and not negated(i)}
    if len(colours) == 1 and not any(t in COLOUR_REJECT_TOKENS for t in toks):
        claims["colour"] = next(iter(colours))
    materials = {MATERIAL_TOKEN_MAP[t] for i, t in enumerate(toks) if t in MATERIAL_TOKEN_MAP and not negated(i)}
    if len(materials) == 1:
        claims["material"] = next(iter(materials))
    n, src = extract_count(description)
    m = re.search(r"\b(single|one)\s+(item|piece|unit)\b", text)
    if n is not None and COUNT_MIN <= n <= COUNT_MAX and (src == "regex" or m):
        claims["count"] = int(n)
    return claims


def guess_noun(description: str) -> str:
    """Last plain word of the description that is not a colour/material/number."""
    toks = [t for t in _WORD.findall((description or "").lower())
            if t not in COLOUR_TOKEN_MAP and t not in MATERIAL_TOKEN_MAP and t not in {"set", "of", "pack", "a", "an", "the", "single", "item"}]
    return toks[-1] if toks else "item"


MODES = ("fused", "probe")


class ListingVerifier:
    """mode='fused' uses probe + spectrum + fusion; mode='probe' needs only the
    linear probe (the fallback if fusion does not help on validation).
    The claim calibrator is chosen to match the VLM setting: calibrator.npz
    was trained with VLM features, calibrator_clip_only.npz without."""

    def __init__(self, bundle_dir: Path, device: str = "cpu", use_vlm: bool = True, lazy: bool = False,
                 mode: str = "fused"):
        assert mode in MODES, mode
        self.bundle = Path(bundle_dir)
        self.device = device
        self.use_vlm = use_vlm
        self.mode = mode
        self.models = load_models(self.bundle, device=device, require_fusion=(mode == "fused"))
        self.thresholds = self.models["thresholds"]
        cal = self.bundle / ("calibrator.npz" if use_vlm else "calibrator_clip_only.npz")
        self.calibrator = ClaimCalibrator.load(cal) if cal.exists() else None
        self.calibrator_file = cal.name if cal.exists() else None
        if self.calibrator is None:
            print(f"[verifier] warning: {cal.name} missing; consistency outputs will be marked 'uncalibrated'")
        self._backbone = self._zs = self._vlm = None
        if not lazy:
            self.backbone  # noqa: B018  (warm up)
            if use_vlm:
                self.vlm  # noqa: B018

    @property
    def backbone(self):
        if self._backbone is None:
            from ..features.clip_backbone import ClipBackbone
            self._backbone = ClipBackbone(device=self.device)
            self._zs = ClipZeroShot(self._backbone)
        return self._backbone

    @property
    def vlm(self):
        if self._vlm is None and self.use_vlm:
            from ..features.vlm import SmolVLM
            self._vlm = SmolVLM(device=self.device)
        return self._vlm

    # ------------------------------------------------------------------
    def authenticity(self, img: Image.Image, mode: Optional[str] = None) -> Dict:
        mode = mode or self.mode
        clip = self.backbone.encode_images([img], normalise=False)
        f = spectrum_features(to_square_gray(img))
        s = score_features(self.models, clip, f["dct_map"][None], f["dct_radial"][None])
        out = {}
        for m in ("probe", "spectrum", "fused"):
            if m not in s:
                continue
            p = float(s[m][0])
            th = self.thresholds.get(m)
            out[m] = {"p_fake": p, "threshold": th, "flag": bool(th is not None and p >= th)}
        decider = "fused" if (mode == "fused" and "fused" in out) else "probe"
        out["decided_by"] = decider
        out["verdict"] = "likely AI-generated" if out[decider]["flag"] else "no generation evidence"
        out["clip_feature"] = clip[0]
        return out

    def consistency(self, img: Image.Image, clip_feat: np.ndarray, noun: str, claims: Dict) -> List[Dict]:
        rows = []
        feat = clip_feat / np.linalg.norm(clip_feat)
        for attr, value in claims.items():
            if attr not in ATTRIBUTE_OPTIONS:
                continue
            try:
                ci = option_index(attr, value)
            except ValueError:
                continue
            cp = self._zs.option_probabilities(feat, attr, noun)
            vp = self.vlm.option_probabilities(img, VLM_QUESTIONS[attr], vlm_options(attr)) if self.vlm is not None else None
            opts = vlm_options(attr)
            if self.calibrator is not None and self.calibrator.has(attr):
                p = float(self.calibrator.predict_proba(attr, claim_features(cp, vp, ci)[None])[0])
                how = "calibrated"
            else:
                p = float(cp[ci]) if vp is None else float(0.5 * (cp[ci] + vp[ci]))
                how = "uncalibrated"
            rows.append({"attribute": attr, "claimed": value, "p_match": p, "verdict": "matches" if p >= 0.5 else "mismatch",
                         "calibration": how,
                         "clip_top": opts[int(np.argmax(cp))], "clip_p_claim": float(cp[ci]),
                         "vlm_top": opts[int(np.argmax(vp))] if vp is not None else None,
                         "vlm_p_claim": float(vp[ci]) if vp is not None else None,
                         "options": opts, "clip_dist": [float(x) for x in cp],
                         "vlm_dist": [float(x) for x in vp] if vp is not None else None})
        return rows

    def verify(self, image, description: str = "", product_noun: Optional[str] = None,
               claims: Optional[Dict] = None, jpeg_quality: Optional[int] = None, mode: Optional[str] = None) -> Dict:
        t0 = time.time()
        img = canonicalise(load_rgb(image))      # same decoder as training (EXIF, alpha)
        if jpeg_quality:
            img = degrade(img, int(jpeg_quality))
        claims = dict(claims) if claims else extract_claims(description)
        noun = (product_noun or "").strip().lower() or guess_noun(description)
        auth = self.authenticity(img, mode)
        clip_feat = auth.pop("clip_feature")
        cons = self.consistency(img, clip_feat, noun, claims)
        return {"authenticity": auth, "consistency": cons, "claims": claims, "product_noun": noun,
                "jpeg_quality": jpeg_quality, "image_size": list(img.size), "elapsed_s": round(time.time() - t0, 2),
                "vlm_used": self.vlm is not None, "mode": auth["decided_by"], "calibrator": self.calibrator_file}
