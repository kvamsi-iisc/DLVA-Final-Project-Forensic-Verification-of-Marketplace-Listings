"""FastAPI backend + static React front-end for the demo.

  python -m listing_verifier.demo.server --bundle outputs/demo_bundle --offline [--mode probe] [--no-vlm]
Then open http://127.0.0.1:8000. Everything runs on CPU; --offline sets
HF_HUB_OFFLINE=1 so no network call is ever attempted. Binds to loopback by
default; requests are bounded (20 MB, 40 MP) and claims are validated against
the vocabularies, one inference at a time.

Endpoints
  GET  /api/health                      model status
  GET  /api/examples                    pre-tested inputs (examples.csv in the bundle)
  GET  /api/examples/{filename}         the example image
  POST /api/verify  (multipart)         image, description, product_noun, jpeg_quality, claims (json)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ..config import COLOUR_VOCAB, COUNT_MAX, COUNT_MIN, DEMO_BUNDLE_DIR, MATERIAL_VOCAB, PROJECT_ROOT
from ..features.image_ops import load_rgb
from .verifier import MODES, ListingVerifier, extract_claims

WEB_DIST = PROJECT_ROOT / "web" / "dist"
MAX_UPLOAD_BYTES = 20 * 2**20       # 20 MB
MAX_PIXELS = 40_000_000             # 40 MP (a 12 MP phone photo is fine)
MAX_TEXT = 2000


def validate_claims(raw: str) -> Optional[dict]:
    """Only the three attributes, only vocabulary values. 400 on anything else."""
    if not raw:
        return None
    try:
        c = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(400, "claims must be JSON")
    if not isinstance(c, dict):
        raise HTTPException(400, "claims must be a JSON object")
    out = {}
    for k, v in c.items():
        if k == "colour" and v in COLOUR_VOCAB:
            out[k] = v
        elif k == "material" and v in MATERIAL_VOCAB:
            out[k] = v
        elif k == "count" and isinstance(v, (int, float)) and COUNT_MIN <= int(v) <= COUNT_MAX:
            out[k] = int(v)
        else:
            raise HTTPException(400, f"unsupported claim {k}={v!r}")
    return out


def create_app(bundle: Path, use_vlm: bool = True, lazy: bool = False, mode: str = "fused") -> FastAPI:
    app = FastAPI(title="Listing Verifier", version="0.1.0")
    state = {"verifier": None, "bundle": Path(bundle), "use_vlm": use_vlm, "mode": mode}
    lock = threading.Lock()   # one CPU inference at a time; the demo is single-user

    def verifier() -> ListingVerifier:
        if state["verifier"] is None:
            state["verifier"] = ListingVerifier(state["bundle"], device="cpu", use_vlm=state["use_vlm"], mode=state["mode"])
        return state["verifier"]

    if not lazy:
        verifier()

    @app.get("/api/health")
    def health():
        v = state["verifier"]
        return {"ok": True, "models_loaded": v is not None, "vlm": bool(v and v.vlm is not None),
                "calibrator": bool(v and v.calibrator is not None), "calibrator_file": getattr(v, "calibrator_file", None),
                "mode": state["mode"], "bundle": str(state["bundle"])}

    @app.get("/api/examples")
    def examples():
        csv = state["bundle"] / "examples" / "examples.csv"
        if not csv.exists():
            return []
        df = pd.read_csv(csv).fillna("")
        return json.loads(df.to_json(orient="records"))

    @app.get("/api/examples/{filename}")
    def example_image(filename: str):
        p = (state["bundle"] / "examples" / Path(filename).name)
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.post("/api/claims")
    def claims(description: str = Form("")):
        return extract_claims(description[:MAX_TEXT])

    @app.post("/api/verify")
    async def verify(image: UploadFile = File(...), description: str = Form(""), product_noun: str = Form(""),
                     jpeg_quality: Optional[int] = Form(None), claims: str = Form(""), mode: str = Form("")):
        data = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"image larger than {MAX_UPLOAD_BYTES // 2**20} MB")
        try:
            img = load_rgb(bytes(data))        # the SAME decoder as training (EXIF, alpha)
        except Exception:
            raise HTTPException(400, "not an image")
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(413, "image has too many pixels")
        c = validate_claims(claims)
        q = int(jpeg_quality) if jpeg_quality else None
        if q is not None:
            if q >= 95:
                q = None
            elif not 30 <= q < 95:
                raise HTTPException(400, "jpeg_quality must be between 30 and 95")
        if mode and mode not in MODES:
            raise HTTPException(400, f"mode must be one of {MODES}")

        def work():
            with lock:
                return verifier().verify(img, description=description[:MAX_TEXT], product_noun=(product_noun or "")[:100] or None,
                                         claims=c, jpeg_quality=q, mode=mode or None)
        res = await asyncio.get_running_loop().run_in_executor(None, work)   # keep the event loop responsive
        return JSONResponse(res)

    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
    else:
        @app.get("/")
        def no_ui():
            return {"message": "web/dist not built. Run `cd web && npm install && npm run build`, or use the /api endpoints."}
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(DEMO_BUNDLE_DIR))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--offline", action="store_true", help="set HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1")
    ap.add_argument("--lazy", action="store_true", help="load models on first request instead of at start")
    ap.add_argument("--mode", default="fused", choices=list(MODES), help="verdict from fused heads or the probe alone")
    args = ap.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import uvicorn
    app = create_app(Path(args.bundle), use_vlm=not args.no_vlm, lazy=args.lazy, mode=args.mode)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
