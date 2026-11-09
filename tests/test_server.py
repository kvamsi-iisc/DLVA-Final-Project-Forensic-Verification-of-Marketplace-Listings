"""API contract test with a stub verifier (no CLIP / VLM weights needed)."""
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

try:
    from fastapi.testclient import TestClient
    HAVE_FASTAPI = True
except ImportError:  # pragma: no cover
    HAVE_FASTAPI = False


class StubVerifier:
    def __init__(self, bundle, device="cpu", use_vlm=True, mode="fused"):
        self.vlm = object() if use_vlm else None
        self.calibrator = None
        self.calibrator_file = None
        self.mode = mode

    def verify(self, image, description="", product_noun=None, claims=None, jpeg_quality=None, mode=None):
        from listing_verifier.demo.verifier import extract_claims
        claims = claims or extract_claims(description)
        return {"authenticity": {"probe": {"p_fake": 0.2, "threshold": 0.6, "flag": False},
                                 "spectrum": {"p_fake": 0.7, "threshold": 0.65, "flag": True},
                                 "fused": {"p_fake": 0.3, "threshold": 0.55, "flag": False}, "verdict": "no generation evidence"},
                "consistency": [{"attribute": a, "claimed": v, "p_match": 0.9, "verdict": "matches", "calibration": "uncalibrated",
                                 "clip_top": str(v), "clip_p_claim": 0.8, "vlm_top": None, "vlm_p_claim": None,
                                 "options": [str(v)], "clip_dist": [1.0], "vlm_dist": None} for a, v in claims.items()],
                "claims": claims, "product_noun": product_noun or "item", "jpeg_quality": jpeg_quality,
                "image_size": list(image.size), "elapsed_s": 0.01, "vlm_used": self.vlm is not None,
                "mode": mode or self.mode, "calibrator": None}


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class ServerTests(unittest.TestCase):
    def setUp(self):
        import listing_verifier.demo.server as server
        server.ListingVerifier = StubVerifier
        self.tmp = tempfile.TemporaryDirectory()
        bundle = Path(self.tmp.name)
        (bundle / "examples").mkdir()
        Image.new("RGB", (120, 90), (120, 60, 20)).save(bundle / "examples" / "ex1.jpg")
        pd.DataFrame([{"filename": "ex1.jpg", "description": "Brown leather sofa", "product_noun": "sofa", "colour": "brown",
                       "material": "leather", "count": 1, "truth_label": 0, "generator": "real", "mode": "none",
                       "domain": "catalogue", "image_key": "abo:X"}]).to_csv(bundle / "examples" / "examples.csv", index=False)
        self.client = TestClient(server.create_app(bundle, use_vlm=False))

    def tearDown(self):
        self.tmp.cleanup()

    def test_health_and_examples(self):
        h = self.client.get("/api/health").json()
        self.assertTrue(h["models_loaded"])
        ex = self.client.get("/api/examples").json()
        self.assertEqual(ex[0]["filename"], "ex1.jpg")
        self.assertEqual(self.client.get("/api/examples/ex1.jpg").status_code, 200)
        self.assertEqual(self.client.get("/api/examples/nope.jpg").status_code, 404)

    def test_claims_and_verify(self):
        c = self.client.post("/api/claims", data={"description": "Brown leather sofa, set of 2"}).json()
        self.assertEqual(c, {"colour": "brown", "material": "leather", "count": 2})
        buf = io.BytesIO()
        Image.new("RGB", (300, 200)).save(buf, format="JPEG")
        r = self.client.post("/api/verify", files={"image": ("a.jpg", buf.getvalue(), "image/jpeg")},
                             data={"description": "red mug", "product_noun": "mug", "jpeg_quality": "50", "claims": json.dumps({"colour": "red"})})
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        self.assertEqual(j["jpeg_quality"], 50)
        self.assertEqual(j["consistency"][0]["attribute"], "colour")
        self.assertEqual(self.client.post("/api/verify", files={"image": ("a.txt", b"nope", "text/plain")}).status_code, 400)

    def test_validation(self):
        buf = io.BytesIO()
        Image.new("RGB", (30, 20)).save(buf, format="JPEG")
        img = {"image": ("a.jpg", buf.getvalue(), "image/jpeg")}
        self.assertEqual(self.client.post("/api/verify", files=img, data={"claims": "{not json"}).status_code, 400)
        self.assertEqual(self.client.post("/api/verify", files=img, data={"claims": json.dumps({"colour": "teal"})}).status_code, 400)
        self.assertEqual(self.client.post("/api/verify", files=img, data={"claims": json.dumps({"count": 9})}).status_code, 400)
        self.assertEqual(self.client.post("/api/verify", files=img, data={"jpeg_quality": "5"}).status_code, 400)
        self.assertEqual(self.client.post("/api/verify", files=img, data={"mode": "magic"}).status_code, 400)
        r = self.client.post("/api/verify", files=img, data={"mode": "probe", "jpeg_quality": "97"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["mode"], "probe")
        self.assertIsNone(r.json()["jpeg_quality"])
        self.assertEqual(self.client.post("/api/verify", files={"image": ("big.bin", b"x" * (21 * 2**20), "image/jpeg")}).status_code, 413)
        self.assertEqual(self.client.get("/api/health").json()["mode"], "fused")

    def test_static_ui_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Listing Verifier", r.text)


if __name__ == "__main__":
    unittest.main()
