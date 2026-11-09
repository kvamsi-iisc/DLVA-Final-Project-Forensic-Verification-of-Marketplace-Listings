# Demo: CPU, offline, pre-tested inputs

Architecture: FastAPI backend (`listing_verifier/demo/server.py`) that loads the trained heads from
`outputs/demo_bundle/`, and a prebuilt React + Tailwind single-page app (`web/dist`, shadcn/ui-style
design tokens, light/dark, responsive) served by the same process. No Node needed on the demo laptop.

## One-time preparation (needs internet once)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# fetch CLIP ViT-L/14 (~1.7 GB) and SmolVLM-500M (~1 GB) into the HF cache
python scripts/12_build_demo_bundle.py --tag main --prefetch     # or just run the server once online
```

If you prefer a self-contained folder, set `HF_HOME=/path/to/hf_cache` before the prefetch and again
when launching; copy that folder along with the project.

## Launch (offline)

```bash
python -m listing_verifier.demo.server --bundle outputs/demo_bundle --offline
# open http://127.0.0.1:8000
```

Options: `--mode probe|fused` (use the `recommended_mode` printed in `manifest.json`), `--no-vlm`
(CLIP-only consistency; the server then loads `calibrator_clip_only.npz`, which was trained without VLM
features), `--lazy` (load models on first request), `--port 8080`. The server binds to loopback, bounds
uploads (20 MB, 40 MP), validates claims against the vocabularies and runs one inference at a time.

## Rehearsal protocol (the bundle is "pre-tested" only after this)

1. `examples/expected.json` in the bundle holds the Kaggle-side outputs for every example at JPEG none/70/50/30.
2. On the laptop, from a fresh process with networking disabled, run the CLI on every example and compare:
   ```bash
   python scripts/13_verify_cli.py --image outputs/demo_bundle/examples/<file>.jpg --description "<its description>" --jpeg 50
   ```
   Verdicts must match `expected.json`; probabilities within ±0.02 (CPU vs GPU numerics).
3. Three complete runs through the UI script below after a restart, identical verdicts, zero request failures.
   Record cold start and the median / slowest warm request; the target is a warm request under 10 s.
4. Fill `visible_ok` in `examples/examples.csv`: confirm by eye that each description matches the photo
   (fakes inherit the SOURCE listing's description, so a generator may have changed the colour).
5. Freeze: no bundle or code change after the last passing rehearsal. Record a backup video.

Expected CPU latency on a laptop: CLIP ~0.5–1 s, SmolVLM ~1–3 s per claim. Warm up with one example
before presenting.

## Script for the 5-minute demo

1. Pick a **real** example from the gallery. Verify. Point at the three meters: fused clear, probe clear,
   spectrum clear. Claims match.
2. Pick an **i2i fake** of the same category. Verify. Fused flagged. Explain the two branches.
3. Drag the **JPEG slider** to 50, verify again, and describe what the *measured* curve in
   `outputs/results/main/jpeg_curve.png` showed. If the frequency head degraded more than the probe, say
   so; if not, say that too. A single image's scores need not move the way the aggregate AUROC did.
4. Edit the description to claim a wrong colour ("blue" on a brown sofa). Verify: mismatch with
   calibrated probability; open "details" to show CLIP and SmolVLM distributions.
5. Optional: a **phone photo** example to show the realism split.

Every input in the gallery was scored the day before; nothing is generated live and nothing is fetched.

## Rebuilding the front-end (only if you change `web/src`)

```bash
cd web && npm install && npm run build      # Node 18+; output goes to web/dist
```
Dev mode with hot reload: `npm run dev` (proxies `/api` to the FastAPI server on :8000).
