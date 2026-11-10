# Listing Verifier: is this product photo real, and does the description match it?

DLVA 2026 course project (IISc). A buyer-side verifier for marketplace listings built on **one frozen
OpenAI CLIP ViT-L/14 backbone** with two small heads:

1. **Authenticity**: linear probe on CLIP features (UnivFD recipe) + a small head on the log-DCT
   spectrum (AIDE cue), late-fused, trained with JPEG-70/50 compression augmentation. Fakes are generated
   *from ABO itself* with SDXL-Turbo, SD 1.5 and Kandinsky 2.2 (the third family is held out for the `main`
   model and added in `all3`); tested on seen generators, in-domain unseen generators, a streamed
   CommunityForensics subset, own phone photos, and under JPEG 90→30. A PNG-twin diagnostic measures what
   JPEG storage costs the frequency cue.
2. **Consistency**: CLIP zero-shot contrast + SmolVLM-500M multiple-choice answers per claim (colour,
   count; material as the stretch goal), calibrated on synthetic mismatch pairs, sanity-checked on 100
   hand-labelled listings.

Design choices and the reasons behind them are in `listing_integrity_risk_assessment.md`; the one-page
proposal is `Vamsi_Krishna_DLVA_Project_Proposal.pdf`. Two of that document's phrasings are amended by the
implementation: splits are family-disjoint with brand overlap *reported* (a strict brand split is unusable on
ABO), and "one backbone" means one shared frozen CLIP for both heads plus SmolVLM as an auxiliary oracle.

## Layout

```
listing_verifier/            the package
  config.py                  every constant the report has to state
  data/abo_metadata.py       ABO loading + cleaning (12 colours, 8 materials, strict counts)
  data/splits.py             family-based, hash-deterministic train/val/test split
  data/index.py              one table for every image (reals, fakes, forensics, phone)
  data/fakes.py              chunked, resumable fake generation (diffusers)
  data/forensics_stream.py   stream a CommunityForensics subset (never the 260 GB)
  data/mismatch.py           synthetic mismatch pairs
  features/image_ops.py      canonicalise (256 px, in-memory JPEG 92 for features; stored fakes copy the paired real's JPEG tables) + JPEG degradation
  features/dct.py            log-DCT map + radial profile (numpy only)
  features/clip_backbone.py  frozen CLIP via open_clip
  features/vlm.py            SmolVLM multiple-choice via next-token logits
  features/extract.py        resumable feature cache (CLIP + DCT per quality)
  models/probe.py            linear probe;  models/spectrum_head.py  small CNN;  models/fusion.py
  models/consistency.py      zero-shot prompts, claim features, calibrator
  models/baselines.py        UnivFD released weights as a baseline
  eval/metrics.py            AUROC, TPR@FPR, ECE, per-attribute F1, bootstrap CI
  eval/authenticity.py       scoring + evaluation helpers;  eval/plots.py
  demo/verifier.py           ListingVerifier (CPU, offline);  demo/server.py  FastAPI
scripts/01..13_*.py          the pipeline, one step per script (run in order)
notebooks/kaggle_0*.ipynb    thin Kaggle runbooks that call the scripts (regenerate: make notebooks)
web/                         React + Tailwind front-end (prebuilt in web/dist)
tests/                       unit + smoke tests (python -m unittest discover -s tests -t .)
docs/                        TRAINING_RUNBOOK.md, DEMO.md, PHONE_SET.md, REPORT_OUTLINE.md
```

## Quick start

```bash
pip install -r requirements.txt               # laptop; Kaggle uses requirements-kaggle.txt
python -m unittest discover -s tests -t .     # 31 tests, no downloads, ~30 s
```
Training: follow `docs/TRAINING_RUNBOOK.md` (Kaggle T4 x2, ~5–6 session-hours total; the generation,
extraction and VLM scripts take `--gpus auto` and shard across both cards).
Demo: `docs/DEMO.md` → `python -m listing_verifier.demo.server --bundle outputs/demo_bundle --offline`.

## The pipeline in one screen

```
01_download_abo      ABO tars (CC BY 4.0)
02_prepare_abo       clean labels, family splits, working subset        -> data/processed/abo_clean.csv
03_generate_fakes    SDXL-Turbo t2i/i2i, SD1.5 i2i, chunks of 500       -> data/fakes/<gen>/<mode>/
04_stream_forensics  ~3000 CommunityForensics images, many generators    -> data/forensics/
05_build_index       one table                                          -> data/processed/index.csv
06_extract_features  CLIP + log-DCT at q=none,90,70,50,30               -> data/features/features_q*.npz
07_train_authenticity  probe, spectrum head, fusion, thresholds          -> outputs/checkpoints/<tag>/
08_eval_authenticity   seen/unseen, JPEG curve, per-mode, baseline       -> outputs/results/<tag>/
09_run_consistency     CLIP + SmolVLM option probabilities (cached)      -> data/features/consistency_probs.csv
10_train_eval_consistency  calibrator + metrics + reliability plots     -> outputs/results/consistency/
11_make_handlabel_sheet    100-listing sheet + contact sheet             -> data/hand_labels/
12_build_demo_bundle       checkpoints + 12 pre-tested examples          -> outputs/demo_bundle/
13_verify_cli              one image from the command line (rehearsal)
14_png_diagnostic          storage cost of the frequency cue (JPEG vs lossless twins)   -> outputs/results/<tag>/png_diagnostic.json
```

## Non-negotiables (also in CLAUDE.md)

* Split by product family, never by listing. Fakes inherit the split of their source listing.
* Every image goes through `canonicalise()` before any feature is computed, fakes are cropped to their source's aspect ratio, and `05_build_index.py` runs a nuisance-only audit (geometry, size, JPEG tables) that must stay near chance.
* Validation is split by family into a selection half and a calibration half; thresholds are frozen there and reported as actual TPR/FPR on test.
* Labels only from structured fields; ambiguous values get no label.
* Pinned versions; `scripts/env_report.py` records what actually ran.
* Report the drops (material, cross-generator, compression) as findings.

## Licences to cite

ABO: CC BY 4.0 · SDXL-Turbo: Stability AI Non-Commercial Research Community License (`sai-nc-community`) · SD 1.5: CreativeML OpenRAIL-M ·
Kandinsky 2.2: Apache 2.0 · CommunityForensics: CC BY-NC-SA 4.0 · SmolVLM: Apache 2.0 · CLIP (OpenAI weights via open_clip): MIT.
