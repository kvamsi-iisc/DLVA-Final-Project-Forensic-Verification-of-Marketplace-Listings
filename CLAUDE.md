# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

Code is implemented (package `listing_verifier/`, pipeline `scripts/01..13`, Kaggle notebooks, React demo).
No model has been trained yet: training runs on Kaggle following `docs/TRAINING_RUNBOOK.md`.
The two defining documents are still the source of truth for design decisions:

- `Vamsi_Krishna_DLVA_Project_Proposal.pdf`: the one-page proposal for the DLVA course (IISc). Read it with a PDF text extractor; `pdftotext`/poppler is not installed on this machine.
- `listing_integrity_risk_assessment.md`: the ranked risk list, data-sourcing plan and licence notes. Read this before making any design or data decision; it records why choices were made.

`README.md` maps the package; `docs/` holds the runbook, demo, phone-set and report notes.

## What the project is

A buyer-side **marketplace listing verifier**: given a product photo and its description, return two judgements.

1. **Authenticity**: real capture vs diffusion-generated image.
2. **Consistency**: do checkable description claims (colour, item count; material as a stretch goal) match the image?

Both heads sit on **one frozen OpenAI CLIP ViT-L/14 backbone**. Nothing fine-tunes the backbone.

## Planned architecture

Two branches sharing the frozen backbone:

- **Authenticity branch**
  - Linear probe on frozen CLIP features (the UnivFD recipe, Ojha et al. CVPR 2023).
  - A small head on log-DCT spectra of the image (frequency cue, from AIDE, Yan et al. ICLR 2025).
  - Late fusion of the two. Report probe-only, spectrum-only and fused results separately; the expected finding is that the frequency head collapses under JPEG compression while the semantic probe holds.
  - Published UnivFD and AIDE checkpoints are **baselines and a fallback**, not the deliverable. They were trained on generic scenes, so their accuracy on product photos must be measured, never assumed.
- **Consistency branch**
  - CLIP zero-shot contrast per claim (e.g. "a red X" vs "a blue X").
  - Multiple-choice questions to a small VLM (SmolVLM-500M). Always multiple-choice, never open-ended, to limit hallucination.
  - A tiny calibration layer trained on synthetic mismatch pairs (swap the colour/count label of a correct listing).
  - 100 hand-labelled listings as a sanity check.

### Data

- **Reals**: ABO (Amazon Berkeley Objects, CC BY 4.0). Structured colour/material/item-count fields are the ground truth; do not parse free-text descriptions for labels.
- **Fakes (training)**: generated *from ABO itself* with SDXL-Turbo and SD 1.5 (text-to-image from item names, image-to-image from ABO photos at several strengths). A third family, Kandinsky 2.2 (MoVQ decoder, Apache 2.0), is generated the same way: it is **unseen** for the `main` tag and **seen** for the `all3` tag, which tests the Community Forensics claim that more generator families transfer better. This is deliberate: fakes must share framing, background and category with the reals or the probe learns "catalogue photo vs not" instead of "real vs fake".
- **Fakes (unseen-generator test)**: a streamed subset (a few thousand images) of CommunityForensics-Small from Hugging Face. Never download it in full (about 260 GB). Optional: UnivFD test set, DiffusionDB, Chameleon.
- **Phone-photo set**: 100–200 own photos of household items plus matching fakes, reported as a separate "phone" split. Do not merge it into the catalogue split.

### Evaluation (write this code before the models)

- AUROC and TPR at 5% FPR, on seen vs unseen generator families.
- AUROC vs JPEG quality 90 → 30, for probe-only, spectrum-only and fused.
- Per-attribute F1 and calibration error for consistency.
- A control experiment with style-mismatched fakes to show the confound was checked.

## Constraints that must hold in any code written here

- **Split train/val/test by ABO product family**, never by listing (brand overlap is reported, not forbidden; a strict brand split is unusable on ABO). Near-duplicate listings leak templates across splits and inflate scores.
- **Clean ABO metadata before use**: map colours to a fixed vocabulary of about 12 colours, drop listings with placeholder descriptions ("Content coming soon"), and verify only attributes present in structured fields.
- **Pin every dependency version** in `requirements.txt` from the first commit (`open_clip`, `transformers`, `diffusers`, `datasets`, `torch`). API drift in these libraries is a named project risk.
- **Generate fakes in chunks of about 500** and persist each chunk (Google Drive or Kaggle output) before continuing; Colab sessions disconnect.
- **Demo must run on CPU, offline, on pre-tested inputs.** No live generation, no fetching listings from the internet. Everything fits under 6 GB GPU memory for training and runs on a laptop CPU at inference.
- **Report honestly.** Material consistency and cross-generator transfer are expected weak spots; a measured drop is a finding, not something to hide or tune away.
- **Fakes are stored as JPEG-92 at 256 px like the reals.** Every 10th fake also keeps a lossless PNG twin so `scripts/14_png_diagnostic.py` can measure what the storage step costs the frequency cue. Report that number; do not switch to lossless fakes (storage-format confound).
- **Train with compression augmentation.** `07_train_authenticity.py` adds JPEG-70/50 features of the train images when those caches exist (`06_extract_features.py --qualities 70,50 --splits train`). Validation and the robustness curve stay un-augmented.
- Cite licences per artifact: Stability AI **Non-Commercial Research** Community License (SDXL-Turbo, `sai-nc-community`), CreativeML OpenRAIL-M (SD 1.5, one-time Hugging Face click-through), Apache 2.0 (Kandinsky 2.2, SmolVLM), CC BY-NC-SA 4.0 (CommunityForensics-Small), CC BY 4.0 (ABO), MIT (CLIP weights). All permit academic use; none makes trained heads or copied images freely relicensable.
- **Pilot before scale.** Run the `pilot` generation and take it through notebooks 04-06 before the full budget (`docs/TRAINING_RUNBOOK.md`, "Pilot first" and the acceptance gates). Validation is split into selection and calibration halves; thresholds are frozen there; report `tpr_at_threshold`/`fpr_at_threshold`, not only AUROC.
- **Shortcut audit must pass.** `05_build_index.py` writes `nuisance_audit.json`; a nuisance-only AUROC above 0.6 means geometry or storage leaks the label and training must wait.
- **Demo artifacts are scored, not assumed.** `12_build_demo_bundle.py` fails on missing heads/calibrators and writes `expected.json` for every example at every JPEG setting; `recommended_mode` from training decides fused vs probe-only.

## To-Do (later, if time permits)

- [ ] **Fine-tuning data point for the report.** Train a LoRA adapter on the last two CLIP visual blocks (rank 8, same train set, same augmentation) and report it next to the frozen probe on `catalogue/all_seen`, `catalogue/kandinsky22 [unseen]` and `forensics/internal`. Prediction to test: it wins on seen generators and loses on unseen ones (UnivFD's argument). Keep the frozen probe as the deliverable regardless; this is one ablation paragraph. Needs: a `models/lora_probe.py`, a `--tag lora` path in 07/08, roughly 20 min on a T4 x2. Decided on 2026-11-09 after the question "is pre-train + fine-tune better?"; do not start it before the main results and the demo are in.

## Commands

```bash
pip install -r requirements.txt                 # laptop (Python 3.9-3.11); Kaggle: requirements-kaggle.txt
python -m unittest discover -s tests -t .       # all tests, no model downloads
python notebooks/make_notebooks.py              # regenerate the Kaggle notebooks from scripts/
# pipeline (each script has --help; notebooks call exactly these)
python scripts/01_download_abo.py --images
python scripts/02_prepare_abo.py --copy-images
python scripts/03_generate_fakes.py --generator sdxl-turbo --mode i2i --strength 0.5 --gpus auto   # GPU; --gpus shards over T4 x2
python scripts/04_stream_forensics.py --inspect ; python scripts/04_stream_forensics.py
python scripts/05_build_index.py
python scripts/06_extract_features.py --qualities 0 --gpus auto ; python scripts/06_extract_features.py --qualities 90,70,50,30 --gpus auto
python scripts/06_extract_features.py --qualities 70,50 --splits train --gpus auto    # augmentation (appends to the q70/q50 caches)
python scripts/07_train_authenticity.py --tag main ; python scripts/08_eval_authenticity.py --tag main
python scripts/07_train_authenticity.py --tag all3 --train-generators sdxl-turbo,sd15,kandinsky22 ; python scripts/08_eval_authenticity.py --tag all3
python scripts/14_png_diagnostic.py --tag main          # storage cost of the frequency cue
python scripts/09_run_consistency.py --gpus auto ; python scripts/10_train_eval_consistency.py
python scripts/12_build_demo_bundle.py --tag main --prefetch
python -m listing_verifier.demo.server --bundle outputs/demo_bundle --offline   # http://127.0.0.1:8000
cd web && npm install && npm run build          # only when web/src changes; dist/ is committed
```

## Code conventions

- Scripts import `_common` first (adds the repo root to `sys.path`) and take every path as an argument with defaults from `config.py`; `LV_DATA_ROOT` / `LV_OUTPUT_ROOT` env vars relocate everything (Kaggle uses `/kaggle/working`).
- Feature caches are keyed by `image_key`, never by path; paths in CSVs are re-pointed by `data/index.py` when data moves between machines.
- Checkpoints are plain `.npz` / `.pt` plus a `.json` with metadata so the CPU demo needs no pickles.
- The front-end is hand-written shadcn-style components (`web/src/components/ui.tsx`); no Gradio.
- Python 3.9 compatible: no `match`, no `X | Y` unions at runtime.
- Precision comes from `features/device.py` (`pick_dtype`): fp16 on any `cuda:N`, fp32 on CPU, bf16 only when supported and requested. Never compare `device == "cuda"`.
- One image decoder for everything: `features/image_ops.load_rgb` accepts paths, bytes and PIL images (EXIF rotation, alpha on white); the API server uses it too.
- Multi-GPU is process-level fan-out (`scripts/_common.py: spawn_per_gpu`): the parent loads no model, workers are the same script with `--device cuda:i` plus a hidden shard flag (`--chunk-modulo`, `--part-modulo`, `--shard`), the parent merges. Keep new GPU-heavy scripts on that pattern instead of DataParallel.
