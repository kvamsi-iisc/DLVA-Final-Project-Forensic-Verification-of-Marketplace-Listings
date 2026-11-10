# Training runbook (Kaggle)

Infrastructure choice: **Kaggle notebooks** (free **T4 x2**, 30 GPU-hours per week, 12-hour
sessions, 20 GB persistent output per notebook, datasets as the hand-off between notebooks).
The quota is charged per session hour, not per card, so the three GPU notebooks shard their work
across both T4s with `--gpus auto` (one worker process per card, resumable, merged by the parent).
The same flag runs one worker on a single-GPU session, so nothing needs editing if only a P100 is free.
Colab Free is the fallback for a single step if Kaggle GPU quota runs out; nothing here depends
on Colab-only features. The six notebooks in `notebooks/` are thin wrappers around `scripts/`.

Budget: roughly **6–7 session-hours** on T4 x2 for the whole pipeline (about 11–12 on a single GPU).

## Pilot first (do this before spending the budget)

A vertical slice with real data, roughly 1.5 session-hours, that proves every hand-off and gives the first
honest numbers. Scale only after inspecting it.

1. Notebook 01 as normal → `abo-prepared`.
2. Notebook 02 with `RUN = "pilot"` (SDXL-Turbo t2i + i2i 0.5 + i2i 0.85, 300/50/100 each, ~40 min on T4 x2) → `abo-fakes-pilot`.
   The first pilot (2026-11-10) failed the shortcut audit twice and taught two things now built in: fakes must copy
   the paired real's JPEG tables (ABO is not q92), and t2i fakes do not keep the catalogue framing, so they are
   the style-mismatched control (`--mismatch-modes t2i`) and i2i (0.5, 0.85) is the training data.
   Note the printed seconds per image; multiply by the full-run counts to get YOUR timings.
3. Notebook 04 with only `abo-prepared`, `abo-fakes-pilot` and `cf-subset` attached. It detects pilot mode and
   trains `main` only. Gate: the nuisance audit AUROC is below 0.6 (the notebook asserts it), `catalogue/all_seen`
   is clearly above chance, `recommended_mode` is printed.
4. Notebook 05 (`--max-train 2500` is fine; the pilot has fewer). Gate: the VLM pilot cell shows letter-logit and
   generated answers agreeing most of the time; colour F1 on synthetic pairs is well above the 0.5 baseline.
5. Notebook 06 → `demo_bundle.zip`. Unzip on the laptop, prefetch once online, launch with `--offline`, verify a
   real, a fake, a wrong-colour claim and a count claim. This is the fallback deliverable; it exists from day 2.

If any gate fails, fix the cause (labels, shortcut audit, preprocessing) before generating more images. More
examples of the same bias only make a biased detector more confident.

## Acceptance gates (set now, before any test result is seen)

| Gate | Target |
|---|---|
| Data integrity | zero families across splits (asserted); nuisance-only AUROC < 0.6; achieved counts recorded; both labels in every split |
| Runtime | every selected generator runs on both `cuda:0` and `cuda:1` in fp16 without OOM; a resumed run continues; the bundle launches offline from a fresh process |
| Seen-family utility | `catalogue/all_seen`: AUROC ≥ 0.85 and `tpr_at_threshold` ≥ 0.60 with `fpr_at_threshold` ≤ 0.10 at the frozen threshold, with CIs |
| Colour / count | mismatch-F1 ≥ 0.80 on the hand-labelled claims (not only synthetic pairs); count reported per 1 / 2 / 3+ |
| Unseen families | no required score: report whatever is measured, with a failure analysis |
| Demo | 8–12 frozen cases; three offline rehearsals from a fresh restart with identical verdicts; warm request ≤ 10 s on the laptop |

These are planning targets for a useful demonstration, not predictions. If a gate fails, narrow the claim in
the report; do not re-select test samples.

## Week plan

| Day | Notebook | Accelerator | Time | Produces (Kaggle dataset) |
|---|---|---|---|---|
| 1 | `kaggle_01_prepare_abo` | CPU | 45 min | `abo-prepared` (10k reals, clean labels, family splits) |
| 1–2 | `kaggle_02_generate_fakes` ×5 runs | T4 x2 | 70 + 180 + 80 + 35 + 35 min (pilot rate: 1.4 s/img t2i, 1.8 s/img i2i) | `abo-fakes-turbo-t2i` (control), `abo-fakes-turbo-i2i` (0.5 + 0.85), `abo-fakes-sd15-i2i`, `abo-fakes-kand-t2i` (control), `abo-fakes-kand-i2i` |
| 2 | `kaggle_03_stream_forensics` | CPU | 30–60 min | `cf-subset` (3000 images, ~75 generators) |
| 3 | `kaggle_04_features_and_authenticity` | T4 x2 | 50 min | `lv-features-checkpoints` (features, 5 trained tags, results, PNG diagnostic) |
| 3 | `kaggle_05_consistency` | T4 x2 | 30 min | `lv-consistency` (calibrator, results) |
| 4 | hand-label 100 listings (laptop, 1 h), re-run 05 | – | – | `hand-labels` |
| 4 | phone photos (see docs/PHONE_SET.md), re-run 02 (`--source phone`) and 04 | T4 x2 | 20 min | `phone-set` |
| 5 | `kaggle_06_demo_bundle`, laptop demo rehearsal | CPU | 30 min | `demo_bundle.zip` |

## Step 0: one-time setup (15 min)

1. Create a Kaggle account, verify the phone number (required for GPU and Internet).
2. Hugging Face account; open https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 once while
   logged in and accept the OpenRAIL-M terms if it asks. Create a read token, store it as the Kaggle
   secret `HF_TOKEN` (Add-ons → Secrets).
3. Licences to record exactly (per artifact, not as a blanket statement): ABO CC BY 4.0; SDXL-Turbo
   **Stability AI Non-Commercial Research Community License** (`sai-nc-community`, research use permitted);
   SD 1.5 CreativeML OpenRAIL-M; Kandinsky 2.2 Apache 2.0; CommunityForensics-Small CC BY-NC-SA 4.0 (each
   generated image also carries its source model's licence); SmolVLM Apache 2.0; CLIP weights MIT. Trained heads
   and copied images are not automatically relicensable; keep attribution.
4. Put the code on Kaggle in one of two ways:
   * **GitHub** (recommended): push this folder to a repo, set `REPO_URL` in the setup cell of every notebook.
   * **Dataset**: zip the folder without `data/`, `outputs/`, `web/node_modules/`; upload as a Kaggle dataset
     named `listing-verifier-code`. Re-upload a version whenever the code changes.
5. Upload each notebook (`File → Import notebook`), set **Internet: on**, accelerator as in the table
   (Settings → Accelerator → **GPU T4 x2** for notebooks 02, 04, 05). The setup cell prints `visible GPUs: 2`.

## How the two-GPU mode works (so you can explain it)

| Script | What is sharded | How it resumes | Merge |
|---|---|---|---|
| `03_generate_fakes.py --gpus auto` | chunks of 500 images: worker *i* takes chunk index % 2 == *i* | finished images are skipped per chunk | none needed; each chunk has its own CSV |
| `06_extract_features.py --gpus auto` | 2000-image feature parts, same modulo rule | finished part files are skipped | parent concatenates parts per JPEG quality |
| `09_run_consistency.py --gpus auto` | every second image row → `consistency_probs.shard{i}.csv` | rows already in the main or shard file are skipped | parent de-duplicates into `consistency_probs.csv` |

Each worker is the same script with `--device cuda:i`; the parent process loads no model and only streams
the logs (prefixed `[gpu0]`, `[gpu1]`). If one worker fails, re-run the same cell: it continues where both left off.
The heads themselves (`07_train_authenticity.py`, `10_train_eval_consistency.py`) train on cached features in
seconds to minutes and stay single-device on purpose.

## Step 1: reals (`kaggle_01_prepare_abo`)

Run all. Check `cleaning_report.json`: `n_with_colour` should be a majority of `n_kept`. Check
`split_report.json`: zero leaking families (asserted), brand overlap reported. Look at the 12-image grid.
Then *Output → New Dataset* → `abo-prepared`.

Local alternative (no GPU needed): `python scripts/01_download_abo.py --images && python scripts/02_prepare_abo.py --copy-images`.

## Step 2: fakes (`kaggle_02_generate_fakes`, three commits)

Set `RUN = "turbo_t2i"`, commit (Save & Run All). Output → dataset `abo-fakes-turbo-t2i`.
Repeat with `turbo_i2i`, `sd15_i2i`, `kand_t2i` and `kand_i2i` (dataset names `abo-fakes-<run with - for _>`).
If a session dies, commit again with the same `RUN`: finished images are skipped and only the missing ones are
generated. Inspect the real/fake grid; the i2i fakes should keep the framing of the real photo (that is the point).
The t2i runs are the style-mismatched control only (domain `mismatch` in the index); generate them smaller if time
is short. The unseen-generator row needs `kand_i2i`, not `kand_t2i`.

Kandinsky 2.2 is the third generator family (different decoder from both SD models). It stays **unseen** for the
`main` model and is added only in the `all3` model, so the report can show both "unseen in-domain generator" and
"does a third family improve transfer". Every 10th fake also keeps a lossless PNG twin for the storage diagnostic.

Optional: `--strength 0.3` and `0.7` runs with `--per-split train=700,val=100,test=200` for a strength ablation.

## Step 3: unseen generators (`kaggle_03_stream_forensics`)

Run the inspect cell, confirm the column names, run the stream (stop early if it is slow; 1500 images are
still useful). Verify label 1 = fake by eye. Output → `cf-subset`.

## Step 4: authenticity (`kaggle_04_features_and_authenticity`)

Attach `abo-prepared`, the five fakes datasets, `cf-subset`, optionally `phone-set` and `univfd-weights`
(UnivFD's `fc_weights.pth` from the UniversalFakeDetect GitHub README). Run all. Five tags are trained:
`main` (two generators, JPEG-70/50 augmentation), `all3` (three generators), `turbo_only`, `main_noaug`
(ablation for the augmentation), `control_mismatch` (confound check). Read:

* `outputs/results/main/summary.csv`: AUROC / TPR@5%FPR per test set per model.
* `outputs/results/turbo_only/`: the `catalogue/sd15 [unseen_generator_same_domain]` row is the honest in-domain transfer.
* `outputs/results/main/jpeg_curve.png`: the headline robustness figure.
* `outputs/results/control_mismatch/`: the confound check for the report.
* `outputs/results/main/forensics_per_generator.csv`: which generator families transfer worst.
* `main` vs `all3` on `catalogue/kandinsky22` and `forensics/internal`: what a third family buys.
* `main` vs `main_noaug` JPEG curves: what compression augmentation buys.
* `outputs/results/main/png_diagnostic.json`: storage cost of the frequency cue (report the gap; keep JPEG storage).

Output → `lv-features-checkpoints`.

## Step 5: consistency (`kaggle_05_consistency`)

Attach `abo-prepared`, `lv-features-checkpoints`. Run all (~1 h, SmolVLM is the slow part). Read
`outputs/results/consistency/results.json`: per attribute F1 (mismatch), accuracy, ECE for calibrated vs
raw baselines. Run the hand-label cell, download the sheet, label 100 listings on the laptop (contact sheet
HTML helps), upload as dataset `hand-labels`, re-run. Output → `lv-consistency`.

## Step 6: demo (`kaggle_06_demo_bundle` + laptop)

Run, download `demo_bundle.zip`, unzip to `outputs/demo_bundle/`. On the laptop follow `docs/DEMO.md`.
Rehearse with the 12 examples and the JPEG slider; freeze the inputs one week before the presentation;
record a backup video.

## Fallbacks (decided in advance)

| Situation | Decision |
|---|---|
| Fusion does not beat the probe on the calibration half | `recommended_mode` becomes `probe`; the server runs `--mode probe`; fusion stays a reported experiment |
| SmolVLM too slow or weak | run with `--no-vlm`; the bundle contains `calibrator_clip_only.npz`, trained without VLM features, and the UI labels the mode |
| Count weak | report it as partially met with the per-count table; do not call it solved on easy examples |
| Kandinsky OOM or gated | one validated mode (i2i 0.5), or drop it and keep `turbo_only → sd15` as the unseen in-domain row |
| SD 1.5 run skipped | `main` refuses to train without every requested generator (exit 3); pass `--train-generators sdxl-turbo` for `main` and say so in the report |
| Augmentation cache missing or partial | 07 stops with exit 3 unless q70 and q50 each cover **every** training image (`train_summary.json → augmentation` must equal `n_train` for each quality); run the `--qualities 70,50 --splits train` extraction, which appends only the missing images. There is no override flag; an un-augmented model is trained explicitly with `--augment-qualities ''` |
| Generation blocked entirely | train on the streamed CF fakes as a disclosed, domain-confounded experiment (the control tag's recipe) |
| Training fails | UnivFD's released head gives a measured baseline through 08; it is not a drop-in UI model |
| Laptop fails on the day | play the recorded run and show the saved tables, labelled as recorded results |

## If something fails

| Symptom | Do |
|---|---|
| `Failed building wheel for <pkg>` during setup | the Kaggle image's Python has no wheel for that pin; every pin in `requirements-kaggle.txt` was checked for Python 3.13 wheels, so this means the image changed again: re-check the pin on PyPI and bump it, never add build tools |
| Kaggle refuses a GPU | queue is busy; retry later or run that step on Colab with the same scripts |
| Only `visible GPUs: 1` is printed | the session is T4 x1 or P100; `--gpus auto` runs one worker, just slower |
| One `[gpu1]` worker fails mid-run | re-run the same cell; both workers resume from their finished chunks/parts/shards |
| CUDA out of memory on T4 x2 | each card has 15 GB; SD 1.5 fp16 needs ~4 GB, so OOM means a leaked process: restart the session |
| SD 1.5 download 401/403 | accept the licence on the HF model page while logged in, set `HF_TOKEN` secret |
| CommunityForensics columns differ | pass `--image-col/--label-col/--generator-col`, see `--inspect` output |
| Streaming is very slow | lower `--max-total 1500`; or download the UnivFD test set instead (docs in risk assessment) |
| Probe AUROC near 1.0 on *all* sets | suspicious; check for a storage confound (PNG vs JPEG, size); every image must pass `canonicalise()` |
| Spectrum head below 0.6 even at full quality | normal for i2i at low strength; check that `dct_map` std is not ~0 (feature cache corrupted) |
| Material F1 low | expected; report as the weak spot, do not tune on test |
| Kandinsky download or OOM | ~6 GB fp16 across prior + decoder; if the T4 OOMs, run with `--per-split` halves or drop `kand_t2i` (i2i is the unseen-family row; t2i is only the control) |
| `[train] augmentation q=70: no cache` | run the third extraction command (`--qualities 70,50 --splits train`) before training |
| Demo too slow on CPU | run server with `--no-vlm` (CLIP-only consistency) or keep VLM but pre-warm with one example before presenting |
