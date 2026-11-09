"""Generate the Kaggle notebooks (run: python notebooks/make_notebooks.py).

The notebooks are deliberately thin: every cell calls a script in scripts/, so
the notebook is a runbook, not a second copy of the code. Edit the scripts,
re-run this file, re-upload.

Conventions (evaluation P0.2): scripts run from REPO with cwd=REPO and read/write
under DATA=/kaggle/working/data and OUT=/kaggle/working/outputs via the LV_*
environment variables. Inspection cells therefore ALWAYS use DATA / OUT paths,
never relative 'data/...'. Pipeline steps go through run(), which raises on a
non-zero exit so a failed script stops the notebook instead of being ignored.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP_MD = """## 0. Setup (same cell in every notebook)

**Before running:** attach the code as a Kaggle Dataset named `listing-verifier-code`
(upload the repo folder without `data/`, `outputs/`, `web/node_modules/`), **or** set `REPO_URL`
to your GitHub repo. Turn on **Internet** (pip, ABO and HF downloads). For the GPU notebooks pick
**Settings → Accelerator → GPU T4 x2**. Store your Hugging Face token as the Kaggle secret `HF_TOKEN`."""

SETUP_CODE = r'''import os, shutil, subprocess, sys, pathlib, json
REPO_URL = "https://github.com/kvamsi-iisc/DLVA-Final-Project-Forensic-Verification-of-Marketplace-Listings.git"   # used if the code dataset is not attached
WORK = pathlib.Path("/kaggle/working"); REPO = WORK / "repo"; DATA = WORK / "data"; OUT = WORK / "outputs"
_inp = pathlib.Path("/kaggle/input")   # classic /kaggle/input/<slug> or newer /kaggle/input/datasets/<owner>/<slug>
code_ds = next((p for p in [_inp / "listing-verifier-code", *sorted(_inp.glob("datasets/*/listing-verifier-code"))] if p.exists()), None)
if REPO.exists() and (REPO / ".git").exists():
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only", "-q"], check=False)     # pick up code fixes in a reused session
elif not REPO.exists():
    if code_ds is not None:
        shutil.copytree(code_ds, REPO)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
os.chdir(REPO)
os.environ["LV_DATA_ROOT"] = str(DATA); os.environ["LV_OUTPUT_ROOT"] = str(OUT)
os.environ["HF_HOME"] = str(WORK / "hf_cache")           # model weights land here (removed before publishing)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print("no HF_TOKEN secret:", type(e).__name__, "(fine unless a gated model is needed)")
print("python", sys.version.split()[0], "| code:", subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip())
pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], capture_output=True, text=True)
if pip.returncode != 0:
    print(pip.stdout[-3000:], pip.stderr[-3000:])          # show WHY before stopping (e.g. a pin without a wheel for this Python)
    raise SystemExit("pip install failed; see output above")
subprocess.run([sys.executable, "-m", "pip", "check"], check=False)
sys.path.insert(0, str(REPO))
from listing_verifier.kaggle import init, run, attach, optional_attach, merge_fakes, is_attached, input_file, list_attached   # helpers live in the repo: fixes need no re-import
WORK, REPO, DATA, OUT, LOGS = init(WORK)
run("scripts/env_report.py", "--out", OUT / "results" / "env_report.json")
import torch; print(f"visible GPUs: {torch.cuda.device_count()}")
print("attached inputs:", list_attached())
'''

PERSIST_MD = """## Persisting outputs
When this notebook finishes under **Save Version → Save & Run All (Commit)**, everything in `/kaggle/working`
except the model cache is kept as the notebook output. Then: *Output tab → "New Dataset"* and name it as stated
at the top of this notebook. The next notebook attaches that dataset. The cell below checks the artifacts this
notebook promised actually exist, then removes the model cache so the output stays small."""


def cleanup_code(required):
    req = ", ".join(f'"{r}"' for r in required)
    return f'''required = [{req}]
run("scripts/verify_outputs.py", *[WORK / r for r in required])   # exits non-zero on a missing/empty artifact
shutil.rmtree(WORK / "hf_cache", ignore_errors=True)
shutil.rmtree(REPO / "web" / "node_modules", ignore_errors=True)
print(subprocess.run(["du", "-sh", str(DATA), str(OUT)], capture_output=True, text=True).stdout)'''


def nb(title, intro_md, cells, out_name, required):
    n = nbf.v4.new_notebook()
    n["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}}
    n["cells"] = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro_md}"),
                  nbf.v4.new_markdown_cell(SETUP_MD), nbf.v4.new_code_cell(SETUP_CODE)]
    for kind, src in cells:
        n["cells"].append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    n["cells"] += [nbf.v4.new_markdown_cell(PERSIST_MD), nbf.v4.new_code_cell(cleanup_code(required))]
    nbf.write(n, HERE / out_name)
    print("wrote", out_name)


FAKE_SLUGS = ["abo-fakes-pilot", "abo-fakes-turbo-t2i", "abo-fakes-turbo-i2i", "abo-fakes-sd15-i2i",
              "abo-fakes-kand-t2i", "abo-fakes-kand-i2i"]

MERGE_FAKES = r'''merge_fakes(FAKE_SLUGS, DATA / "fakes")'''


# --------------------------------------------------------------------------- 1
nb("01 · Prepare ABO (reals + metadata + splits)",
   """**Accelerator:** none · **Runtime:** ~30–45 min (3 GB download dominates) · **Output dataset name:** `abo-prepared`

Download ABO listings + small images (CC BY 4.0), clean the metadata (12-colour vocabulary, 8 materials, strict
"Set of N" counts, placeholder descriptions dropped), assign **family-based** splits, select the working subset
(7000 / 1000 / 2000 listings) and copy only those images into `data/processed/images`.

**Student intuition:** we never split by listing. The same sofa in four colours shares a *family key*, and the key
decides the split, so no template can appear on both sides. `split_report.json` reports brand overlap (reported,
not forbidden: the risk document's brand-disjoint wording is amended in the report).""",
   [("code", 'run("scripts/01_download_abo.py", "--images", "--raw-dir", DATA / "raw")'),
    ("code", 'run("scripts/02_prepare_abo.py", "--train", 7000, "--val", 1000, "--test", 2000, "--cap-per-type", 250, "--copy-images")'),
    ("md", "### Sanity-check the cleaning. If `n_with_colour` is below ~60% of `n_kept`, inspect `colour_raw` values and extend `COLOUR_TOKEN_MAP`. Quote the ACHIEVED counts in the report, never the requested ones."),
    ("code", r'''import pandas as pd
rep = json.load(open(DATA / "processed" / "cleaning_report.json")); print(json.dumps(rep, indent=1)[:3000])
print(json.load(open(DATA / "processed" / "split_report.json")))
df = pd.read_csv(DATA / "processed" / "abo_clean.csv")
print(df["split"].value_counts()); print(df["colour"].value_counts().head(12)); print(df["product_type"].value_counts().head(15))'''),
    ("md", "### Look at 12 random listings with their labels (always look at your data)."),
    ("code", r'''import matplotlib.pyplot as plt
from PIL import Image
s = df.sample(12, random_state=1)
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for ax, (_, r) in zip(axes.ravel(), s.iterrows()):
    ax.imshow(Image.open(DATA / "processed" / "images" / r["image_rel_path"])); ax.axis("off")
    ax.set_title(f"{r['colour']} | {r['material']} | n={r['count']}\n{str(r['item_name'])[:30]}", fontsize=8)
plt.tight_layout()'''),
    ("md", "### Remove the raw 398k-image tree so the output dataset is just the working subset (~400 MB)."),
    ("code", 'shutil.rmtree(DATA / "raw", ignore_errors=True)')],
   "kaggle_01_prepare_abo.ipynb",
   required=["data/processed/abo_clean.csv", "data/processed/images", "data/processed/split_report.json"])

# --------------------------------------------------------------------------- 2
nb("02 · Generate fakes from ABO (pilot first, then SDXL-Turbo, SD 1.5, Kandinsky 2.2)",
   """**Accelerator:** GPU **T4 x2** · **Attach:** `abo-prepared` · **Output dataset name:** `abo-fakes-<run with dashes>`

Set `RUN` and commit once per run. **Do the `pilot` run first** and take it through notebooks 04 → 05 → 06
before spending the rest of the budget (runbook, "Pilot").

| RUN | generator | mode | strength | images (train/val/test) | note |
|---|---|---|---|---|---|
| `pilot` | sdxl-turbo | t2i **and** i2i 0.5 | – | 300/50/100 each | ~15 min; the vertical slice |
| `turbo_t2i` | sdxl-turbo | t2i | – | 2100/300/600 | |
| `turbo_i2i` | sdxl-turbo | i2i | 0.5 | 2100/300/600 | 2 effective denoising steps |
| `sd15_i2i`  | sd15 | i2i | 0.5 | 1400/200/400 | gated: HF_TOKEN secret |
| `kand_t2i`  | kandinsky22 | t2i | – | 700/100/200 | third family, unseen for `main` |
| `kand_i2i`  | kandinsky22 | i2i | 0.5 | 700/100/200 | |

Timings are estimates until the pilot measures them: benchmark seconds/image from the pilot log and scale.

**Why fakes from ABO itself:** fakes that differ in *style* (clutter, framing, resolution) would let the probe learn
"catalogue photo vs not" (Risk 1). Same-domain generation reduces that confound; it does not remove it, which is why
notebook 04 runs a nuisance-only audit and a style-mismatch control.

**Geometry:** the i2i source is letterboxed (never stretched) and every fake is cropped back to its source listing's
aspect ratio, so shape carries no label. Every 10th fake keeps a lossless PNG twin for the storage diagnostic.

`--gpus auto` spawns one worker per visible GPU (chunks split by index modulo). Re-running the same RUN resumes.
Each run writes to its own folder (`<generator>/<mode>/n<sizes>_seed<seed>/`), so the pilot and the full run of
the same generator never share chunk files when their datasets are merged in notebook 04.""",
   [("code", r'''RUN = "pilot"   # pilot | turbo_t2i | turbo_i2i | sd15_i2i | kand_t2i | kand_i2i
RUNS = {"turbo_t2i": [["--generator", "sdxl-turbo", "--mode", "t2i", "--per-split", "train=2100,val=300,test=600"]],
        "turbo_i2i": [["--generator", "sdxl-turbo", "--mode", "i2i", "--strength", "0.5", "--per-split", "train=2100,val=300,test=600"]],
        "sd15_i2i":  [["--generator", "sd15", "--mode", "i2i", "--strength", "0.5", "--per-split", "train=1400,val=200,test=400"]],
        "kand_t2i":  [["--generator", "kandinsky22", "--mode", "t2i", "--per-split", "train=700,val=100,test=200"]],
        "kand_i2i":  [["--generator", "kandinsky22", "--mode", "i2i", "--strength", "0.5", "--per-split", "train=700,val=100,test=200"]],
        "pilot":     [["--generator", "sdxl-turbo", "--mode", "t2i", "--per-split", "train=300,val=50,test=100"],
                      ["--generator", "sdxl-turbo", "--mode", "i2i", "--strength", "0.5", "--per-split", "train=300,val=50,test=100"]]}
attach("abo-prepared", "data/processed", DATA / "processed", must_contain="abo_clean.csv")
import time
for job in RUNS[RUN]:
    t0 = time.time(); run("scripts/03_generate_fakes.py", *job, "--chunk-size", 500, "--gpus", "auto")
    print(f"[timing] {job[1]} {job[3]}: {time.time() - t0:.0f} s wall")'''),
    ("md", "### Look at real vs fake side by side, and check failed images (black, text overlays, distortions). Log any exclusion rule BEFORE looking at detector scores."),
    ("code", r'''import glob, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
csvs = sorted(glob.glob(str(DATA / "fakes" / "**" / "chunk_*.csv"), recursive=True)); allf = pd.concat(map(pd.read_csv, csvs))
fk = allf.sample(min(6, len(allf)), random_state=0)
abo = pd.read_csv(DATA / "processed" / "abo_clean.csv").set_index("item_id")
fig, axes = plt.subplots(2, len(fk), figsize=(3 * len(fk), 6))
for i, (_, r) in enumerate(fk.iterrows()):
    axes[0, i].imshow(Image.open(DATA / "processed" / "images" / abo.loc[r["item_id"], "image_rel_path"])); axes[0, i].set_title("real", fontsize=9)
    axes[1, i].imshow(Image.open(r["path"])); axes[1, i].set_title(f"{r['generator']} {r['mode']} steps={r.get('effective_steps', '?')}", fontsize=8)
    axes[0, i].axis("off"); axes[1, i].axis("off")
plt.tight_layout(); print(len(allf), "fakes so far;", "png twins:", int(allf.get("png_twin", pd.Series(dtype=bool)).sum()))
print(allf.groupby(["generator", "mode"]).size())''')],
   "kaggle_02_generate_fakes.ipynb", required=["data/fakes"])

# --------------------------------------------------------------------------- 3
nb("03 · Stream a CommunityForensics-Small subset (unseen generators)",
   """**Accelerator:** none · **Runtime:** 20–60 min (network bound) · **Output dataset name:** `cf-subset`

We never download the 260 GB dataset. The loader streams `OwensLab/CommunityForensics-Small` (split `train`,
bytes in `image_data`, generator in `model_name`, label 1 = fake; verified against the dataset card) and keeps
explicit quotas: 1500 real + 1500 fake, at most 40 fakes per model, bounded scan. Re-running resumes from the
manifest. Licence: CC BY-NC-SA 4.0 (each generated image also carries its source model's licence).""",
   [("code", 'run("scripts/04_stream_forensics.py", "--inspect")'),
    ("md", "If the printed feature names differ from `image_data` / `label` / `model_name`, pass `--image-col/--label-col/--generator-col`. Then verify by eye that label 1 is fake."),
    ("code", 'run("scripts/04_stream_forensics.py", "--n-real", 1500, "--n-fake", 1500, "--per-generator", 40, "--max-scan", 200000)'),
    ("code", r'''import pandas as pd, matplotlib.pyplot as plt
from PIL import Image
df = pd.read_csv(DATA / "forensics" / "forensics_subset.csv"); print(df["label"].value_counts()); print(df["generator"].nunique(), "generators;", df["architecture"].value_counts().head(10) if "architecture" in df else "")
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for ax, (_, r) in zip(axes.ravel(), df.groupby("label").head(6).iterrows()):
    ax.imshow(Image.open(r["path"])); ax.set_title(f"label={r['label']} {str(r['generator'])[:18]}", fontsize=8); ax.axis("off")
plt.tight_layout()''')],
   "kaggle_03_stream_forensics.ipynb", required=["data/forensics/forensics_subset.csv"])

# --------------------------------------------------------------------------- 4
nb("04 · Features + authenticity branch (train, evaluate, audits)",
   """**Accelerator:** GPU **T4 x2** · **Attach:** `abo-prepared`, every `abo-fakes-*` you have, `cf-subset`, optional `phone-set`, `univfd-weights` · **Output dataset name:** `lv-features-checkpoints`

Index → nuisance-only shortcut audit (must be near 0.5) → CLIP + log-DCT features (canonical for all; JPEG 90/70/50/30
for evaluation splits; 70/50 for the train split as augmentation) → tags:

* `main`: SDXL-Turbo + SD 1.5 (what exists), compression augmentation; Kandinsky unseen.
* `main_noaug`, `turbo_only`, `control_mismatch` (CF-trained confound check), `all3` (adds Kandinsky; a separate, predeclared experiment; lower priority).

Validation is split by family into a selection half and a calibration half; thresholds are frozen there and
reported as actual TPR/FPR on test. Then: JPEG curve on identical images, per-generator rows, UnivFD baseline
if present, PNG storage diagnostic. On the **pilot**, only `main` is trained.""",
   [("code", 'FAKE_SLUGS = ' + repr(FAKE_SLUGS) + '\nattach("abo-prepared", "data/processed", DATA / "processed", must_contain="abo_clean.csv")\n' + MERGE_FAKES + r'''
optional_attach("cf-subset", "data/forensics", DATA / "forensics", must_contain="forensics_subset.csv")
optional_attach("phone-set", "phone", DATA / "phone", must_contain="labels.csv")
uf = input_file("univfd-weights", "fc_weights.pth")
if uf is not None:
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True); shutil.copy(uf, OUT / "checkpoints" / "univfd_fc_weights.pth")
PILOT = not is_attached("abo-fakes-turbo-t2i")   # pilot = only the small run is attached
print("PILOT mode:", PILOT)'''),
    ("code", r'''run("scripts/05_build_index.py")
audit = json.load(open(DATA / "processed" / "nuisance_audit.json")); print(json.dumps(audit, indent=1)[:1500])
assert audit.get("nuisance_only_test_auroc", 0.5) < 0.6, "geometry/storage leaks the label: fix the data before training"'''),
    ("code", r'''run("scripts/06_extract_features.py", "--qualities", "0", "--gpus", "auto")
run("scripts/06_extract_features.py", "--qualities", "90,70,50,30", "--splits", "test,phone,forensics", "--gpus", "auto")
run("scripts/06_extract_features.py", "--qualities", "70,50", "--splits", "train", "--gpus", "auto")'''),
    ("code", r'''# The pilot has SDXL-Turbo fakes only, so its `main` is trained on that generator; the full run uses the default (turbo + sd15).
# Every requested generator must exist in the index (exit 3 otherwise): if you skipped sd15, pass --train-generators sdxl-turbo here.
tags = [("main", ["--train-generators", "sdxl-turbo"])] if PILOT else [("main", []), ("main_noaug", ["--augment-qualities", ""]), ("turbo_only", ["--train-generators", "sdxl-turbo"]),
                                     ("control_mismatch", ["--train-fake-domain", "forensics"]), ("all3", ["--train-generators", "sdxl-turbo,sd15,kandinsky22"])]
for tag, extra in tags:
    try:
        run("scripts/07_train_authenticity.py", "--tag", tag, *extra); run("scripts/08_eval_authenticity.py", "--tag", tag)
    except subprocess.CalledProcessError as e:
        if tag != "main" and e.returncode == 3:      # exit 3 = 'no data for this configuration' (e.g. no Kandinsky attached)
            print(f"[skip] {tag}: no data for this configuration"); continue
        raise                                          # OOM, broken training etc. must stop the notebook
try: run("scripts/14_png_diagnostic.py", "--tag", "main", "--split", "test")
except subprocess.CalledProcessError as e:
    if e.returncode != 3: raise
    print("[skip] png diagnostic: no PNG twins in this run")'''),
    ("md", "### Read the results like an examiner\n* `train_summary.json → recommended_mode`: if fusion does not beat the probe on the calibration half, the demo runs in probe mode.\n* `catalogue/all_seen`: seen generators. Read `tpr_at_threshold` / `fpr_at_threshold` (frozen threshold), not only AUROC.\n* `catalogue/kandinsky22 [unseen_generator_same_domain]` under `main`: the in-domain transfer number; `catalogue/sd15` under `turbo_only` is a second one.\n* `forensics/internal`: unseen generators AND unseen domain, on the same reserved 40% for every tag.\n* `jpeg_curve.png` for `main` vs `main_noaug`: the measured robustness story, whatever it turns out to be.\n* `control_mismatch`: near chance on `catalogue/*` while high on `forensics/internal` means the style confound is real.\n* `png_diagnostic.json`: the storage cost of the frequency cue."),
    ("code", r'''import pandas as pd
from IPython.display import Image as IImage, display
for d in sorted((OUT / "results").glob("*/summary.csv")):
    s = pd.read_csv(d); tag = d.parent.name
    print(f"\n=== {tag} ==="); print(s[s["quality"] == 0].pivot_table(index="test_set", columns="model", values=["auroc", "tpr_at_threshold", "fpr_at_threshold"]).round(3).to_string())
    if (d.parent / "jpeg_curve.png").exists(): display(IImage(str(d.parent / "jpeg_curve.png")))
print(json.load(open(OUT / "checkpoints" / "main" / "train_summary.json"))["recommended_mode"])''')],
   "kaggle_04_features_and_authenticity.ipynb",
   required=["data/features/features_q0.npz", "outputs/checkpoints/main/probe.npz", "outputs/results/main/summary.csv", "data/processed/index.csv"])

# --------------------------------------------------------------------------- 5
nb("05 · Consistency branch (CLIP zero-shot + SmolVLM + calibration)",
   """**Accelerator:** GPU **T4 x2** · **Attach:** `abo-prepared`, `lv-features-checkpoints`, the `abo-fakes-*` datasets, optional `phone-set`, `hand-labels` · **Output dataset name:** `lv-consistency`

For every real listing with a trusted label we ask CLIP and SmolVLM-500M (multiple choice, letter read from
next-token logits) and cache the option probabilities. Then synthetic mismatch pairs train TWO calibrators
(with and without VLM features; the demo picks the matching one), evaluated against raw CLIP-argmax and raw
VLM-argmax, per attribute, plus the phone split and the hand labels when present.

**VLM pilot check first:** look at 20 development images and compare the letter-logit answer with a generated
answer; both should agree most of the time and the distributions must be finite and normalised.""",
   [("code", r'''attach("abo-prepared", "data/processed", DATA / "processed", must_contain="abo_clean.csv")
attach("lv-features-checkpoints", "data/features", DATA / "features", must_contain="features_q0.npz")
attach("lv-features-checkpoints", "outputs/checkpoints", OUT / "checkpoints", must_contain="main")
FAKE_SLUGS = ''' + repr(FAKE_SLUGS) + r'''
''' + MERGE_FAKES + r'''
optional_attach("phone-set", "phone", DATA / "phone", must_contain="labels.csv")
run("scripts/05_build_index.py", "--no-audit")     # rebuild so paths match this session (phone included)'''),
    ("code", r'''# VLM pilot: letter-logit answers vs generated answers on 20 train images
import pandas as pd
sys.path.insert(0, str(REPO))
from listing_verifier.features.vlm import SmolVLM
from listing_verifier.features.image_ops import canonicalise, load_rgb
from listing_verifier.models.consistency import VLM_QUESTIONS, vlm_options
idx = pd.read_csv(DATA / "processed" / "index.csv", low_memory=False)
dev = idx[(idx["split"] == "train") & (idx["label"] == 0) & idx["colour"].notna()].sample(20, random_state=0)
vlm = SmolVLM(device="cuda:0"); agree = 0
for _, r in dev.iterrows():
    img = canonicalise(load_rgb(r["path"])); opts = vlm_options("colour")
    p = vlm.option_probabilities(img, VLM_QUESTIONS["colour"], opts); g, txt = vlm.generate_answer(img, VLM_QUESTIONS["colour"], opts)
    agree += int(g is not None and opts[g] == opts[int(p.argmax())])
    print(f"{r['colour']:8s} logit->{opts[int(p.argmax())]:8s} ({p.max():.2f})  generated->{txt.strip()[:12]!r}")
print(f"logit/generation agreement: {agree}/20 (expect most to agree; if not, inspect the prompt format)")
del vlm'''),
    ("code", 'run("scripts/09_run_consistency.py", "--splits", "train,val,test,phone", "--max-train", 2500, "--gpus", "auto")'),
    ("code", r'''hl = input_file("hand-labels", "hand_labels_filled.csv")
if hl is not None:
    (DATA / "hand_labels").mkdir(exist_ok=True); shutil.copy(hl, DATA / "hand_labels" / "hand_labels_filled.csv")
run("scripts/10_train_eval_consistency.py", "--tag", "main")
run("scripts/11_make_handlabel_sheet.py", "--n", 100)'''),
    ("md", "### Hand labels\nDownload `data/hand_labels/` (the contact sheet embeds thumbnails, so it opens on any laptop). Fill `human_label` (1 = the metadata claim matches the photo, 0 = it does not) and `visible_count` where asked, upload as dataset `hand-labels` (`hand_labels_filled.csv`) and re-run this notebook. These 100 listings are a held-out audit: never use them to tune anything."),
    ("code", r'''from IPython.display import Image as IImage, display
r = json.load(open(OUT / "results" / "consistency" / "results.json"))
for section in ("synthetic_test", "phone_synthetic", "hand_labelled"):
    if section not in r: continue
    print(f"\n== {section} ==")
    res = r[section] if section != "synthetic_test" else r[section]["calibrated"]
    print({a: f"n={v['n']} F1mis={v['f1_mismatch']:.2f} acc={v['accuracy']:.2f} ece={v.get('ece', float('nan')):.3f}" for a, v in res.items() if isinstance(v, dict)})
if "synthetic_test" in r:
    for name, res in r["synthetic_test"].items(): print(f"{name:22s}", {a: round(v["f1_mismatch"], 2) for a, v in res.items()})
display(IImage(str(OUT / "results" / "consistency" / "f1_by_attribute.png")))''')],
   "kaggle_05_consistency.ipynb",
   required=["outputs/checkpoints/main/calibrator.npz", "outputs/checkpoints/main/calibrator_clip_only.npz", "outputs/results/consistency/results.json"])

# --------------------------------------------------------------------------- 6
nb("06 · Demo bundle (scored, hashed, ready for the laptop)",
   """**Accelerator:** none (CPU scoring mirrors the laptop) · **Runtime:** ~15 min · **Attach:** `abo-prepared`, `lv-features-checkpoints`, `lv-consistency`, the `abo-fakes-*` datasets, optional `phone-set` · **Output:** `demo_bundle.zip`

Copies the heads, thresholds and BOTH calibrators, picks a structured example set (reals with claims, one t2i and
one i2i fake per generator, a count>1 item, phone items), scores every example at JPEG none/70/50/30 in the
recommended mode with and without the VLM, and records expected outputs, latency and SHA-256 hashes.
Then fill `visible_ok` in `examples.csv` by eye. On the laptop: `docs/DEMO.md` (one online prefetch of CLIP and
SmolVLM, then three offline rehearsals from a fresh process).""",
   [("code", r'''attach("abo-prepared", "data/processed", DATA / "processed", must_contain="abo_clean.csv")
attach("lv-features-checkpoints", "outputs/checkpoints", OUT / "checkpoints", must_contain="main")
cal = input_file("lv-consistency", "outputs/checkpoints/main")
if cal is not None:   # ONLY the consistency-stage artifacts; never let an older copy of the heads overwrite the authenticity models
    for f in cal.glob("calibrator*"):
        shutil.copy(f, OUT / "checkpoints" / "main" / f.name); print("calibrator:", f.name)
FAKE_SLUGS = ''' + repr(FAKE_SLUGS) + r'''
''' + MERGE_FAKES + r'''
optional_attach("phone-set", "phone", DATA / "phone", must_contain="labels.csv")
run("scripts/05_build_index.py", "--no-audit")
run("scripts/12_build_demo_bundle.py", "--tag", "main", "--n-examples", 12)
print(json.dumps(json.load(open(OUT / "demo_bundle" / "manifest.json")), indent=1))
shutil.make_archive(str(WORK / "demo_bundle"), "zip", OUT / "demo_bundle"); print("zip ready: /kaggle/working/demo_bundle.zip")''')],
   "kaggle_06_demo_bundle.ipynb", required=["demo_bundle.zip", "outputs/demo_bundle/examples/expected.json"])
