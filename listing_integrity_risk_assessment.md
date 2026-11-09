# Listing Integrity: honest risk assessment

Idea 1 from `hybrid_project_ideas.html`: *is this product photo real, and does the description match what the photo shows?*
Written 6 Oct 2026 for the DLVA proposal decision.

---

## 0. Lineage

- **Primary parent:** shortlist pick S1 (generalising fake-image detector), anchored on UnivFD (Ojha et al., CVPR 2023), AIDE (Yan et al., ICLR 2025) and Community Forensics (Park and Owens, CVPR 2025).
- **Second parent:** Stanford report R1 (product descriptions on ABO, CS231N 2024), which supplies the ABO catalogue, metadata-as-ground-truth and the small-VLM pattern.
- **Minor parent:** Stanford report R5 (clothing retrieval, CS231N 2025), for the image-text embedding comparison.
- **Reusable weights come from the published papers, not from any Stanford report.** No Stanford team released code or weights. What we reuse with certainty is the frozen CLIP backbone; the UnivFD and AIDE detector checkpoints are a generic-domain safety net and baseline.

## 1. Bottom line

| Question | Answer |
|---|---|
| Will the demo work on the day? | **High confidence (roughly 85–90%)**, provided the demo runs on pre-tested inputs and does not generate fakes live or fetch random listings from the internet. |
| Will the authenticity detector work at all? | **Very high.** The part we always reuse is the frozen OpenAI CLIP backbone; our own training is a linear probe on it, done in minutes. As a safety net, the authors of two anchor papers (UnivFD, AIDE) publish trained detector checkpoints. Those were trained on generic scene images, not catalogue photos, so they are likely but not guaranteed to transfer to product shots; they give a working panel and a published baseline from week 1, not a replacement for our probe. |
| Will the consistency check (text vs image) work? | **High for colour and count, medium for material.** Materials are visually ambiguous and will be reported as the honest weak spot. |
| Will it generalise to brand-new 2026 generators? | **Medium, and that is fine.** This is the research question, not a requirement. A partial failure here is a finding to report, not a demo failure. |
| Can we obtain fakes? | **Yes, two independent ways** (generate our own, or stream existing datasets), so neither can block the project. See section 3. |
| Any legal or licence blocker? | **No.** Every component is free for academic use. The flagged line is about download size and giving credit, not about permission. See section 4. |

---

## 2. The real risks, ranked

### Risk 1 (scientific, the one that matters): the detector learns the wrong thing
**What can go wrong.** ABO real photos are clean catalogue shots on white backgrounds. If our fakes look different in *style* (cluttered scenes, different framing, different resolution), the probe learns "catalogue photo vs not catalogue photo" instead of "real vs generated". Accuracy on our own test set would look excellent and mean nothing.
**How we know it is a risk.** Report R1 in the Stanford review had the same pattern: a CIDEr of 473 caused by near-duplicate listings leaking between train and test.
**Mitigation.**
- Generate fakes *from ABO itself*: image-to-image from real ABO photos at several strengths, and text-to-image from ABO item names. Fakes then share framing, background and category with the reals.
- Add real photos that are *not* catalogue shots (our own phone photos of objects) so "real" is not synonymous with "white background".
- Split train/test by product family and brand, not by listing, so templates cannot leak.
- Report a control experiment: a detector trained on *style-mismatched* fakes, to show we checked.

### Risk 2 (demo): a live input behaves unexpectedly
**What can go wrong.** An image pasted live from a current generator is classified wrongly, or the JPEG slider makes the verdict flip in an embarrassing direction.
**Mitigation.** Every demo input is chosen and tested the day before. The JPEG slider is a *feature*: the script says in advance "watch the frequency branch die below quality 70 while the semantic branch holds". A recorded backup video exists. Nothing in the demo depends on internet access or on a GPU.

### Risk 3 (domain): "marketplace" promise vs "Amazon catalogue" data
**What can go wrong.** Real second-hand listings are phone photos in living rooms. Our training data is studio catalogue photography. An examiner may ask whether the tool would work on OLX-style photos.
**Mitigation.** Build a small set (100–200 images) of our own phone photos of household items as additional reals, and generate matching fakes from them. Report results on both "catalogue" and "phone" splits. Say clearly in the proposal that the catalogue is the training ground and the phone set is the realism check.

### Risk 4 (data quality): ABO metadata is messy
**What can go wrong.** The colour field contains strings like "Multicolor Yellow Accent"; descriptions contain placeholders ("Content coming soon"); some fields are missing.
**Mitigation.** Map colours to a fixed vocabulary of ~12 colours, drop listings with placeholder descriptions, and only verify attributes that exist in the structured metadata (colour, material, item count). This is a one-day cleaning job, done in week 1.

### Risk 5 (consistency branch): the small VLM hallucinates or is vague
**What can go wrong.** Asked "what material is this?", a 500 M-parameter VLM may answer confidently and wrongly.
**Mitigation.** Ask multiple-choice questions ("Is this item red, blue, black, white, or other?") instead of open questions; also use CLIP zero-shot contrast as a second opinion; learn a tiny calibration layer on the synthetic mismatch pairs. Report materials as weak if they are weak.

### Risk 6 (compute/time): Colab GPU availability during fake generation
**What can go wrong.** Generating 10k images needs 1–2 GPU-hours; Colab may deny a GPU at peak times or disconnect mid-run.
**Mitigation.** Generate in chunks of 500, save to Google Drive after each chunk, or do it on Kaggle where output persists. If generation stalls entirely, use streamed existing fakes (section 3) and the project still proceeds.

### Risk 7 (engineering): library versions
**What can go wrong.** `open_clip`, `transformers`, `diffusers` change APIs. Two Stanford teams lost their main experiment this way.
**Mitigation.** Pin versions in a `requirements.txt` on day one. Risk is low here because every component is mainstream and actively maintained.

### Risk 8 (evaluation): the Chameleon benchmark is request-only
**What can go wrong.** The hardest public benchmark for fake detection is sent by e-mail on request and may not arrive.
**Mitigation.** Request it in week 1; the project does not depend on it. Our own fresh 2026 mini test set plays the same role.

### What is *not* a risk
- Legal or ethical exposure: no personal data, no copyrighted media, no scraping. Scam-listing screenshots used in the intro slide are anonymised.
- Model size: everything fits under 6 GB of GPU memory and runs on a laptop CPU at demo time.
- Team size: the two branches split cleanly between two people.

---

## 3. Fakes: generate our own, or use an existing dataset?

**Answer: both, and the project is designed so that either alone is enough.**

### Option A: generate our own (preferred for the training set)
- **Tools:** SDXL-Turbo (one to four denoising steps, about half a second per 512 px image on a T4, ~6 GB VRAM) and Stable Diffusion 1.5 for image-to-image.
- **Why preferred:** fakes come from the *same domain* as the reals (ABO item names and ABO photos as the source), which removes Risk 1. We also control the count and the metadata, so each fake has a known colour and material for the consistency branch.
- **Cost:** 10k images in roughly 1–2 GPU-hours plus ~2 GB of disk.
- **Confidence we can do it:** very high. This is standard `diffusers` usage with no fine-tuning.

### Option B: existing datasets (for the *cross-generator* test, and as the fallback)
| Dataset | What it is | Access | Size | Role |
|---|---|---|---|---|
| CommunityForensics-Small (CVPR 2025) | 278k generated + 278k real images from thousands of generators | Hugging Face, no form, CC BY-NC-SA 4.0 | ~260 GB on disk; stream a few thousand | Unseen-generator test set |
| UnivFD test set (CVPR 2023) | Fakes from 11 GAN and diffusion generators | Google Drive links in the official repo | ~19 GB | Standard comparison with published numbers |
| AIDE released weights and test lists (ICLR 2025) | Trained detector plus evaluation protocol | GitHub | Weights only, small | Fallback detector and published baseline (generic-domain training; transfer to catalogue photos must be measured) |
| DiffusionDB | Millions of Stable Diffusion images with prompts | Hugging Face, streamable, CC0 | Huge; stream a subset | Extra diffusion fakes if generation stalls |
| Chameleon (ICLR 2025) | Fakes that fool humans | E-mail request, academic use | Small | Nice-to-have hard benchmark |

**Important caveat for Option B:** none of these are product photos. Using them as the *training* set would re-introduce Risk 1 (style confound). They are right for testing generalisation, and acceptable as training data only as a fallback, with the confound acknowledged in the report.

### Why having both matters for a "sure shot"
If generation fails, we train on streamed fakes and the demo still runs. If streaming fails, we train on our own fakes and test on the UnivFD download. If our own training fails, we load AIDE's or UnivFD's published checkpoints; they were trained on generic images, so their accuracy on product photos must be measured, but they keep the authenticity panel working. Three independent paths to a working detector.

---

## 4. The licence and download line, in plain words

The line in the resource table was:

> "SDXL-Turbo is ungated under the Stability AI Community License (free for research); SD 1.5 is under CreativeML OpenRAIL-M via a maintained mirror (click-through). CommunityForensics-Small is CC BY-NC-SA 4.0, ungated, and about 260 GB, so stream it from Hugging Face rather than download."

It is marked with a caution sign (!) not because anything is forbidden, but because each item has one practical detail to know. Item by item:

**SDXL-Turbo, "ungated under the Stability AI Community License (free for research)."**
- *Ungated* means you can download the weights immediately; there is no form to fill and nothing to wait for.
- The *Community License* means Stability AI lets anyone use the model for research, personal and non-commercial work for free (and for commercial work below a revenue threshold). A university project is squarely inside what is allowed.
- What to do: nothing, except cite the model and name the licence in the report.

**Stable Diffusion 1.5, "under CreativeML OpenRAIL-M via a maintained mirror (click-through)."**
- The company that originally hosted SD 1.5 (RunwayML) removed its copy in 2024. The weights are still freely available from other maintained copies ("mirrors") on Hugging Face.
- *OpenRAIL-M* is an open licence that permits free use but asks you not to use the model for a list of harmful purposes (illegal content, harassment, and so on). Our use is fine.
- *Click-through* means some mirrors ask you to log in with a free Hugging Face account and press "agree" once before the download starts.
- What to do: create a free Hugging Face account, click agree, cite the licence. Five minutes.

**CommunityForensics-Small, "CC BY-NC-SA 4.0, ungated, and about 260 GB, so stream it rather than download."**
- *CC BY-NC-SA 4.0* is a Creative Commons licence: free to use, you must credit the authors (BY), you may not use it commercially (NC), and anything you build and release from it must carry the same licence (SA). Coursework is non-commercial, so this is fine; we simply cite it.
- *Ungated*: no form, no approval.
- *260 GB*: despite the word "Small" in its name, the dataset holds about 556,000 images. Colab's scratch disk is about 78 GB and Kaggle's persistent output is 20 GB, so it cannot be downloaded in full on free tiers.
- *Stream it*: the Hugging Face `datasets` library can read a dataset piece by piece over the network without saving all of it. We pull a few thousand images, save that subset (2–3 GB) once to Google Drive or as a Kaggle dataset, and never touch the rest.
- Residual risk: streaming depends on network speed and on the dataset's file layout allowing us to pick particular generator families. Mitigation: do the streaming once, early, on Kaggle, and keep the saved subset. If the layout is awkward, the UnivFD 19 GB download covers the same purpose.

**In one sentence:** all three are legal and free for this project; the only actions are a free account click for SD 1.5, citing three licences in the report, and remembering that the "small" dataset is not small.

---

## 5. What "sure shot" requires from us

1. **Week 1 deliverable is a working baseline**, not a plan: ABO downloaded, 500 fakes generated, UnivFD weights loaded, first AUROC printed. If week 1 fails, the fallback paths above are triggered immediately.
2. **Evaluation code is written before the models** (the main lesson from the nine Stanford reports).
3. **Demo inputs are frozen one week before the presentation** and rehearsed twice; a recorded run exists.
4. **Claims are scoped honestly**: "colour and count checks work; material is hard; generalisation to generator X dropped by Y points under compression". An examiner rewards that more than an inflated number.

## 6. Residual confidence by component

| Component | Confidence it works as demoed | Why |
|---|---|---|
| Authenticity on held-out fakes from the same generator family | Very high | Frozen CLIP backbone plus a probe trained in minutes reproduces the CVPR 2023 recipe; published generic-domain checkpoints as a safety net |
| Authenticity on unseen generator families | Medium to high | Published detectors reach roughly 0.85–0.95 AUROC on diffusion families; we report whatever we get |
| Robustness curve under JPEG/resize | High that the curve is informative; medium that fusion beats the probe alone | The frequency branch collapsing is itself the expected, teachable result |
| Colour and count consistency | High | Multiple-choice VLM plus CLIP contrast; clean metadata labels |
| Material consistency | Medium | Visually ambiguous even for humans |
| Live demo on pre-tested inputs, CPU only | High (85–90%) | No live generation, no internet dependency, backup video |
| Research contribution accepted by an examiner | High | Two branches on one backbone, honest robustness study, fresh 2026 test set |
