# Report outline (what each results file feeds)

1. **Problem and claim.** One frozen CLIP ViT-L/14, two small heads, two judgements. Cite UnivFD, AIDE,
   Community Forensics, ABO.
2. **Data.** ABO cleaning (`cleaning_report.json`: counts, drop reasons, colour/material distributions).
   Family-based splits (`split_report.json`). Fakes from ABO (counts per generator/mode from
   `index.csv`; the real/fake grid from notebook 02). CommunityForensics subset (generator count).
   Phone set. Three licences.
3. **Method.** Canonicalisation (why), probe, spectrum head, fusion; CLIP zero-shot prompts, SmolVLM
   multiple-choice via logits, 7-feature calibrator, synthetic mismatch pairs.
4. **Results: authenticity.** Table from `outputs/results/main/summary.csv` (AUROC, TPR@5%FPR, CI) for
   seen, in-domain unseen (`turbo_only` → sd15), out-of-domain unseen (forensics/internal), phone.
   Figure: `jpeg_curve.png`. Per-mode table (t2i vs i2i strengths). UnivFD baseline row.
5. **Control experiment.** `control_mismatch` results: style-mismatched training fakes → what happens on
   ABO fakes. One paragraph, one table.
6. **Results: consistency.** `outputs/results/consistency/results.json`: per-attribute F1/accuracy/ECE for
   calibrated vs raw CLIP vs raw VLM; reliability diagrams; hand-labelled numbers and the metadata noise rate.
7. **Honest limitations.** Material; cross-generator drop; synthetic mismatches are easier than real fraud;
   count "1" labels are inferred from titles; brand overlap across splits (reported number).
8. **Demo.** Screenshot of the UI; CPU latency; the JPEG-slider story.
9. **Appendix.** `env_report.json` (exact versions, GPU), prompts, colour/material maps.

## Wording to use (and what not to claim)

| Instead of | Write |
|---|---|
| "the same JPEG removes shortcuts" | "controlled preprocessing plus a nuisance-only audit (AUROC x.xx) and a style-mismatch control" |
| "one visual backbone" | "a shared frozen CLIP ViT-L/14 for both heads, with SmolVLM-500M (its own vision encoder) as an auxiliary oracle" |
| "structured count ground truth" | "title-derived weak count labels, audited against visible counts on 100 listings (noise rate y%)" |
| "the frequency head collapses under JPEG" | the measured curve, whichever way it goes, for `main` and `main_noaug` |
| "detects new generators" | the measured unseen-family numbers with CIs and the per-generator table, plus limits |
| "pre-tested offline demo" | only after `expected.json` exists and three offline rehearsals passed |
| "AIDE baseline" | "an AIDE-inspired log-DCT head; the released AIDE model was not run" unless it was |
| "all free, no licence concerns" | the per-artifact licence table (runbook step 3) |
| "85–90% chance the demo works" | nothing: that figure was a planning estimate, not a result |

Also report: achieved (not requested) image counts and independent family counts; validation split into
selection and calibration halves; thresholds frozen before test; `tpr_at_threshold` / `fpr_at_threshold`
with confusion counts per domain and quality; the PNG storage diagnostic; the recommended inference mode and
why; the hand-label noise rate; the phone split as an exploratory domain-shift test with its n.
