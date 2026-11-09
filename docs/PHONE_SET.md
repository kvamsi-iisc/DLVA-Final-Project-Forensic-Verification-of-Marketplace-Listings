# Phone-photo split (the realism check)

Why: ABO reals are studio catalogue photos. A buyer's listing is a phone photo in a living room. The
phone split answers "does any of this survive outside the catalogue?" and is reported separately,
never merged into the catalogue numbers.

## Protocol (1–2 hours)

1. Photograph 100–200 household items with your phone: mugs, chairs, shoes, books, bottles, cushions.
   Vary background and lighting. One main object per photo, roughly centred. Do not include people or
   documents.
2. Put the files in `data/phone/real/`.
3. `python -c "from listing_verifier.data.phone import write_labels_template as w; print(w())"` writes
   `data/phone/labels_template.csv`. Fill it in and save it as `data/phone/labels.csv`:

   | column | value |
   |---|---|
   | filename | as in the folder |
   | item_name | short title, e.g. "Set of 2 ceramic mugs" |
   | product_noun | one noun used in prompts, e.g. "mug" |
   | colour | one of the 12 vocab colours, or blank |
   | material | one of the 8 vocab materials, or blank |
   | count | 1–6, or blank |

4. Generate matching fakes (GPU): `python scripts/03_generate_fakes.py --generator sdxl-turbo --mode i2i --strength 0.5 --source phone`
   (and optionally `--mode t2i`). They land in `data/phone/fake/`.
5. Upload `data/phone/` as a Kaggle dataset `phone-set`; notebooks 04 and 05 pick it up automatically.
   Results appear as `phone/internal` rows and a phone JPEG curve.

Phone photos are large (12 MP); `canonicalise()` downsizes them to 256 px like everything else, so the
detector cannot use resolution as a cue.
