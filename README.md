## 1. Environment setup

```bash
# Core deps (all scripts)
pip install torch torchvision numpy pillow scipy

# WMForger only (preference-model extractor)
pip install omegaconf timm

# evaluate.py only (local Sqlt sanity check)
pip install lpips

# WMcopier only (diffusion fine-tuning, optional — not the winning method)
pip install diffusers

```

## 2. Path configuration

Every script hardcodes its paths in a `CONFIG` block near the top (no CLI
args / env vars). Before running a script, open it and edit:

- `DATASET_DIR` — point at your `Dataset/` folder.
- `TEMP_OUT_DIR` — scratch folder for the 200 forged PNGs.
- `FILE_PATH` — where the final submission `.zip` is written.
- (`WMForger` only) `WMFORGER_DIR` / `CKPT_PATH` — see setup below.

---

## Running WMForger (the winning method)

### One-time setup

```bash
git clone https://github.com/facebookresearch/videoseal.git
cd videoseal/wmforger
wget https://dl.fbaipublicfiles.com/wmforger/convnext_pref_model.pth
pip install timm omegaconf torchvision --user
```

The scripts import the official `wmforger` package and
`configs/extractor.yaml` from that clone, so either run the script from
inside `videoseal/wmforger/`, or edit `WMFORGER_DIR` at the top of the
script to point at it. Also make sure `CKPT_PATH` resolves to the
downloaded `convnext_pref_model.pth`.

### Which version to run?

`WMforger.py` is the original draft; `v3`-`v11` are a tuning sweep of the
same method (config-only diffs — see table below). **`v8` or `v9` were
the best-performing configs. I am not sure please verify by running both**;

```bash
# Edit WMFORGER_DIR / CKPT_PATH / DATASET_DIR / TEMP_OUT_DIR / FILE_PATH
# at the top of the file first.
python WMforger_v9.py       # or WMforger_v8.py
```

This will, per category:
1. Extract a watermark estimate from each of the 25 sources (gradient-ascend
   a preference score for `NUM_STEPS` steps, `watermark = source - cleaned`).
2. Combine the 25 per-source estimates into one pattern via
   `EXTRACTION_MODE` (`"median"` in v8-v11 — confirmed more robust than a
   fixed single source, see table below).
3. Forge onto every target in that category: `forged = clean + ALPHA * pattern`
   (with a per-category `ALPHA_PER_CATEGORY` override and, in v8-v11,
   optional local texture masking for categories in `MASK_CATEGORIES`).
4. Zip the 200 forged PNGs into `FILE_PATH`.

### Config sweep reference (`WMforger.py` → `v11`)

| Version | NUM_STEPS | ALPHA (default) | Per-category override | Extraction mode | Masking |
|---|---|---|---|---|---|
| base | 50 | 1.5 | — | mean | off |
| v3 | 100 | 1.5 | — | mean | off |
| v4 | 100 | 1.5 | — | median | off |
| v5 | 200 | 1.0 | — | median | off |
| v6 | 150 | 1.3 | — | median | off |
| v7 | 200 | 1.0 | — | **per_source** (bad) | off |
| **v8** | 150 | **1.7** | WM_6: 1.3 | median | WM_6 only |
| **v9** | 200 | **2.0** | WM_6: 1.3 | median | WM_6 only |
| v10 | 200 | 3.0 | WM_6: 1.5 | median | WM_6 only |
| v11 | 200 | 3.0 | WM_6: 1.5 | median | all 8 categories |

Key finding along the way: `v7`'s `EXTRACTION_MODE = "per_source"` (forge
every target with one fixed source's raw extraction, no averaging) was
**catastrophic** (overall Sqlt collapsed to ~0.45) — it proved the
per-source extraction is heavily content-entangled, so combining across the
25 sources (mean/median) isn't just noise reduction, it's required to strip
out source-specific content. `v8` onward locks in `"median"`.

---

## Local visual-quality check (`evaluate.py`)

`evaluate.py` computes the same `Sqlt = exp(-8 * LPIPS)` formula the course
uses, per category and overall, given a folder of forged PNGs — this lets
you sweep `ALPHA` / `NUM_STEPS` and check quality locally before spending a
submission attempt. It does **not** compute Sdet (that requires the actual
black-box detector).

```bash
# Edit CLEAN_DIR / FORGED_DIR at the top to point at Dataset/clean_targets
# and whichever TEMP_OUT_DIR you just forged into.
python evaluate.py
```

