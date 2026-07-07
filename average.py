"""
Watermark forgery via plain pattern averaging (no frequency refinement).

Per watermark group (WM_k -> one block of 25 clean targets):

  1. EXTRACT the watermark pattern by averaging:
         delta = mean(watermarked_sources_k) - mean(clean_targets)
     A fixed additive watermark survives averaging at full strength while
     the varied image content collapses toward a smooth blur. Blackbox
     setting: we lack the sources' clean originals, so the clean-target
     average stands in as the "average natural image".

     What you actually get is:   delta = delta_true + (blur_A - blur_B)
                                         = watermark  +  content residual
     The residual shrinks like 1/sqrt(n); with n=25 it is non-trivial.

  2. FORGE:  forged_i = clip(clean_i + alpha * delta, 0, 255).
     alpha trades Detection Strength (higher) vs Visual Quality (lower);
     the final score S_det * S_qlt peaks at some intermediate alpha.
"""

import os
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image

# ------------------------------- CONFIG -------------------------------
ZIP_FILE     = "/home/atml_team006/watermark/Dataset.zip"
DATASET_DIR  = Path("/home/atml_team006/watermark/Dataset")
TEMP_OUT_DIR = Path("/home/atml_team006/watermark/submission_temp")
FILE_PATH    = "/home/atml_team006/watermark/submission.zip"

ALPHA     = 1.0      # injection strength; sweep ~0.5 .. 3.0
CLEAN_REF = "all"    # "all" 200 targets, or "group" 25, as the clean mean

CATEGORIES = [
    ("WM_1",   1,  25),
    ("WM_2",  26,  50),
    ("WM_3",  51,  75),
    ("WM_4",  76, 100),
    ("WM_5", 101, 125),
    ("WM_6", 126, 150),
    ("WM_7", 151, 175),
    ("WM_8", 176, 200),
]

# ----------------------------- HELPERS --------------------------------
def load_rgb(path, size=None):
    """Load an image as float32 (H,W,3). size is PIL (W,H) or None for native."""
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.float32)


def mean_image(paths, size):
    """Mean of many images resized to a common size. Streamed to save memory."""
    acc = None
    for p in paths:
        arr = load_rgb(p, size)
        acc = arr if acc is None else acc + arr
    return acc / len(paths)


def resize_delta(delta, out_wh):
    """Resize a SIGNED float pattern to PIL size (W,H), preserving sign."""
    h, w, c = delta.shape
    if (w, h) == out_wh:
        return delta
    chans = [np.asarray(Image.fromarray(delta[:, :, ch], mode="F")
                        .resize(out_wh, Image.BICUBIC), dtype=np.float32)
             for ch in range(c)]
    return np.stack(chans, axis=2)


# ------------------------------- MAIN ---------------------------------
def main():
    if not DATASET_DIR.exists():
        if not os.path.exists(ZIP_FILE):
            raise FileNotFoundError(f"Could not find {ZIP_FILE}.")
        print(f"Unzipping {ZIP_FILE} ...")
        with zipfile.ZipFile(ZIP_FILE, "r") as z:
            z.extractall(".")

    TEMP_OUT_DIR.mkdir(exist_ok=True)
    target_dir = DATASET_DIR / "clean_targets"
    ref_size = Image.open(target_dir / "1.png").convert("RGB").size  # (W,H)

    global_clean_mean = None
    if CLEAN_REF == "all":
        all_targets = [target_dir / f"{i}.png" for i in range(1, 201)]
        global_clean_mean = mean_image(all_targets, ref_size)

    total = 0
    for wm, start, stop in CATEGORIES:
        src_dir = DATASET_DIR / "watermarked_sources" / wm
        src_paths = sorted(src_dir.glob("*.png"))
        if not src_paths:
            print(f"  [warn] no sources in {src_dir}")
            continue

        grp_targets = [target_dir / f"{n}.png" for n in range(start, stop + 1)]

        mean_wm = mean_image(src_paths, ref_size)
        mean_clean = (global_clean_mean if CLEAN_REF == "all"
                      else mean_image(grp_targets, ref_size))

        delta = mean_wm - mean_clean      # = watermark + (blur_A - blur_B)
        print(f"{wm}: |delta| mean = {np.abs(delta).mean():.3f}  "
              f"(from {len(src_paths)} sources)")

        for tp in grp_targets:
            clean = load_rgb(tp)          # native size
            d = resize_delta(delta, (clean.shape[1], clean.shape[0]))
            forged = np.clip(clean + ALPHA * d, 0, 255).astype(np.uint8)
            Image.fromarray(forged).save(TEMP_OUT_DIR / tp.name)
            total += 1

    print(f"\nForged {total} images.")
    if total != 200:
        print(f"[WARNING] expected 200, got {total} -- submission may be rejected!")

    print(f"Packaging into {FILE_PATH} ...")
    with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in TEMP_OUT_DIR.glob("*.png"):
            zf.write(img, arcname=img.name)
    print("Done.")


if __name__ == "__main__":
    main()