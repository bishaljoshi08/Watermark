"""Visualize the averaging attack internals:
   - blur_A : mean of the 25 watermarked sources, one per WM group (8 panels)
   - blur_B : mean of the clean targets (1 panel, shared across groups)
   - delta  : blur_A - blur_B, amplified so the watermark structure is visible
Run on the real Dataset/ to inspect which methods are content-agnostic.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")  # cluster-safe
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_DIR = Path("Dataset")
OUT = "blur_visualization.png"
CATEGORIES = [("WM_1",1,25),("WM_2",26,50),("WM_3",51,75),("WM_4",76,100),
              ("WM_5",101,125),("WM_6",126,150),("WM_7",151,175),("WM_8",176,200)]

def load_rgb(p, size=None):
    im = Image.open(p).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.BICUBIC)
    return np.asarray(im, np.float32)

def mean_image(paths, size):
    acc = None
    for p in paths:
        a = load_rgb(p, size)
        acc = a if acc is None else acc + a
    return acc / len(paths)

def amp(delta):
    """Normalize a signed pattern to [0,1] with symmetric percentile scaling."""
    s = np.percentile(np.abs(delta), 99) + 1e-8
    return np.clip(delta / (2 * s) + 0.5, 0, 1)

tdir = DATASET_DIR / "clean_targets"
ref = Image.open(tdir / "1.png").convert("RGB").size
blur_B = mean_image([tdir / f"{i}.png" for i in range(1, 201)], ref)

blur_A, delta = {}, {}
for wm, s, e in CATEGORIES:
    src = sorted((DATASET_DIR / "watermarked_sources" / wm).glob("*.png"))
    blur_A[wm] = mean_image(src, ref)
    delta[wm] = blur_A[wm] - blur_B

fig, axes = plt.subplots(3, 8, figsize=(20, 8))
for j, (wm, s, e) in enumerate(CATEGORIES):
    axes[0, j].imshow(blur_A[wm] / 255.0); axes[0, j].set_title(f"blur_A {wm}", fontsize=10)
    axes[2, j].imshow(amp(delta[wm]));     axes[2, j].set_title(f"delta {wm} (amp)", fontsize=10)
    axes[0, j].axis("off"); axes[1, j].axis("off"); axes[2, j].axis("off")
# blur_B once, centered in the middle row
axes[1, 3].imshow(blur_B / 255.0); axes[1, 3].set_title("blur_B (clean mean)", fontsize=10)
axes[1, 3].axis("on"); axes[1, 3].set_xticks([]); axes[1, 3].set_yticks([])
fig.suptitle("Row 1: blur_A per group   |   Row 2: blur_B (shared)   |   "
             "Row 3: delta = blur_A - blur_B (amplified)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"saved {OUT}")