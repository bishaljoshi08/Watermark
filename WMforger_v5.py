import os
import sys
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision
from scipy.ndimage import uniform_filter

# ------------------------------- CONFIG -------------------------------
# Path to the cloned videoseal/wmforger directory (so we can import the
# official model code and load configs/extractor.yaml).
WMFORGER_DIR = Path("/scratch/bjoshi/watermark/videoseal/wmforger")
CKPT_PATH    = WMFORGER_DIR / "convnext_pref_model.pth"

DATASET_DIR  = Path("/scratch/bjoshi/watermark/Dataset")
TEMP_OUT_DIR = Path("/scratch/bjoshi/watermark/submission_temp_wmforger_v5")  # where this script will put the forged outputs
FILE_PATH    = "/scratch/bjoshi/watermark/submission_wmforger_v5.zip"

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


WORK_SIZE = 768


NUM_STEPS = 200
LR        = 0.05

ALPHA = 1.0
ALPHA_PER_CATEGORY = {

}

EXTRACTION_MODE = "median"


ENABLE_MASKING = False
MASK_WINDOW_FRAC = 0.03
MASK_REF_PERCENTILE = 90
MASK_FLOOR = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------- MODEL (official wmforger) --------------------
def load_preference_model():
    """Load the pretrained, watermark-agnostic ConvNeXt preference model
    from the official wmforger repo."""
    import omegaconf
    if str(WMFORGER_DIR) not in sys.path:
        sys.path.insert(0, str(WMFORGER_DIR))
    from wmforger.models import build_extractor

    model_type = "convnext_tiny"
    state_dict = torch.load(CKPT_PATH, weights_only=True, map_location="cpu")["model"]
    extractor_params = omegaconf.OmegaConf.load(
        str(WMFORGER_DIR / "configs" / "extractor.yaml")
    )[model_type]

    model = build_extractor(model_type, extractor_params, img_size=256, nbits=0)
    model.load_state_dict(state_dict)
    model = model.eval().to(DEVICE)
    return model


_to_work = torchvision.transforms.Compose([
    lambda x: x.convert("RGB"),
    torchvision.transforms.Resize((WORK_SIZE, WORK_SIZE)),
    torchvision.transforms.ToTensor(),
    lambda x: x.view(1, 3, WORK_SIZE, WORK_SIZE),
])


def extract_watermark_from_source(src_pil, model):
    """Run the preference-model optimization on one watermarked source and
    return the estimated watermark as a float32 (H,W,3) array at the
    source's OWN native size (matching optimize_image.py's get_watermark).
    """
    img = _to_work(src_pil).to(DEVICE)
    param = torch.nn.Parameter(torch.zeros_like(img)).to(DEVICE)
    optim = torch.optim.SGD([param], lr=LR)

    for _ in range(NUM_STEPS):
        optim.zero_grad()
        # Maximize preference score -> minimize negative score.
        loss = -model((img + param).clip(0, 1)).mean()
        loss.backward()
        optim.step()

    cleaned = (img + param).clip(0, 1).detach().cpu()
    cleaned = cleaned.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).squeeze(0).numpy()
    cleaned_pil = Image.fromarray(cleaned).resize(src_pil.size, Image.BILINEAR)

    # watermark = original source - cleaned source (at native size)
    watermark = np.array(src_pil).astype(np.float32) - np.array(cleaned_pil).astype(np.float32)
    return watermark


# ----------------------------- HELPERS --------------------------------
def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def native_size(path):
    return Image.open(path).convert("RGB").size


def resize_delta(delta, out_wh):
    """Resize a SIGNED float pattern to PIL size (W,H), preserving sign."""
    h, w, c = delta.shape
    if (w, h) == out_wh:
        return delta
    chans = [np.asarray(Image.fromarray(delta[:, :, ch], mode="F")
                        .resize(out_wh, Image.BICUBIC), dtype=np.float32)
             for ch in range(c)]
    return np.stack(chans, axis=2)


def local_texture_mask(clean_rgb, window_frac, ref_percentile, floor):
    gray = clean_rgb.mean(axis=2)
    h, w = gray.shape
    win = max(3, int(window_frac * min(h, w)))
    local_mean = uniform_filter(gray, size=win)
    local_sqmean = uniform_filter(gray * gray, size=win)
    local_var = np.clip(local_sqmean - local_mean ** 2, 0, None)
    local_std = np.sqrt(local_var)
    ref = np.percentile(local_std, ref_percentile)
    if ref < 1e-6:
        ref = 1.0
    mask = np.clip(local_std / ref, 0.0, 1.0)
    mask = floor + (1.0 - floor) * mask
    return mask[:, :, None]


# ------------------------------- MAIN ---------------------------------
def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset dir not found: {DATASET_DIR}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Preference model checkpoint not found: {CKPT_PATH}\n"
            f"Download it with:\n"
            f"  wget https://dl.fbaipublicfiles.com/wmforger/convnext_pref_model.pth"
        )

    TEMP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = DATASET_DIR / "clean_targets"

    print(f"Loading preference model on {DEVICE} ...")
    model = load_preference_model()

    total = 0
    for wm, start, stop in CATEGORIES:
        src_dir = DATASET_DIR / "watermarked_sources" / wm
        src_paths = sorted(src_dir.glob("*.png"))
        if not src_paths:
            print(f"  [warn] no sources in {src_dir}, skipping {wm}")
            continue

        cat_size = native_size(src_paths[0])
        alpha = ALPHA_PER_CATEGORY.get(wm, ALPHA)

        print(f"\n{'='*56}")
        print(f"{wm}: native {cat_size}, {len(src_paths)} sources, alpha={alpha}")
        print(f"{'='*56}")

        # --- Extract the watermark from every source (at cat_size) ------
        # Always extract per-source; how we COMBINE them depends on
        # EXTRACTION_MODE. Extracting all of them is the expensive part
        # (NUM_STEPS backprop per source), done once regardless of mode.
        per_source_wms = []   # list of (cat_size H, W, 3) float arrays
        for i, sp in enumerate(src_paths):
            src_pil = Image.open(sp).convert("RGB")
            w_native = extract_watermark_from_source(src_pil, model)
            per_source_wms.append(resize_delta(w_native, cat_size))
            print(f"    extracted {i+1}/{len(src_paths)} "
                  f"(|w| mean={np.abs(w_native).mean():.3f})")

        stacked = np.stack(per_source_wms, axis=0)   # (N, H, W, 3)

        # Build the combined watermark (for mean/median) OR keep per-source.
        if EXTRACTION_MODE == "mean":
            watermark = stacked.mean(axis=0)
            print(f"  [mean] combined |w| mean = {np.abs(watermark).mean():.3f}")
        elif EXTRACTION_MODE == "median":
            watermark = np.median(stacked, axis=0)
            print(f"  [median] combined |w| mean = {np.abs(watermark).mean():.3f}")
        elif EXTRACTION_MODE in ("per_source", "per_source_cycle"):
            watermark = None   # chosen per-target below
            print(f"  [{EXTRACTION_MODE}] per-source |w| means: "
                  f"{[round(float(np.abs(w).mean()), 2) for w in per_source_wms]}")
        else:
            raise ValueError(f"Unknown EXTRACTION_MODE: {EXTRACTION_MODE}")

        # --- Forge onto this category's targets ------------------------
        grp_targets = [target_dir / f"{n}.png" for n in range(start, stop + 1)]
        for j, tp in enumerate(grp_targets):
            clean = load_rgb(tp)                       # native size

            if EXTRACTION_MODE == "per_source":
                # Fixed single source (the paper's true one-shot spirit).
                src_wm = per_source_wms[0]
            elif EXTRACTION_MODE == "per_source_cycle":
                # Different target -> different single source, round-robin.
                src_wm = per_source_wms[j % len(per_source_wms)]
            else:
                src_wm = watermark

            d = resize_delta(src_wm, (clean.shape[1], clean.shape[0]))
            if ENABLE_MASKING:
                d = d * local_texture_mask(clean, MASK_WINDOW_FRAC,
                                           MASK_REF_PERCENTILE, MASK_FLOOR)
            forged = np.clip(clean + alpha * d, 0, 255).round().astype(np.uint8)
            Image.fromarray(forged).save(TEMP_OUT_DIR / tp.name)
            total += 1
        print(f"  forged {len(grp_targets)} targets")

    print(f"\nForged {total} images total.")
    if total != 200:
        print(f"[WARNING] expected 200, got {total} -- submission may be rejected!")

    print(f"Packaging into {FILE_PATH} ...")
    with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in sorted(TEMP_OUT_DIR.glob("*.png")):
            zf.write(img, arcname=img.name)
    print("Done.")


if __name__ == "__main__":
    main()