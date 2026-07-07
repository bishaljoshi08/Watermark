import os
import zipfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, sobel, median_filter
from scipy.signal import wiener as wiener_filter

# =========================================================================
# CONFIGURATION
# =========================================================================
DATASET_DIR  = Path("/home/atml_team006/watermark/Dataset")
TEMP_OUT_DIR = Path("submission_temp")
FILE_PATH    = "submission_blend1.zip"

ALPHA = 0.45

MASK_LOW    = 0.4
MASK_HIGH   = 1.6
MASK_WINDOW = 7

KUTTER_METHOD = 'wiener'   # 'wiener' or 'median'
KUTTER_SIZE   = 5


BLEND_W = 0.6


BLEND_W_OVERRIDE = {
    # "WM_4": 0.3,   # example: lean more Kutter for a scheme that looks content-adaptive
}

RUN_LOCAL_LPIPS_CHECK = True
LPIPS_SAMPLE_SIZE     = 20

CATEGORIES = [
    ("WM_1", 1,   25),
    ("WM_2", 26,  50),
    ("WM_3", 51,  75),
    ("WM_4", 76,  100),
    ("WM_5", 101, 125),
    ("WM_6", 126, 150),
    ("WM_7", 151, 175),
    ("WM_8", 176, 200),
]

TARGET_SIZE = (256, 256)

# =========================================================================
# SETUP
# =========================================================================
if TEMP_OUT_DIR.exists():
    shutil.rmtree(TEMP_OUT_DIR)
TEMP_OUT_DIR.mkdir(exist_ok=True)

base_path = DATASET_DIR
if (base_path / "Dataset").exists():
    base_path = base_path / "Dataset"

source_base = base_path / "watermarked_sources"
target_dir  = base_path / "clean_targets"

print(f"Resolved base path : {base_path.resolve()}")
print(f"Source base        : {source_base.resolve()}")
print(f"Target dir         : {target_dir.resolve()}")

if not source_base.exists():
    raise FileNotFoundError(f"watermarked_sources not found at {source_base}")
if not target_dir.exists():
    raise FileNotFoundError(f"clean_targets not found at {target_dir}")

# =========================================================================
# HELPERS — loading
# =========================================================================
def load_images_float(folder: Path) -> np.ndarray:
    paths = sorted(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {folder}")
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
        imgs.append(np.array(img, dtype=np.float32) / 255.0)
    return np.stack(imgs, axis=0)


def load_single_image_float(path: Path):
    pil = Image.open(path).convert("RGB")
    orig_size = pil.size
    resized = pil.resize(TARGET_SIZE, Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32) / 255.0
    return arr, orig_size


# =========================================================================
# HELPERS — perceptual masking
# =========================================================================
def compute_texture_mask(img_arr, window=MASK_WINDOW, low=MASK_LOW, high=MASK_HIGH):
    gray = img_arr.mean(axis=2)
    gx = sobel(gray, axis=0)
    gy = sobel(gray, axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    activity = uniform_filter(grad_mag, size=window)
    denom = (activity.max() - activity.min() + 1e-8)
    activity = (activity - activity.min()) / denom
    mask = low + activity * (high - low)
    mask = mask / (mask.mean() + 1e-8)  # mean-normalized
    return mask[:, :, None]


# =========================================================================
# HELPERS — Kutter et al. local-filter estimation
# =========================================================================
def local_predict(img_arr, method=KUTTER_METHOD, size=KUTTER_SIZE):
    pred = np.zeros_like(img_arr)
    for c in range(3):
        channel = img_arr[:, :, c]
        if method == 'wiener':
            filtered = wiener_filter(channel, mysize=size)
            filtered = np.nan_to_num(filtered, nan=channel.mean())
            pred[:, :, c] = filtered
        elif method == 'median':
            pred[:, :, c] = median_filter(channel, size=size)
        else:
            raise ValueError(f"Unknown method: {method}")
    return pred


def kutter_estimate_single(img_arr, method=KUTTER_METHOD, size=KUTTER_SIZE):
    pred = local_predict(img_arr, method=method, size=size)
    return img_arr - pred


def kutter_scheme_estimate(wm_imgs, method=KUTTER_METHOD, size=KUTTER_SIZE):
    residuals = np.stack(
        [kutter_estimate_single(img, method=method, size=size) for img in wm_imgs],
        axis=0
    )
    return residuals.mean(axis=0)


def cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8))


def forge_image(target_arr, blended_estimate, alpha):
    mask = compute_texture_mask(target_arr)
    forged_arr = target_arr + alpha * mask * blended_estimate
    forged_arr = np.clip(forged_arr, 0.0, 1.0)
    return forged_arr


# =========================================================================
# STEP 1: Compute clean mean once
# =========================================================================
print("\nComputing clean target mean...")
clean_imgs = load_images_float(target_dir)
clean_mean = clean_imgs.mean(axis=0)
print(f"  Loaded {len(clean_imgs)} clean images")
print(f"  clean_mean  mean={clean_mean.mean():.4f}  std={clean_mean.std():.4f}")

# =========================================================================
# STEP 2: Per WM set — estimate (avg + kutter, blended), inject (masked)
# =========================================================================
print("\nStarting forgery attack (avg + kutter blend, masked injection)...")
total_processed = 0
lpips_check_pairs = []

for wm_name, target_start, target_stop in CATEGORIES:
    print(f"\n{'='*55}")
    print(f"  {wm_name}  →  targets {target_start}–{target_stop}")
    print(f"{'='*55}")

    wm_dir = source_base / wm_name
    if not wm_dir.exists():
        print(f"  [Warning] {wm_dir} not found, skipping.")
        continue

    wm_imgs = load_images_float(wm_dir)  # (25, H, W, 3)

    # ── Estimate 1: Yang et al. averaging ──────────────────────────────
    wm_mean = wm_imgs.mean(axis=0)
    avg_estimate = wm_mean - clean_mean

    # ── Estimate 2: Kutter et al. per-image local filter ───────────────
    kutter_estimate = kutter_scheme_estimate(wm_imgs)

    # ── Diagnostics ──────────────────────────────────────────────────
    sim = cosine_similarity(avg_estimate, kutter_estimate)
    w = BLEND_W_OVERRIDE.get(wm_name, BLEND_W)
    blended_estimate = w * avg_estimate + (1 - w) * kutter_estimate

    print(f"  avg_estimate     std={avg_estimate.std():.5f}")
    print(f"  kutter_estimate  std={kutter_estimate.std():.5f}")
    print(f"  cosine similarity (avg vs kutter): {sim:.4f}")
    print(f"  blend weight used (avg): {w:.2f}")
    print(f"  blended_estimate std={blended_estimate.std():.5f}")

    # ── Injection ────────────────────────────────────────────────────
    count = 0
    for number in range(target_start, target_stop + 1):
        target_path = target_dir / f"{number}.png"
        if not target_path.exists():
            print(f"  [Warning] {target_path} not found, skipping.")
            continue

        target_arr, orig_size = load_single_image_float(target_path)
        forged_arr = forge_image(target_arr, blended_estimate, ALPHA)

        if RUN_LOCAL_LPIPS_CHECK and len(lpips_check_pairs) < LPIPS_SAMPLE_SIZE:
            avg_only_arr = np.clip(
                target_arr + ALPHA * compute_texture_mask(target_arr) * avg_estimate,
                0.0, 1.0
            )
            lpips_check_pairs.append((target_arr.copy(), avg_only_arr.copy(), forged_arr.copy()))

        forged_pil = Image.fromarray((forged_arr * 255).astype(np.uint8))
        if forged_pil.size != orig_size:
            forged_pil = forged_pil.resize(orig_size, Image.LANCZOS)

        out_path = TEMP_OUT_DIR / f"{number}.png"
        forged_pil.save(out_path, format="PNG")
        count += 1

    print(f"  Forged {count} images.")
    total_processed += count

print(f"\nTotal forged: {total_processed} images.")

# =========================================================================
# STEP 3: Local LPIPS sanity check — avg-only (masked) vs blend (masked)
# =========================================================================
if RUN_LOCAL_LPIPS_CHECK:
    try:
        import torch
        import lpips

        print("\nRunning local LPIPS comparison (avg-only vs avg+kutter blend)...")
        loss_fn = lpips.LPIPS(net='alex')

        def to_lpips_tensor(arr):
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
            return t * 2 - 1

        avg_scores = []
        blend_scores = []
        for orig_arr, avg_arr, blend_arr in lpips_check_pairs:
            orig_t  = to_lpips_tensor(orig_arr)
            avg_t   = to_lpips_tensor(avg_arr)
            blend_t = to_lpips_tensor(blend_arr)

            with torch.no_grad():
                avg_scores.append(loss_fn(orig_t, avg_t).item())
                blend_scores.append(loss_fn(orig_t, blend_t).item())

        avg_scores = np.array(avg_scores)
        blend_scores = np.array(blend_scores)

        print(f"  Avg-only LPIPS : mean={avg_scores.mean():.5f}  std={avg_scores.std():.5f}")
        print(f"  Blend LPIPS    : mean={blend_scores.mean():.5f}  std={blend_scores.std():.5f}")

        s_qlt_avg   = np.exp(-8 * avg_scores).mean()
        s_qlt_blend = np.exp(-8 * blend_scores).mean()
        print(f"  Approx S_qlt avg-only : {s_qlt_avg:.4f}")
        print(f"  Approx S_qlt blend    : {s_qlt_blend:.4f}")

        if s_qlt_blend > s_qlt_avg:
            print("  → Blend improved estimated visual quality over avg-only.")
        else:
            print("  → Blend did not help on this sample; check BLEND_W / BLEND_W_OVERRIDE.")

    except ImportError:
        print("\n[Skipping LPIPS check] Install with: pip install lpips torch")

# =========================================================================
# STEP 4: Package into submission zip
# =========================================================================
print(f"\nPackaging into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in sorted(TEMP_OUT_DIR.glob("*.png")):
        zipf.write(img_path, arcname=img_path.name)

print(f"Done. Submit: {FILE_PATH}")