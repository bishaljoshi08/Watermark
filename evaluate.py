from pathlib import Path
import numpy as np
from PIL import Image
import torch
import lpips

# ------------------------------- CONFIG -------------------------------
CLEAN_DIR  = Path("/scratch/bjoshi/watermark/Dataset/clean_targets")
FORGED_DIR = Path("/scratch/bjoshi/watermark/submission_temp_wmcopier_pt_lr_new")  # where your forging script put the outputs

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

# 'alex' is the standard choice for LPIPS - fastest, and the default net
# the metric was calibrated/validated against in the original paper.
LPIPS_NET = "alex"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------- HELPERS --------------------------------
def load_lpips_tensor(path):
    """Load an image as a (1,3,H,W) tensor in [-1, 1], LPIPS's expected range."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # [0,255] -> [-1,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


def sqlt_from_lpips(lpips_value):
    return float(np.exp(-8.0 * lpips_value))


# ------------------------------- MAIN ---------------------------------
def main():
    if not CLEAN_DIR.exists():
        raise FileNotFoundError(f"Clean targets dir not found: {CLEAN_DIR}")
    if not FORGED_DIR.exists():
        raise FileNotFoundError(
            f"Forged output dir not found: {FORGED_DIR}. "
            f"Run the forging script first."
        )

    print(f"Loading LPIPS ({LPIPS_NET}) on {DEVICE} ...")
    loss_fn = lpips.LPIPS(net=LPIPS_NET).to(DEVICE)
    loss_fn.eval()

    per_image_lpips = {}   # filename -> lpips value
    missing = []

    with torch.no_grad():
        for wm, start, stop in CATEGORIES:
            for n in range(start, stop + 1):
                fname = f"{n}.png"
                clean_path  = CLEAN_DIR / fname
                forged_path = FORGED_DIR / fname

                if not clean_path.exists() or not forged_path.exists():
                    missing.append(fname)
                    continue

                clean_t  = load_lpips_tensor(clean_path)
                forged_t = load_lpips_tensor(forged_path)

                # Sizes should already match (forging script preserves each
                # target's native resolution), but guard against mismatches
                # so a shape error doesn't kill the whole run.
                if clean_t.shape != forged_t.shape:
                    forged_t = torch.nn.functional.interpolate(
                        forged_t, size=clean_t.shape[2:], mode="bicubic",
                        align_corners=False,
                    )

                d = loss_fn(clean_t, forged_t).item()
                per_image_lpips[fname] = d

    if missing:
        print(f"\n[warn] {len(missing)} image(s) missing from one side, skipped: "
              f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")

    # ------------------------- Aggregate & report -----------------------
    print(f"\n{'Category':<10} {'N':>4} {'Mean LPIPS':>12} {'Mean Sqlt':>12} "
          f"{'Min Sqlt':>10} {'Max Sqlt':>10}")
    print("-" * 62)

    all_lpips = []
    for wm, start, stop in CATEGORIES:
        vals = [per_image_lpips[f"{n}.png"] for n in range(start, stop + 1)
                 if f"{n}.png" in per_image_lpips]
        if not vals:
            print(f"{wm:<10} {'--':>4} {'--':>12} {'--':>12} {'--':>10} {'--':>10}")
            continue

        sqlts = [sqlt_from_lpips(v) for v in vals]
        all_lpips.extend(vals)
        print(f"{wm:<10} {len(vals):>4} {np.mean(vals):>12.4f} "
              f"{np.mean(sqlts):>12.4f} {min(sqlts):>10.4f} {max(sqlts):>10.4f}")

    print("-" * 62)
    if all_lpips:
        overall_sqlts = [sqlt_from_lpips(v) for v in all_lpips]
        print(f"{'OVERALL':<10} {len(all_lpips):>4} {np.mean(all_lpips):>12.4f} "
              f"{np.mean(overall_sqlts):>12.4f} {min(overall_sqlts):>10.4f} "
              f"{max(overall_sqlts):>10.4f}")
    else:
        print("No matched image pairs found - nothing to evaluate.")


if __name__ == "__main__":
    main()