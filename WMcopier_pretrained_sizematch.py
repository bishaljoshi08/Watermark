import os
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers import (
    UNet2DModel,
    DDPMScheduler,
    DDIMScheduler,
    DDIMInverseScheduler,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

# ------------------------------- CONFIG -------------------------------
ZIP_FILE     = "/scratch/bjoshi/watermark/Dataset.zip"
DATASET_DIR  = Path("/scratch/bjoshi/watermark/Dataset")
TEMP_OUT_DIR = Path("/scratch/bjoshi/watermark/submission_temp_WMcopier_pretrained_sizematch_batchfix")
CKPT_DIR     = Path("/scratch/bjoshi/watermark/checkpoints")
FILE_PATH    = "/scratch/bjoshi/watermark/submission_WMcopier_pretrained_sizematch_batchfix.zip"

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

BASE_MODEL = "google/ddpm-celebahq-256"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    _cc        = torch.cuda.get_device_capability()
    DTYPE      = torch.bfloat16 if _cc[0] >= 8 else torch.float16
    USE_SCALER = (_cc[0] < 8)
else:
    DTYPE      = torch.float32
    USE_SCALER = False

# Fine-tuning
TRAIN_STEPS   = 5000
# Base batch size calibrated for 256x256 on P100 (16 GB).
# get_batch_size() scales this down automatically for larger images.
BATCH_SIZE    = 4
LEARNING_RATE = 5e-5

# Injection
DDIM_INFERENCE_STEPS = 50
SHALLOW_RATIO        = 0.4

# Refinement
REFINE_LAMBDA = 100.0
REFINE_ETA    = 1e-4
REFINE_ITERS  = 100
REFINE_T_L    = 1


# ----------------------------- HELPERS --------------------------------
def get_batch_size(img_size):
    """Scales batch size inversely with image area to keep GPU memory constant.

    Memory used per batch scales as:  batch_size * H * W * channels
    So to keep total memory fixed when H and W double, batch must quarter.

        256x256 (baseline) -> batch = 4
        512x512 (4x area)  -> batch = 1   <-- fixes P100 OOM for WM_7/WM_8
        128x128 (1/4 area) -> batch = 8   (capped; 25 images is tiny anyway)
    """
    ratio = (256 * 256) / (img_size * img_size)
    bs = max(1, int(BATCH_SIZE * ratio))
    print(f"    -> Batch size for {img_size}x{img_size}: {bs}")
    return bs


# ----------------------------- DATASETS -------------------------------
class ImageDataset(Dataset):
    def __init__(self, image_paths, size):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def load_image_tensor(path, size):
    transform = transforms.Compose([
        transforms.Resize(
            (size, size),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0)


def tensor_to_pil(t, original_size):
    t = t.float()   # cast from FP16/BF16 — numpy does not support half types
    t = (t / 2 + 0.5).clamp(0, 1)
    arr = (t.detach().cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)
    return Image.fromarray(arr).resize(original_size, Image.BICUBIC)


def get_category_size(src_paths):
    """Reads native resolution from the first source image.
    Training and injection both run at this size so the watermark is learned
    and injected at the correct pixel scale."""
    w, h = Image.open(src_paths[0]).size
    assert w == h, f"Expected square images, got {w}x{h} in {src_paths[0]}"
    print(f"    -> Native source resolution: {w}x{h}")
    return w


def get_process_size(cat_size):
    return 256 if cat_size == 128 else cat_size


# ----------------------------- STAGE 1 : TRAINING --------------------
def train_wm_model(src_paths, wm_name, img_size):
    ckpt_path = CKPT_DIR / f"{wm_name}_unet_pretrained.pt"

    # Check checkpoint first — skip training entirely if already done
    if ckpt_path.exists():
        print(f"    -> Checkpoint found, skipping training: {ckpt_path}")
        unet = UNet2DModel.from_pretrained(BASE_MODEL).to(DEVICE)
        unet.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        unet.eval()
        return unet

    print(f"    -> Loading pretrained model: {BASE_MODEL}")
    unet = UNet2DModel.from_pretrained(BASE_MODEL).to(DEVICE)

    # Fix 1: gradient checkpointing for 512x512
    # At 256x256 it is not needed and only slows training (~30% overhead)
    if img_size >= 512:
        unet.enable_gradient_checkpointing()
        print("    -> Gradient checkpointing ENABLED (reduces activation memory ~50%)")

    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL)

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS, eta_min=1e-6)

    # Fix 3: GradScaler — only needed for FP16 (P100), not BF16 (H100/A100)
    scaler = torch.cuda.amp.GradScaler() if USE_SCALER else None
    if scaler:
        print(f"    -> GradScaler ENABLED (FP16 mode, dtype={DTYPE})")
    else:
        print(f"    -> GradScaler not needed (dtype={DTYPE})")

    # Fix 2: smaller batch for 512x512
    cat_batch = get_batch_size(img_size)
    dataset   = ImageDataset(src_paths, img_size)
    dataloader = DataLoader(
        dataset, batch_size=cat_batch, shuffle=True, drop_last=False
    )

    print(f"    -> Fine-tuning on {len(src_paths)} sources at {img_size}x{img_size} "
          f"for {TRAIN_STEPS} steps...")

    unet.train()
    step = 0
    while step < TRAIN_STEPS:
        for batch in dataloader:
            if step >= TRAIN_STEPS:
                break

            clean_images = batch.to(DEVICE)
            noise        = torch.randn_like(clean_images)
            bs           = clean_images.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=DEVICE
            ).long()

            # Forward pass in half precision
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                noise_pred   = unet(noisy_images, timesteps).sample
                loss         = F.mse_loss(noise_pred, noise)

            # Backward pass — with or without GradScaler
            if scaler is not None:
                # FP16 path: scale loss up before backward to prevent underflow,
                # then unscale before optimizer step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # BF16 path: no scaling needed
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % 500 == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                print(f"       Step {step}/{TRAIN_STEPS} | "
                      f"Loss: {loss.item():.5f} | LR: {current_lr:.2e}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(unet.state_dict(), ckpt_path)
    print(f"    -> Saved checkpoint: {ckpt_path}")

    unet.eval()
    return unet


# ----------------------------- STAGE 2 : INJECTION -------------------
@torch.no_grad()
def ddim_invert(init_image, unet, inv_scheduler):
    inv_scheduler.set_timesteps(DDIM_INFERENCE_STEPS)
    shallow_steps = max(1, int(SHALLOW_RATIO * DDIM_INFERENCE_STEPS))

    latents = init_image.to(DEVICE)

    with torch.autocast(device_type="cuda", dtype=DTYPE):
        for t in inv_scheduler.timesteps[:shallow_steps]:
            noise_pred = unet(latents, t).sample
            latents    = inv_scheduler.step(noise_pred, t, latents).prev_sample

    return latents


@torch.no_grad()
def ddim_denoise(x_ts, unet, scheduler):
    """Stage 2b: Biased denoising from x_{T_S} back to x_0."""
    scheduler.set_timesteps(DDIM_INFERENCE_STEPS)
    shallow_steps = max(1, int(SHALLOW_RATIO * DDIM_INFERENCE_STEPS))

    latents = x_ts

    with torch.autocast(device_type="cuda", dtype=DTYPE):
        for t in scheduler.timesteps[-shallow_steps:]:
            noise_pred = unet(latents, t).sample
            latents    = scheduler.step(noise_pred, t, latents).prev_sample

    return latents


# ----------------------------- STAGE 3 : REFINEMENT ------------------
@torch.no_grad()
def refine(x_f, x_clean, unet, scheduler):
    alphas_cumprod       = scheduler.alphas_cumprod.to(DEVICE)
    t_l                  = torch.tensor([REFINE_T_L], device=DEVICE)
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alphas_cumprod[REFINE_T_L])

    x_f     = x_f.clone().to(DEVICE).float()
    x_clean = x_clean.to(DEVICE).float()

    for i in range(REFINE_ITERS):
        tiny_noise = torch.randn_like(x_f)
        x_f_noisy  = (
            torch.sqrt(alphas_cumprod[REFINE_T_L]) * x_f
            + sqrt_one_minus_alpha * tiny_noise
        )

        # UNet forward in half precision
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            eps_pred = unet(x_f_noisy, t_l).sample

        # Score and gradient arithmetic in FP32 for numerical safety
        score         = -eps_pred.float() / sqrt_one_minus_alpha
        fidelity_grad = 2.0 * (x_f - x_clean)
        grad          = score - REFINE_LAMBDA * fidelity_grad

        x_f = x_f + REFINE_ETA * grad

        if (i + 1) % 25 == 0:
            print(f"         refine step {i + 1}/{REFINE_ITERS}")

    return x_f


# ----------------------------- FULL PIPELINE -------------------------
def build_schedulers():
    ddim = DDIMScheduler.from_pretrained(BASE_MODEL)
    ddim.config.clip_sample = False   # must be False — clipping breaks DDIM inversion
    ddim_inv = DDIMInverseScheduler.from_config(ddim.config)
    return ddim, ddim_inv


def inject_watermark(clean_path, unet, scheduler, inv_scheduler, original_size, img_size):
    init_image = load_image_tensor(clean_path, img_size).to(DEVICE)

    x_ts        = ddim_invert(init_image, unet, inv_scheduler)
    x_f         = ddim_denoise(x_ts, unet, scheduler)
    x_f_refined = refine(x_f, init_image, unet, scheduler)

    return tensor_to_pil(x_f_refined, original_size)


# ------------------------------- MAIN ---------------------------------
def main():
    # Print GPU info for sanity check
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"DTYPE  : {DTYPE}  |  GradScaler: {USE_SCALER}")

    if not DATASET_DIR.exists():
        if not os.path.exists(ZIP_FILE):
            raise FileNotFoundError(f"Could not find {ZIP_FILE}.")
        print(f"Unzipping {ZIP_FILE} ...")
        with zipfile.ZipFile(ZIP_FILE, "r") as z:
            z.extractall(".")

    TEMP_OUT_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)

    target_dir = DATASET_DIR / "clean_targets"
    ddim_scheduler, ddim_inv_scheduler = build_schedulers()

    total = 0
    for wm, start, stop in CATEGORIES:
        print(f"\n{'='*55}")
        print(f"  Category: {wm}  |  Targets: {start} - {stop}")
        print(f"{'='*55}")

        src_dir   = DATASET_DIR / "watermarked_sources" / wm
        src_paths = sorted(src_dir.glob("*.png"))
        if not src_paths:
            print(f"  [warn] No sources found in {src_dir}, skipping.")
            continue

        # Flush any leftover GPU memory from the previous category
        torch.cuda.empty_cache()
        if DEVICE == "cuda":
            free_gb = (torch.cuda.get_device_properties(0).total_memory
                       - torch.cuda.memory_allocated()) / 1e9
            print(f"    -> Free VRAM before training: {free_gb:.1f} GB")

        cat_size     = get_category_size(src_paths)
        process_size = get_process_size(cat_size)
        if process_size != cat_size:
            print(f"    -> Native size {cat_size}x{cat_size} remapped to "
                  f"{process_size}x{process_size} for processing "
                  f"(matches pretrained backbone's native scale)")
        grp_targets = [target_dir / f"{n}.png" for n in range(start, stop + 1)]

        # Stage 1: train or load checkpoint
        biased_unet = train_wm_model(src_paths, wm, img_size=process_size)

        # Stages 2 + 3: inject into each target
        print(f"    -> Forging {len(grp_targets)} targets at "
              f"{process_size}x{process_size}...")
        for tp in grp_targets:
            original_size = Image.open(tp).size
            print(f"       {tp.name}  native={original_size}")
            forged_pil = inject_watermark(
                tp, biased_unet,
                ddim_scheduler, ddim_inv_scheduler,
                original_size,
                img_size=process_size
            )
            forged_pil.save(TEMP_OUT_DIR / tp.name)
            total += 1

        del biased_unet
        torch.cuda.empty_cache()

    print(f"\nForged {total} images total.")
    if total != 200:
        print(f"[WARNING] Expected 200, got {total} -- submission may be rejected!")

    print(f"\nPackaging into {FILE_PATH} ...")
    with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in sorted(TEMP_OUT_DIR.glob("*.png")):
            zf.write(img, arcname=img.name)
    print("Done.")


if __name__ == "__main__":
    main()