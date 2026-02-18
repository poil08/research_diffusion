#!/usr/bin/env python3
import os, json, argparse, hashlib
from typing import Any, Dict, Tuple, Optional

import torch
from torchvision.utils import make_grid, save_image
from diffusers import UNet2DModel, DDPMScheduler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# -----------------------------
# Loaders
# -----------------------------
def pick_unet_dir(model_dir: str, prefer_ema: bool) -> str:
    model_dir = os.path.abspath(model_dir)

    if os.path.basename(model_dir) in ("unet", "unet_ema") and os.path.isdir(model_dir):
        return model_dir

    ema = os.path.join(model_dir, "unet_ema")
    unet = os.path.join(model_dir, "unet")
    if prefer_ema and os.path.isdir(ema):
        return ema
    if os.path.isdir(unet):
        return unet
    if os.path.isdir(ema):
        return ema
    raise FileNotFoundError("UNet not found")


def find_scheduler_config(model_dir: str) -> str:
    for root, _, files in os.walk(model_dir):
        if "scheduler_config.json" in files:
            return os.path.join(root, "scheduler_config.json")
    raise FileNotFoundError("scheduler_config.json not found")


def load_unet(model_dir: str, prefer_ema: bool):
    unet_dir = pick_unet_dir(model_dir, prefer_ema)
    model = UNet2DModel.from_pretrained(unet_dir).to(DEVICE).eval()
    return model, unet_dir


def load_scheduler(model_dir: str):
    cfg_path = find_scheduler_config(model_dir)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    sch = DDPMScheduler.from_config(cfg)
    return sch, cfg, cfg_path


# -----------------------------
# Sampling WITHOUT step seed
# -----------------------------
@torch.no_grad()
def sample_ddpm(unet,
                scheduler,
                num_samples,
                batch_size,
                channels,
                image_size,
                noise_seed):

    T = int(scheduler.config.num_train_timesteps)
    scheduler.set_timesteps(T, device=DEVICE)

    # ONLY initial noise seed (optional)
    g_noise = None
    if noise_seed is not None:
        g_noise = torch.Generator(device="cpu").manual_seed(int(noise_seed))

    outs = []
    remain = num_samples

    while remain > 0:
        cur = min(batch_size, remain)

        # Initial noise
        x = torch.randn(
            cur, channels, image_size, image_size,
            generator=g_noise,
            device="cpu",
            dtype=torch.float32
        ).to(DEVICE)

        # Reverse diffusion (fully stochastic)
        for t in scheduler.timesteps:
            eps = unet(x, t).sample

            # NO generator passed → stochastic each run
            step = scheduler.step(eps, t, x)
            x = step.prev_sample

        x01 = (x.clamp(-1, 1) + 1) / 2.0
        outs.append(x01.detach().cpu())

        remain -= cur

    return torch.cat(outs, dim=0)


def save_grid(imgs, path, nrow):
    ensure_dir(os.path.dirname(path) or ".")
    save_image(make_grid(imgs, nrow=nrow, padding=2), path)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_a", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefer_ema", action="store_true")

    ap.add_argument("--num_samples", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=None)
    ap.add_argument("--channels", type=int, default=3)

    # ONLY noise seed now
    ap.add_argument("--noise_seed", type=int, default=None)

    ap.add_argument("--nrow", type=int, default=8)

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    unet, _ = load_unet(args.model_a, args.prefer_ema)
    scheduler, cfg, _ = load_scheduler(args.model_a)

    if args.image_size is None:
        args.image_size = int(unet.config.sample_size)

    print("Device:", DEVICE)
    print("Noise seed:", args.noise_seed)

    imgs = sample_ddpm(
        unet,
        scheduler,
        args.num_samples,
        args.batch_size,
        args.channels,
        args.image_size,
        args.noise_seed
    )

    save_grid(imgs, os.path.join(args.out_dir, "final_grid.png"), args.nrow)

    print("\nSaved:", os.path.join(args.out_dir, "final_grid.png"))


if __name__ == "__main__":
    main()
