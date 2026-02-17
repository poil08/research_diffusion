#!/usr/bin/env python3
import os, json, argparse, hashlib
from typing import Any, Dict, Tuple, Optional

import torch
from torchvision.utils import make_grid, save_image
from diffusers import UNet2DModel, DDPMScheduler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Small utils
# -----------------------------
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# -----------------------------
# Loaders (Diffusers-style folders)
# -----------------------------
def pick_unet_dir(model_dir: str, prefer_ema: bool) -> str:
    model_dir = os.path.abspath(model_dir)

    # if user passes unet/ or unet_ema/ directly
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
    raise FileNotFoundError(f"UNet not found under {model_dir} (expected unet/ or unet_ema/)")


def find_scheduler_config(model_dir: str) -> str:
    model_dir = os.path.abspath(model_dir)
    for root, _, files in os.walk(model_dir):
        if "scheduler_config.json" in files:
            return os.path.join(root, "scheduler_config.json")
    raise FileNotFoundError(f"scheduler_config.json not found under {model_dir}")


def load_unet(model_dir: str, prefer_ema: bool) -> Tuple[UNet2DModel, str]:
    unet_dir = pick_unet_dir(model_dir, prefer_ema)
    model = UNet2DModel.from_pretrained(unet_dir).to(DEVICE).eval()
    return model, unet_dir


def load_scheduler(model_dir: str) -> Tuple[DDPMScheduler, Dict[str, Any], str]:
    cfg_path = find_scheduler_config(model_dir)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    sch = DDPMScheduler.from_config(cfg)
    return sch, cfg, cfg_path


# -----------------------------
# Sampling (Diffusers scheduler.step)
# -----------------------------
@torch.no_grad()
def sample_ddpm(unet: UNet2DModel,
                scheduler: DDPMScheduler,
                num_samples: int,
                batch_size: int,
                channels: int,
                image_size: int,
                noise_seed: Optional[int],
                step_seed: Optional[int]) -> torch.Tensor:
    """
    Returns images in [0,1] on CPU: (N,C,H,W)
    """

    # Set the scheduler timesteps explicitly
    T = int(scheduler.config.num_train_timesteps)
    scheduler.set_timesteps(T, device=DEVICE)

    # RNGs (CPU generators so GPU RNG differences don't matter)
    g_noise = torch.Generator(device="cpu")
    if noise_seed is not None:
        g_noise.manual_seed(int(noise_seed))

    g_step = torch.Generator(device="cpu")
    if step_seed is not None:
        g_step.manual_seed(int(step_seed))

    outs = []
    remain = num_samples
    while remain > 0:
        cur = min(batch_size, remain)

        # initial x_T noise on CPU -> move to GPU
        x = torch.randn(cur, channels, image_size, image_size,
                        generator=g_noise, device="cpu", dtype=torch.float32).to(DEVICE)

        for t in scheduler.timesteps:
            # DDPM uses epsilon prediction by default
            model_out = unet(x, t).sample

            # scheduler handles variance_type, clipping, etc.
            step = scheduler.step(model_out, t, x, generator=g_step)
            x = step.prev_sample

        # model space [-1,1] -> [0,1]
        x01 = (x.clamp(-1, 1) + 1) / 2.0
        outs.append(x01.detach().cpu())

        remain -= cur

    return torch.cat(outs, dim=0)


def save_grid(imgs01: torch.Tensor, out_path: str, nrow: int):
    ensure_dir(os.path.dirname(out_path) or ".")
    grid = make_grid(imgs01, nrow=nrow, padding=2)
    save_image(grid, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefer_ema", action="store_true")

    ap.add_argument("--num_samples", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--image_size", type=int, default=None)  # infer from unet.config.sample_size
    ap.add_argument("--channels", type=int, default=3)

    # Optional seeds:
    ap.add_argument("--noise_seed", type=int, default=None,
                    help="Seed for initial x_T noise. If omitted, random each run.")
    ap.add_argument("--step_seed", type=int, default=None,
                    help="Seed for per-step randomness in scheduler.step. If omitted, random each run.")

    ap.add_argument("--nrow", type=int, default=8)
    ap.add_argument("--save_pt", action="store_true", help="Also save final tensor as final.pt")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    unet, unet_path = load_unet(args.model_a, args.prefer_ema)
    scheduler, sch_cfg, sch_path = load_scheduler(args.model_a)

    # infer image size
    sample_size = getattr(unet.config, "sample_size", None)
    if args.image_size is None:
        if sample_size is None:
            raise ValueError("Could not infer image_size from UNet config; pass --image_size.")
        args.image_size = int(sample_size)

    # provenance
    sch_hash = sha256_bytes(json.dumps(sch_cfg, sort_keys=True).encode("utf-8"))
    meta = {
        "model_dir": os.path.abspath(args.model_a),
        "unet_path": os.path.abspath(unet_path),
        "scheduler_path": os.path.abspath(sch_path),
        "scheduler_sha256": sch_hash,
        "num_train_timesteps": int(scheduler.config.num_train_timesteps),
        "image_size": args.image_size,
        "channels": args.channels,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "prefer_ema": bool(args.prefer_ema),
        "noise_seed": args.noise_seed,
        "step_seed": args.step_seed,
        "device": DEVICE,
    }

    print("Device:", DEVICE)
    print("UNet:", unet_path)
    print("Scheduler:", sch_path)
    print("Timesteps:", meta["num_train_timesteps"])
    print("image_size:", args.image_size, "channels:", args.channels)
    print("noise_seed:", args.noise_seed, "step_seed:", args.step_seed)

    imgs01 = sample_ddpm(
        unet=unet,
        scheduler=scheduler,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        channels=args.channels,
        image_size=args.image_size,
        noise_seed=args.noise_seed,
        step_seed=args.step_seed,
    )

    # Save outputs
    save_grid(imgs01, os.path.join(args.out_dir, "final_grid.png"), nrow=args.nrow)
    if args.save_pt:
        torch.save(imgs01, os.path.join(args.out_dir, "final.pt"))

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print("\nSaved:")
    print(" ", os.path.join(args.out_dir, "final_grid.png"))
    if args.save_pt:
        print(" ", os.path.join(args.out_dir, "final.pt"))
    print(" ", os.path.join(args.out_dir, "meta.json"))


if __name__ == "__main__":
    main()
