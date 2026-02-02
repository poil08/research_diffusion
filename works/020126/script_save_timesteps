#source /yunity/poil08/hugging_face/.venv/bin/activate

"""
python save_timesteps_modelA.py \
  --model_a /yunity/poil08/hugging_face/hf_cifar10_ddpm_rgb_seed0 \
  --out_dir /yunity/poil08/cifar_branch/modelA_seed0_saved_xt \
  --num_samples 64 --batch_size 64 \
  --channels 3 \
  --prefer_ema \
  --repro_flags \
  --save_xt_png \
  --save_timesteps 999 900 800 700 600 500 400 300 200 100 0
"""

#!/usr/bin/env python3
import os, json, argparse, hashlib
from typing import List, Dict, Any

import torch
from torchvision.utils import make_grid, save_image
from diffusers import UNet2DModel, DDPMScheduler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Small utils
# ============================================================
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_tensor(x: torch.Tensor) -> str:
    arr = x.detach().cpu().to(torch.float32).contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()

def set_repro_flags(enable: bool):
    if not enable:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # DO NOT call torch.use_deterministic_algorithms(True) unless you also set CUBLAS_WORKSPACE_CONFIG.

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


# ============================================================
# Loaders (Diffusers-style folders)
# ============================================================
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

    raise FileNotFoundError(f"UNet not found under {model_dir} (expected unet/ or unet_ema/)")

def find_scheduler_config(model_dir: str) -> str:
    model_dir = os.path.abspath(model_dir)
    for root, _, files in os.walk(model_dir):
        if "scheduler_config.json" in files:
            return os.path.join(root, "scheduler_config.json")
    raise FileNotFoundError(f"scheduler_config.json not found under {model_dir}")

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


# ============================================================
# Deterministic noise
# ============================================================
def make_base_noise(num: int, batch: int, c: int, h: int, seed: int) -> List[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    remain = num
    while remain > 0:
        cur = min(batch, remain)
        out.append(torch.randn(cur, c, h, h, generator=g, device="cpu", dtype=torch.float32))
        remain -= cur
    return out


# ============================================================
# Manual DDPM step + x0 prediction (with clip_sample support)
# ============================================================
@torch.no_grad()
def ddpm_prepare(scheduler: DDPMScheduler):
    pred_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if pred_type != "epsilon":
        raise ValueError(f"prediction_type={pred_type} not supported here (need epsilon).")

    betas = scheduler.betas.to(DEVICE, dtype=torch.float32)
    alphas = (1.0 - betas).to(DEVICE, dtype=torch.float32)
    acp = torch.cumprod(alphas, dim=0)
    acp_prev = torch.cat([torch.ones(1, device=DEVICE, dtype=torch.float32), acp[:-1]], dim=0)
    post_var = betas * (1.0 - acp_prev) / (1.0 - acp)

    T = int(scheduler.config.num_train_timesteps)
    return betas, alphas, acp, acp_prev, post_var, T

@torch.no_grad()
def ddpm_step_and_x0(unet: UNet2DModel,
                     scheduler: DDPMScheduler,
                     betas, alphas, acp, acp_prev, post_var,
                     x: torch.Tensor, t: int, g_step_cpu: torch.Generator):
    """
    Returns (x_{t-1}, x0_pred).
    Adds clip_sample behavior to match Diffusers configs.
    """
    t_tensor = torch.tensor([t], device=DEVICE, dtype=torch.long)
    eps = unet(x, t_tensor).sample

    a_bar = acp[t]
    sqrt_a_bar = torch.sqrt(torch.clamp(a_bar, min=1e-20))
    sqrt_1m_a_bar = torch.sqrt(torch.clamp(1.0 - a_bar, min=1e-20))

    # x0 prediction
    x0 = (x - sqrt_1m_a_bar * eps) / sqrt_a_bar

    # --- IMPORTANT: match scheduler clip behavior (if enabled) ---
    if getattr(scheduler.config, "clip_sample", False):
        clip_range = float(getattr(scheduler.config, "clip_sample_range", 1.0))
        x0 = x0.clamp(-clip_range, clip_range)

    beta_t = betas[t]
    alpha_t = alphas[t]
    a_bar_prev = acp_prev[t]
    denom = torch.clamp(1.0 - a_bar, min=1e-20)

    mean = (
        torch.sqrt(a_bar_prev) * beta_t / denom * x0
        + torch.sqrt(alpha_t) * (1.0 - a_bar_prev) / denom * x
    )

    if t > 0:
        z_cpu = torch.randn(x.shape, generator=g_step_cpu, device="cpu", dtype=torch.float32)
        z = z_cpu.to(DEVICE)
        next_x = mean + torch.sqrt(torch.clamp(post_var[t], min=1e-20)) * z
    else:
        next_x = mean

    return next_x, x0


# ============================================================
# Saving helpers
# ============================================================
def save_pt(payload: Dict[str, Any], out_path: str):
    ensure_dir(os.path.dirname(out_path) or ".")
    torch.save(payload, out_path)

def to_png01_from_model_space(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu().to(torch.float32).clamp(-1, 1)
    return (x + 1.0) / 2.0

def to_png01_quickview_xt(x_t: torch.Tensor) -> torch.Tensor:
    """
    Only for sanity; x_t is not pixel space.
    """
    x = x_t.detach().cpu().to(torch.float32)
    mean = x.mean(dim=(1,2,3), keepdim=True)
    std = x.std(dim=(1,2,3), keepdim=True).clamp(min=1e-6)
    x = (x - mean) / (3.0 * std)
    x = x.clamp(-1, 1)
    return (x + 1.0) / 2.0

def save_grid_png(imgs01_cpu: torch.Tensor, out_path: str, nrow: int):
    ensure_dir(os.path.dirname(out_path) or ".")
    grid = make_grid(imgs01_cpu, nrow=nrow, padding=2)
    save_image(grid, out_path)


# ============================================================
# Main
# ============================================================
@torch.no_grad()
def run(args):
    set_repro_flags(args.repro_flags)

    unet, unet_path = load_unet(args.model_a, args.prefer_ema)
    scheduler, sch_cfg, sch_path = load_scheduler(args.model_a)
    betas, alphas, acp, acp_prev, post_var, T = ddpm_prepare(scheduler)

    sample_size = getattr(unet.config, "sample_size", None)
    if args.image_size is None:
        if sample_size is None:
            raise ValueError("Could not infer image_size; please pass --image_size.")
        args.image_size = int(sample_size)

    if sample_size is not None and int(sample_size) != int(args.image_size):
        raise ValueError(f"image_size mismatch: --image_size={args.image_size} but UNet sample_size={sample_size}")

    save_ts = [int(t) for t in args.save_timesteps]
    for t in save_ts:
        if t < 0 or t >= T:
            raise ValueError(f"Requested timestep {t} invalid (valid 0..{T-1}).")
    save_ts_set = set(save_ts)

    print("Device:", DEVICE)
    print("UNet:", unet_path)
    print("Scheduler:", sch_path)
    print("num_train_timesteps:", T)
    print("Saving timesteps:", save_ts)
    print("clip_sample:", getattr(scheduler.config, "clip_sample", False),
          "clip_range:", getattr(scheduler.config, "clip_sample_range", None))

    noise_batches = make_base_noise(
        num=args.num_samples,
        batch=args.batch_size,
        c=args.channels,
        h=args.image_size,
        seed=args.noise_seed,
    )
    print("Base noise sha256(first batch):", sha256_tensor(noise_batches[0]))

    g_step = torch.Generator(device="cpu").manual_seed(args.step_seed)

    pt_dir = os.path.join(args.out_dir, "pt")
    png_dir = os.path.join(args.out_dir, "png")
    ensure_dir(pt_dir)
    ensure_dir(png_dir)

    sch_hash = sha256_bytes(json.dumps(sch_cfg, sort_keys=True).encode("utf-8"))
    saved_counts = {t: 0 for t in save_ts}

    for b_idx, z0_cpu in enumerate(noise_batches):
        x = z0_cpu.to(DEVICE, dtype=torch.float32)  # x_{T-1}

        for t in range(T - 1, -1, -1):
            next_x, x0_pred = ddpm_step_and_x0(
                unet, scheduler, betas, alphas, acp, acp_prev, post_var, x, t, g_step
            )

            if t in save_ts_set:
                meta = {
                    "model_dir": os.path.abspath(args.model_a),
                    "unet_path": os.path.abspath(unet_path),
                    "scheduler_path": os.path.abspath(sch_path),
                    "scheduler_sha256": sch_hash,
                    "t": t,
                    "batch_index": b_idx,
                    "num_samples_total": args.num_samples,
                    "batch_size": args.batch_size,
                    "noise_seed": args.noise_seed,
                    "step_seed": args.step_seed,
                    "channels": args.channels,
                    "image_size": args.image_size,
                    "prefer_ema": bool(args.prefer_ema),
                }

                # Save noisy x_t for branching
                save_pt(
                    {"x_t": x.detach().cpu().contiguous(), "meta": meta},
                    os.path.join(pt_dir, f"x_t{t:04d}_batch{b_idx:02d}.pt")
                )

                # Save x0_pred for analysis/sanity
                save_pt(
                    {"x0_pred": x0_pred.detach().cpu().contiguous(), "meta": meta},
                    os.path.join(pt_dir, f"x0pred_t{t:04d}_batch{b_idx:02d}.pt")
                )

                # PNG sanity: x0_pred (meaningful)
                save_grid_png(
                    to_png01_from_model_space(x0_pred),
                    os.path.join(png_dir, f"x0pred_t{t:04d}_batch{b_idx:02d}.png"),
                    nrow=args.nrow
                )

                # Optional: quickview of x_t (often saturates)
                if args.save_xt_png:
                    save_grid_png(
                        to_png01_quickview_xt(x.detach().cpu()),
                        os.path.join(png_dir, f"xt_quickview_t{t:04d}_batch{b_idx:02d}.png"),
                        nrow=args.nrow
                    )

                saved_counts[t] += x.shape[0]

            x = next_x  # advance

        # After finishing t=0, x is the final sample x_0
        final_png = to_png01_from_model_space(x)
        save_grid_png(
            final_png,
            os.path.join(png_dir, f"final_x0_batch{b_idx:02d}.png"),
            nrow=args.nrow
        )

    print("\nDone. Saved counts per timestep (should equal num_samples):")
    for t in save_ts:
        print(f"  t={t:4d}: {saved_counts[t]} samples")

    print("\nOutputs:")
    print("  PT tensors:", pt_dir)
    print("  PNG grids:", png_dir)
    print("  Final images:", os.path.join(png_dir, "final_x0_batch*.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--prefer_ema", action="store_true")
    ap.add_argument("--repro_flags", action="store_true")

    ap.add_argument("--num_samples", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--image_size", type=int, default=None)
    ap.add_argument("--channels", type=int, default=3)

    ap.add_argument("--noise_seed", type=int, default=1234)
    ap.add_argument("--step_seed", type=int, default=777)

    ap.add_argument("--nrow", type=int, default=8)

    ap.add_argument(
        "--save_timesteps",
        type=int,
        nargs="+",
        default=[999, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0],
    )

    ap.add_argument("--save_xt_png", action="store_true")

    args = ap.parse_args()
    run(args)
