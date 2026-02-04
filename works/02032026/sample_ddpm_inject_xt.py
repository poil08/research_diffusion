#!/usr/bin/env python3
import os, json, argparse
from typing import Any, Dict, Tuple, Optional, List

import torch
from torchvision.utils import make_grid, save_image
from diffusers import UNet2DModel, DDPMScheduler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Loaders (Diffusers folders)
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
# DDPM math (manual epsilon DDPM)
# -----------------------------
@torch.no_grad()
def ddpm_prepare(scheduler: DDPMScheduler):
    pred_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if pred_type != "epsilon":
        raise ValueError(f"prediction_type={pred_type} not supported (need epsilon).")

    betas = scheduler.betas.to(DEVICE, dtype=torch.float32)
    alphas = (1.0 - betas).to(DEVICE, dtype=torch.float32)
    acp = torch.cumprod(alphas, dim=0)
    acp_prev = torch.cat([torch.ones(1, device=DEVICE, dtype=torch.float32), acp[:-1]], dim=0)
    post_var = betas * (1.0 - acp_prev) / (1.0 - acp)
    T = int(scheduler.config.num_train_timesteps)
    return betas, alphas, acp, acp_prev, post_var, T


@torch.no_grad()
def ddpm_step(unet: UNet2DModel, betas, alphas, acp, acp_prev, post_var,
              x: torch.Tensor, t: int, g_step: torch.Generator) -> torch.Tensor:
    t_tensor = torch.tensor([t], device=DEVICE, dtype=torch.long)
    eps = unet(x, t_tensor).sample

    a_bar = acp[t]
    sqrt_a_bar = torch.sqrt(torch.clamp(a_bar, min=1e-20))
    sqrt_1m_a_bar = torch.sqrt(torch.clamp(1.0 - a_bar, min=1e-20))

    # x0 estimate
    x0 = (x - sqrt_1m_a_bar * eps) / sqrt_a_bar

    beta_t = betas[t]
    alpha_t = alphas[t]
    a_bar_prev = acp_prev[t]
    denom = torch.clamp(1.0 - a_bar, min=1e-20)

    mean = (
        torch.sqrt(a_bar_prev) * beta_t / denom * x0
        + torch.sqrt(alpha_t) * (1.0 - a_bar_prev) / denom * x
    )

    if t > 0:
        z_cpu = torch.randn(x.shape, generator=g_step, device="cpu", dtype=torch.float32)
        z = z_cpu.to(DEVICE)
        x = mean + torch.sqrt(torch.clamp(post_var[t], min=1e-20)) * z
    else:
        x = mean

    return x


@torch.no_grad()
def resume_from_xt(unet: UNet2DModel, scheduler: DDPMScheduler, xt_cpu: torch.Tensor,
                   start_t: int, step_seed: int) -> torch.Tensor:
    betas, alphas, acp, acp_prev, post_var, T = ddpm_prepare(scheduler)
    if not (0 <= start_t < T):
        raise ValueError(f"start_t must be in [0,{T-1}], got {start_t}")

    g_step = torch.Generator(device="cpu").manual_seed(step_seed)

    x = xt_cpu.to(DEVICE, dtype=torch.float32)
    for t in range(start_t, -1, -1):
        x = ddpm_step(unet, betas, alphas, acp, acp_prev, post_var, x, t, g_step)

    # model space [-1,1] -> [0,1]
    x = (x.clamp(-1, 1) + 1) / 2.0
    return x.detach().cpu()


def save_grid(imgs01: torch.Tensor, out_path: str, nrow: int):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    grid = make_grid(imgs01, nrow=nrow, padding=2)
    save_image(grid, out_path)


def load_xt_from_pt(pt_path: str) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict):
        # support your payload style
        if "x_t" in obj:
            return obj["x_t"], obj.get("meta", None)
        # sometimes you might store directly as {"tensor": ...}
    if torch.is_tensor(obj):
        return obj, None
    raise ValueError(f"Unsupported .pt format in {pt_path}. Expected Tensor or dict with x_t.")


# -----------------------------
# Main
# -----------------------------
def run(args):
    os.makedirs(args.out_dir, exist_ok=True)

    unet_b, unet_path_b = load_unet(args.model_b, args.prefer_ema)
    sch_b, cfg_b, sch_path_b = load_scheduler(args.model_b)

    print("Device:", DEVICE)
    print("Model B UNet:", unet_path_b)
    print("Model B Scheduler:", sch_path_b)

    # Format: --inject xt_path:start_t xt_path:start_t ...
    for item in args.inject:
        if ":" not in item:
            raise ValueError(f"--inject item must be 'path.pt:start_t' but got: {item}")
        pt_path, start_t_str = item.rsplit(":", 1)
        start_t = int(start_t_str)

        xt_cpu, meta = load_xt_from_pt(pt_path)

        # Helpful warning if meta says a different timestep than you're injecting as
        if meta is not None and isinstance(meta, dict) and "t" in meta:
            t_saved = int(meta["t"])
            if t_saved != start_t:
                print(f"\n[WARNING] Injecting tensor saved at t={t_saved} as start_t={start_t}. "
                      f"(This is a deliberate mismatch experiment.)")

        # Run B from start_t -> 0
        out = resume_from_xt(unet_b, sch_b, xt_cpu, start_t=start_t, step_seed=args.step_seed)

        base = os.path.splitext(os.path.basename(pt_path))[0]
        subdir = os.path.join(args.out_dir, f"{base}_as_t{start_t:04d}")
        os.makedirs(subdir, exist_ok=True)

        png_path = os.path.join(subdir, "B_final.png")
        save_grid(out, png_path, nrow=args.nrow)

        if args.save_final_pt:
            torch.save({"final": out, "meta_in": meta, "injected_start_t": start_t}, os.path.join(subdir, "B_final.pt"))

        print(f"\nSaved: {png_path}")
        if args.save_final_pt:
            print(f"Saved: {os.path.join(subdir, 'B_final.pt')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_b", required=True, help="Model B directory (diffusers format)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefer_ema", action="store_true")
    ap.add_argument("--step_seed", type=int, default=777)
    ap.add_argument("--nrow", type=int, default=8)

    ap.add_argument(
        "--inject",
        nargs="+",
        required=True,
        help="One or more items: path_to_xt.pt:start_t  (e.g., x_t0999_batch00.pt:998)"
    )

    ap.add_argument("--save_final_pt", action="store_true", help="Also save final outputs as .pt tensors")
    args = ap.parse_args()
    run(args)
