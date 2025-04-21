import os
import sys
import torch
import numpy as np
from weight_diffusion import WeightUNet, WeightDiffusion, load_checkpoint_weights

def summary(arr, name):
    a = arr.ravel()
    print(f"{name:20s} mean={a.mean():.4e} std={a.std():.4e} min={a.min():.4e} max={a.max():.4e}", flush=True)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unet_ckpt",
        default="diffusion_checkpoints/diffusion_epoch_45.pth",
        help="Path to diffusion UNet .pth"
    )
    parser.add_argument(
        "--tiny_ckpt",
        default="./CNN_checkpoints/run_0/1_run0_save052_epoch0074.pth",
        help="One TinyCNN checkpoint (for input_size inference)"
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    print("=== DEBUG START ===", flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # 1) Load and summarize real TinyCNN weights
    real = load_checkpoint_weights(args.tiny_ckpt).cpu().numpy()
    summary(real, "REAL weights")
    input_size = real.size

    # 2) Load full‑size UNet
    unet = WeightUNet(input_size).cpu()
    state = torch.load(args.unet_ckpt, map_location="cpu")
    unet.load_state_dict(state)
    unet.to(device).eval()
    diffusion = WeightDiffusion(unet)

    # 2a) Patch p_sample to accept tensor or int
    orig_p = diffusion.p_sample
    def p_sample_wrapped(x_t, t_tensor):
        t_int = int(t_tensor[0].item())
        return orig_p(x_t, t_int)
    diffusion.p_sample = p_sample_wrapped

    # Free GPU cache
    torch.cuda.empty_cache()

    # 3) Prepare a batch of x_t
    B = 16
    x = torch.randn(B, input_size, device=device)

    # 4) Timesteps to log
    checkpoints = {999, 900, 800, 600, 400, 200, 0}

    # 5) Reverse‑diffusion with debug summaries
    for t in reversed(range(diffusion.timesteps)):
        t_vec = torch.full((B,), t, device=device, dtype=torch.long)

        # predict noise
        with torch.no_grad():
            eps = unet(x, t_vec).cpu().numpy()

        # compute x0_pred
        alpha_bar = diffusion.alphas_cumprod[t].item()
        sqrt1m     = np.sqrt(1 - alpha_bar)
        sqrt_ab    = np.sqrt(alpha_bar)
        x_np       = x.cpu().numpy()
        x0_pred    = (x_np - sqrt1m * eps) / sqrt_ab

        if t in checkpoints:
            print(f"\n--- Timestep t={t} ---", flush=True)
            summary(x_np,    "x_t")
            summary(eps,     "eps_pred")
            summary(x0_pred, "x0_pred")

        # one reverse step
        with torch.no_grad():
            x = diffusion.p_sample(x, t_vec)

    # 6) Final sampled x0
    x0 = x.cpu().numpy()
    print("\n--- After final p_sample (t=0) ---", flush=True)
    summary(x0, "sampled x_0")
    print("=== DEBUG END ===", flush=True)
