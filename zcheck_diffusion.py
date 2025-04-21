#!/usr/bin/env python3
# zcheck_diffusion.py

import os
import torch
import numpy as np
import argparse
from weight_diffusion import WeightUNet, WeightDiffusion

def load_weights_cpu(path):
    """Load a checkpoint's state_dict onto CPU and flatten to a vector."""
    ck = torch.load(path, map_location="cpu")["state_dict"]
    parts = [v.view(-1) for v in ck.values()]
    return torch.cat(parts)

def summary_stats(arr, name):
    """Print basic distribution statistics for a 1D numpy array."""
    print(f"\n--- {name} distribution ---")
    print(f" count: {arr.size}")
    print(f"   mean: {arr.mean():.6e}")
    print(f"    std: {arr.std():.6e}")
    print(f"    min: {arr.min():.6e}")
    print(f"    5%:  {np.percentile(arr, 5):.6e}")
    print(f"   25%:  {np.percentile(arr,25):.6e}")
    print(f"   50%:  {np.percentile(arr,50):.6e}")
    print(f"   75%:  {np.percentile(arr,75):.6e}")
    print(f"   95%:  {np.percentile(arr,95):.6e}")
    print(f"    max: {arr.max():.6e}")

def check_noise_prediction(diffusion, device, input_size):
    print("\n=== Noise Prediction Statistics ===")
    for t in [0, diffusion.timesteps//4, diffusion.timesteps//2,
              3*diffusion.timesteps//4, diffusion.timesteps-1]:
        x = torch.randn(4, input_size, device=device)
        t_vec = torch.full((4,), t, dtype=torch.long, device=device)
        with torch.no_grad():
            eps = diffusion.model(x, t_vec).cpu().numpy().ravel()
        print(f"t={t:4d} | mean={eps.mean():.4e}, std={eps.std():.4e}, "
              f"min={eps.min():.4e}, max={eps.max():.4e}")

def sample_weights(diffusion, device, input_size):
    x = torch.randn(1, input_size, device=device)
    with torch.no_grad():
        for t in reversed(range(diffusion.timesteps)):
            x = diffusion.p_sample(x, torch.tensor([t], device=device))
    return x.squeeze(0).cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug checks for diffusion UNet sampling"
    )
    parser.add_argument("--unet_ckpt", required=False, default="diffusion_checkpoints/diffusion_epoch_45.pth", 
                        help="Path to diffusion UNet .pth")
    parser.add_argument("--tiny_dir", default="./CNN_checkpoints/run_0",
                        help="One TinyCNN checkpoint folder for input_size")
    parser.add_argument(
        "--device", default="cuda",
        help="Device to run sampling on (cuda or cpu)"
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Infer input_size from an existing TinyCNN checkpoint
    tiny_ckpt = next(f for f in os.listdir(args.tiny_dir) if f.endswith(".pth"))
    tiny_path = os.path.join(args.tiny_dir, tiny_ckpt)
    real_weights = load_weights_cpu(tiny_path).numpy()
    input_size = real_weights.size
    print("Input size:", input_size)

    # 2) Load and prepare the full‐size UNet + diffusion
    unet = WeightUNet(input_size).to("cpu")
    state = torch.load(args.unet_ckpt, map_location="cpu")
    unet.load_state_dict(state)
    unet = unet.to(device).eval()
    diffusion = WeightDiffusion(unet)

    # 3) Check noise prediction at several timesteps
    check_noise_prediction(diffusion, device, input_size)

    # 4) Sample one final weight vector
    print("\nSampling final weight vector...")
    sampled_weights = sample_weights(diffusion, device, input_size)

    # 5) Compare distributions numerically
    summary_stats(real_weights,   "Real checkpoint weights")
    summary_stats(sampled_weights,"Sampled weights")

    # 6) Compute RMSE and Pearson correlation
    rmse = np.sqrt(((real_weights - sampled_weights)**2).mean())
    corr = np.corrcoef(real_weights, sampled_weights)[0,1]
    print(f"\nRMSE(real, sampled)     = {rmse:.6e}")
    print(f"Pearson correlation     = {corr:.6f}")
