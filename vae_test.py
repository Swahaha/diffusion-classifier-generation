import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 1) TinyCNN definition (must match your original)
# ──────────────────────────────────────────────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 100), nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ──────────────────────────────────────────────────────────────────────────────
# 2) VAE & Diffusion definitions (must match your training script)
# ──────────────────────────────────────────────────────────────────────────────
from vae_diffusion import WeightVAE, DiffUNet, Diffusion


# ──────────────────────────────────────────────────────────────────────────────
# 3) Helpers to map flat vectors ↔ state_dict
# ──────────────────────────────────────────────────────────────────────────────
def vector_to_state_dict(vector: torch.Tensor, model: nn.Module):
    sd = model.state_dict()
    new_sd = {}
    idx = 0
    for k, v in sd.items():
        numel = v.numel()
        new_sd[k] = vector[idx:idx+numel].view_as(v)
        idx += numel
    return new_sd


# ──────────────────────────────────────────────────────────────────────────────
# 4) Main: load, sample, evaluate
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}\n')

    # ─── paths to your saved checkpoints ───
    vae_path      = './models/vae_final.pth'
    diffunet_path = './models/diffusion_unet_final.pth'

    # ─── 1) Infer dims from the VAE checkpoint ───
    print("Loading VAE checkpoint to infer dimensions...")
    vae_ckpt = torch.load(vae_path, map_location='cpu')
    # last decoder layer is at index 4 in your Sequential: Linear(2048, input_dim)
    dec_w = vae_ckpt['decoder.4.weight']  # shape [input_dim, 2048]
    input_dim  = dec_w.shape[0]
    latent_dim = vae_ckpt['mu_layer.weight'].shape[0]
    print(f"  inferred input_dim  = {input_dim}")
    print(f"  inferred latent_dim = {latent_dim}\n")

    # ─── 2) Instantiate & load the VAE ───
    vae = WeightVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    vae.load_state_dict(vae_ckpt)
    vae.eval()
    print("Loaded VAE ✓\n")

    # ─── 3) Instantiate & load the Diffusion UNet ───
    diffunet_ckpt = torch.load(diffunet_path, map_location='cpu')
    diffunet = DiffUNet(latent_dim=latent_dim).to(device)
    diffunet.load_state_dict(diffunet_ckpt)
    diffusion = Diffusion(diffunet)
    diffunet.eval()
    print("Loaded DiffUNet & reconstructed schedule ✓\n")

    # ─── 4) Sample N weight‑vectors in latent space → decode → flatten → CPU ───
    N = 5
    print(f"Sampling {N} new TinyCNN weight‑vectors…")
    z = torch.randn(N, latent_dim, device=device)
    for t in reversed(range(diffusion.timesteps)):
        z = diffusion.p_sample(z, t)
    with torch.no_grad():
        w_vectors = vae.decoder(z).cpu()
    print("Sampling done!\n")

    # ─── 5) Prepare CIFAR‑10 validation set ───
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
    val_ds = CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    val_loader = DataLoader(val_ds, batch_size=100, shuffle=False, num_workers=4)

    class_names = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

    # ─── 6) For each sampled vector: build TinyCNN, load weights, eval ───
    for i, w_vec in enumerate(w_vectors):
        print(f"\n=== Evaluating sample #{i+1} ===")
        model = TinyCNN().to(device)
        model_sd = vector_to_state_dict(w_vec, model)
        model.load_state_dict(model_sd)
        model.eval()

        y_true, y_pred = [], []
        for imgs, tgt in tqdm(val_loader, desc=f" Sample #{i+1}"):
            imgs, tgt = imgs.to(device), tgt.to(device)
            with torch.no_grad():
                logits = model(imgs)
            preds = logits.argmax(dim=1)
            y_true.extend(tgt.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

        acc = 100.0 * sum(yt==yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
        print(f"→ Accuracy: {acc:.2f}%")
        print("-- Classification Report --")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        print(df_cm)
