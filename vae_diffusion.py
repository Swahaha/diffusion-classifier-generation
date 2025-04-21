import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1) Utilities: load and stack CNN checkpoints
# ──────────────────────────────────────────────────────────────────────────────
def find_checkpoints(dir_path, ext='.pth'):
    pts = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith(ext):
                pts.append(os.path.join(root, f))
    return sorted(pts)

def load_weights(path, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    sd   = ckpt.get('state_dict', ckpt)
    return torch.cat([v.view(-1) for v in sd.values()])

class WeightDataset(Dataset):
    """Wrap a stack of weight-vectors into a PyTorch Dataset."""
    def __init__(self, weights_tensor):
        self.weights = weights_tensor
    def __len__(self): return len(self.weights)
    def __getitem__(self, idx): return self.weights[idx]

# ──────────────────────────────────────────────────────────────────────────────
# 2) Stage1: Autoencoder for weight embeddings
# ──────────────────────────────────────────────────────────────────────────────
class WeightVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.ReLU(True),
            nn.Linear(2048, 1024), nn.ReLU(True)
        )
        self.mu_layer     = nn.Linear(1024, latent_dim)
        self.logvar_layer = nn.Linear(1024, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(True),
            nn.Linear(1024, 2048), nn.ReLU(True),
            nn.Linear(2048, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h      = self.encoder(x)
        mu     = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z      = self.reparameterize(mu, logvar)
        recon  = self.decoder(z)
        return recon, mu, logvar

    def loss_function(self, recon, x, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(recon, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return recon_loss + beta * kld, recon_loss.item(), kld.item()

# ──────────────────────────────────────────────────────────────────────────────
# 3) Positional/time embedding
# ──────────────────────────────────────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device   = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Diffusion U-Net in latent space
# ──────────────────────────────────────────────────────────────────────────────
class DiffUNet(nn.Module):
    def __init__(self, latent_dim, time_emb_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU(True),
        )
        ch = latent_dim
        self.net = nn.Sequential(
            nn.Linear(ch + time_emb_dim, ch*2), nn.ReLU(True),
            nn.Linear(ch*2, ch),        nn.ReLU(True),
            nn.Linear(ch, ch)
        )
    def forward(self, x, t):
        temb = self.time_mlp(t)
        h    = torch.cat([x, temb], dim=-1)
        return self.net(h)

class Diffusion:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        self.model     = model
        self.timesteps = timesteps
        dev = next(model.parameters()).device
        betas          = torch.linspace(beta_start, beta_end, timesteps, device=dev)
        alphas         = 1 - betas
        self.betas     = betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        a_bar = self.alpha_bar[t].unsqueeze(-1)
        return a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise, noise

    def p_losses(self, x0, t):
        x_noisy, noise = self.q_sample(x0, t)
        pred_noise     = self.model(x_noisy, t)
        return F.mse_loss(pred_noise, noise)

    def p_sample(self, x_t, t):
        beta_t      = self.betas[t].item()
        a_bar_t     = self.alpha_bar[t].item()
        a_bar_prev  = self.alpha_bar[t-1].item() if t > 0 else 1.0
        pred_noise  = self.model(x_t, torch.full((x_t.size(0),), t, dtype=torch.long, device=x_t.device))
        x0_pred     = (x_t - (1 - a_bar_t)**0.5 * pred_noise) / (a_bar_t**0.5)
        coef2       = a_bar_prev**0.5 * (1 - beta_t) / (1 - a_bar_t)
        mean        = a_bar_prev**0.5 * x0_pred + coef2 * x_t
        if t > 0:
            var   = beta_t * (1 - a_bar_prev) / (1 - a_bar_t)
            noise = torch.randn_like(x_t)
            return mean + var**0.5 * noise
        return mean

# ──────────────────────────────────────────────────────────────────────────────
# 5) Training loops
# ──────────────────────────────────────────────────────────────────────────────
def train_vae(weights, input_dim, latent_dim=512, epochs=100, batch_size=16, lr=1e-3, device='cuda'):
    vae = WeightVAE(input_dim, latent_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    ds  = WeightDataset(weights)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        total, rec_l, kld_l = 0, 0, 0
        for batch in dl:
            x      = batch.to(device)
            recon, mu, logvar   = vae(x)
            loss, rec_loss, kld_loss = vae.loss_function(recon, x, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); rec_l += rec_loss; kld_l += kld_loss
        print(f"VAE Epoch {ep+1}/{epochs} | total_loss={total/len(dl):.4f} rec={rec_l/len(dl):.4f} kld={kld_l/len(dl):.4f}")
    return vae


def train_diffusion(latents, model, diffusion, epochs=1000, batch_size=32, lr=1e-4, device='cuda'):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds  = WeightDataset(latents)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for ep in range(epochs):
        total_loss = 0
        for batch in dl:
            x  = batch.to(device)
            t  = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device)
            loss = diffusion.p_losses(x, t)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"Diff Epoch {ep+1}/{epochs} | loss={total_loss/len(dl):.4f}")
    return model

# ──────────────────────────────────────────────────────────────────────────────
# 6) Sampling, evaluation, and saving
# ──────────────────────────────────────────────────────────────────────────────
def sample_weights(diff_model, diffusion, vae, num_samples=1, device='cuda'):
    latents = torch.randn(num_samples, vae.mu_layer.out_features, device=device)
    for t in reversed(range(diffusion.timesteps)):
        latents = diffusion.p_sample(latents, t)
    with torch.no_grad():
        samples = vae.decoder(latents)
    return samples.cpu()

if __name__ == '__main__':
    ckpt_dir   = './Toy_CNN'
    latent_dim = 512
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) Load CNN weight data
    paths   = find_checkpoints(ckpt_dir)
    weights = torch.stack([load_weights(p) for p in paths])
    input_dim = weights.size(1)

    # 2) Train VAE
    vae     = train_vae(weights, input_dim, latent_dim, epochs=50, batch_size=4, device=device)

    # 2.1) Save trained VAE
    os.makedirs('models', exist_ok=True)
    vae_path = os.path.join('models', 'vae_final.pth')
    torch.save(vae.state_dict(), vae_path)
    print(f"Saved VAE weights to {vae_path}")

    # 3) Encode all weights to latent means
    vae.eval()
    with torch.no_grad():
        mus = []
        for w in weights.to(device):
            _, mu, _ = vae(w.unsqueeze(0))
            mus.append(mu.squeeze(0))
        latents = torch.stack(mus)

    # 4) Train diffusion model
    diff_unet = DiffUNet(latent_dim).to(device)
    diffusion = Diffusion(diff_unet)
    diff_unet = train_diffusion(latents, diff_unet, diffusion, epochs=2000, batch_size=4, device=device)

    # 5) Save trained diffusion U-Net weights
    unet_path = os.path.join('models', 'diffusion_unet_final.pth')
    torch.save(diff_unet.state_dict(), unet_path)
    print(f"Saved diffusion U-Net weights to {unet_path}")

    # 6) Save beta schedule
    sched_path = os.path.join('models', 'diffusion_schedule.pth')
    torch.save({'betas': diffusion.betas, 'alpha_bar': diffusion.alpha_bar}, sched_path)
    print(f"Saved diffusion schedule to {sched_path}")

    # 7) Sample and evaluate models as before
    sampled_weights = sample_weights(diff_unet, diffusion, vae, num_samples=10, device=device)
    # ... load sampled_weights into your TinyCNN for evaluation
