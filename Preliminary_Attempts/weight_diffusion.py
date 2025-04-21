import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Ensure only GPU 1 is visible
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class WeightUNet(nn.Module):
    def __init__(self, input_size, timesteps=1000):
        super().__init__()
        self.input_size  = input_size
        self.timesteps   = timesteps

        # figure out square grid multiple of 8
        raw       = int(np.ceil(np.sqrt(input_size)))
        self.grid = int(np.ceil(raw/8)*8)
        self.pad  = self.grid*self.grid - input_size

        # full-size UNet channels
        self.enc1 = self._block(2, 64)
        self.enc2 = self._block(64,128)
        self.enc3 = self._block(128,256)
        self.enc4 = self._block(256,512)
        self.pool = nn.MaxPool2d(2)

        self.up4  = nn.ConvTranspose2d(512,256,2,stride=2)
        self.dec4 = self._block(256+256,256)
        self.up3  = nn.ConvTranspose2d(256,128,2,stride=2)
        self.dec3 = self._block(128+128,128)
        self.up2  = nn.ConvTranspose2d(128, 64,2,stride=2)
        self.dec2 = self._block(64+ 64, 64)
        self.dec1 = nn.Conv2d(64,1,1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch,3,padding=1,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch,3,padding=1,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, t):
        dtype = next(self.enc1.parameters()).dtype
        x = x.to(dtype)

        B = x.size(0)
        if self.pad>0:
            x = F.pad(x,(0,self.pad))
        G = self.grid
        x = x.view(B,1,G,G)

        tt = (t.float()/self.timesteps).view(B,1,1,1).expand(B,1,G,G).to(dtype)
        x  = torch.cat([x,tt],dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d4 = self.dec4(torch.cat([self.up4(e4), e3],dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2],dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1],dim=1))
        d1 = self.dec1(d2)

        out = d1.view(B,-1)
        if self.pad>0:
            out = out[:,:self.input_size]
        return out

class WeightDiffusion:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        self.model     = model
        self.timesteps = timesteps
        dev = next(model.parameters()).device
        self.betas          = torch.linspace(beta_start, beta_end, timesteps, device=dev)
        self.alphas         = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t):
        noise = torch.randn_like(x_start)
        acp   = self.alphas_cumprod[t].view(-1,1)
        return acp.sqrt()*x_start + (1-acp).sqrt()*noise

    def p_sample(self, x_t, t):
        # accept t as int or tensor
        if isinstance(t, torch.Tensor):
            t_idx = int(t[0].item())
            t_tensor = t
        else:
            t_idx = t
            t_tensor = torch.full((x_t.size(0),), t_idx, dtype=torch.long, device=x_t.device)

        with torch.no_grad():
            eps        = self.model(x_t, t_tensor)
            beta_t     = self.betas[t_idx].view(-1,1)
            alpha_t    = self.alphas[t_idx].view(-1,1)
            alpha_bar  = self.alphas_cumprod[t_idx].view(-1,1)
            alpha_bar_prev = self.alphas_cumprod[t_idx-1].view(-1,1) if t_idx>0 else torch.ones_like(alpha_bar)

            x0_pred = (x_t - (1-alpha_bar).sqrt()*eps) / alpha_bar.sqrt()

            coef1 = beta_t * alpha_bar_prev.sqrt() / (1-alpha_bar)
            coef2 = alpha_t.sqrt() * (1-alpha_bar_prev) / (1-alpha_bar)
            mean  = coef1 * x0_pred + coef2 * x_t
            var   = beta_t * (1-alpha_bar_prev) / (1-alpha_bar)

            if t_idx>0:
                noise = torch.randn_like(x_t)
                return mean + var.sqrt()*noise
            return mean

    def train_step(self, x_start, optimizer):
        optimizer.zero_grad()
        t = torch.randint(0, self.timesteps, (x_start.shape[0],), device=x_start.device)
        x_noisy = self.q_sample(x_start, t)
        pred    = self.model(x_noisy, t)
        loss    = F.mse_loss(pred, x_start - x_noisy)
        loss.backward()
        optimizer.step()
        return loss.item()

def load_checkpoint_weights(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    weights = []
    for param in state_dict.values():
        weights.append(param.view(-1))
    return torch.cat(weights)

def train_diffusion_model(checkpoint_dir, num_epochs=100, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Recursively find all .pth files under checkpoint_dir
    checkpoint_paths = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for fname in files:
            if fname.endswith('.pth'):
                checkpoint_paths.append(os.path.join(root, fname))
    checkpoint_paths.sort()

    if not checkpoint_paths:
        raise ValueError(f"No .pth files found in {checkpoint_dir}")

    # Infer input size from first checkpoint
    sample_weights = load_checkpoint_weights(checkpoint_paths[0])
    input_size = sample_weights.shape[0]
    print(f"Found {len(checkpoint_paths)} checkpoints. Input size: {input_size}")

    # Prepare save directory
    save_dir = 'diffusion_checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # Initialize model, diffusion, optimizer
    model     = WeightUNet(input_size, timesteps=1000).to(device)
    diffusion = WeightDiffusion(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Stack all weights into a tensor
    weights = [load_checkpoint_weights(p) for p in checkpoint_paths]
    weights = torch.stack(weights).to(device)

    # Training loop
    for epoch in range(1, num_epochs+1):
        start = time.time()
        total_loss = 0.0
        steps = math.ceil(len(weights) / batch_size)

        pbar = tqdm(range(0, len(weights), batch_size),
                    desc=f"Epoch {epoch}/{num_epochs}")
        for i in pbar:
            batch = weights[i:i+batch_size]
            loss = diffusion.train_step(batch, optimizer)
            total_loss += loss
            pbar.set_postfix({'batch_loss': f"{loss:.4f}"})

        avg_loss = total_loss / steps
        elapsed  = time.time() - start
        lr       = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{num_epochs} | Avg Loss: {avg_loss:.4f} | "
              f"LR: {lr:.6f} | Time: {elapsed:.1f}s")

        # Save intermediate model
        save_path = os.path.join(save_dir, f"diffusion_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved diffusion model checkpoint to: {save_path}\n")

    return model

if __name__ == "__main__":
    checkpoint_dir = "./CNN_checkpoints"
    model = train_diffusion_model(checkpoint_dir)
    torch.save(model.state_dict(), "weight_diffusion_final.pth")
