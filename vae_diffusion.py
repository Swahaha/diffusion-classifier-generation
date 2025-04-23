# The purpose of this code is to train a latent diffusion model on a dataset of TinyCNN checkpoints and previously trained VAE latent vectors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
from tqdm import tqdm
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

from diffusion_model import TinyCNN
from vae_model import WeightVAE, weights_to_model

# ---------------------------------Latent Diffusion Model ---------------------------------
class LatentDiffusion(nn.Module):
    def __init__(self, latent_dim, time_emb_dim=64, hidden_dims=[128, 128]):
        super().__init__()
        
        # Time embedding network
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Noise prediction network
        self.net = nn.ModuleList([])
        in_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            self.net.append(DiffusionBlock(
                in_dim, hidden_dim, time_emb_dim
            ))
            in_dim = hidden_dim
            
        self.final = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x, t):
        # Time embeddings
        t_emb = self.time_mlp(t)
        
        # Predict noise through the network
        h = x
        for block in self.net:
            h = block(h, t_emb)
            
        return self.final(h)
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.GELU()
        )
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.GELU()
        )
        
    def forward(self, x, t):
        h = self.net(x)
        time_emb = self.time_mlp(t)
        return h + time_emb

class LatentDiffusionTrainer:
    def __init__(self, vae, diffusion_model, timesteps=1000, beta_schedule='cosine'):
        self.vae = vae
        self.model = diffusion_model
        self.timesteps = timesteps
        self.beta_schedule_type = beta_schedule
        
        # Set up diffusion parameters
        if beta_schedule == 'linear':
            self.betas = self._linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        # Pre-compute diffusion parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Useful constants for diffusion process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.02)
    
    def _linear_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
        
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start, t, noise)
        
        predicted_noise = self.model(x_noisy, t)
        
        loss = F.mse_loss(noise, predicted_noise)
        
        return loss

    def train(self, latent_vectors, batch_size=32, epochs=100, lr=1e-4, device="cuda", 
             lr_decay_epochs=200, lr_decay_factor=0.5):
        # Create data loader from latent vectors
        dataset = torch.utils.data.TensorDataset(latent_vectors)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                             step_size=lr_decay_epochs, 
                                             gamma=lr_decay_factor)
        
        # Create log directory
        log_dir = f"latent_diffusion_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        
        # Training loop
        best_loss = float('inf')
        loss_history = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                # Get latent vectors
                z = batch[0].to(device)
                
                # Sample random timesteps
                t = torch.randint(0, self.timesteps, (z.shape[0],), device=device).long()
                
                # Calculate loss
                loss = self.p_losses(z, t)
                
                # Backprop
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Update learning rate scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Calculate average loss
            avg_loss = total_loss / len(dataloader)
            loss_history.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
            
            # Save if best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = os.path.join(log_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'timesteps': self.timesteps,
                    'beta_schedule': self.beta_schedule_type,
                    'latent_dim': self.model.final.out_features 
                }, checkpoint_path)
                print(f"New best model saved to {checkpoint_path}")
                
            # Save intermediate checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                checkpoint_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'timesteps': self.timesteps,
                    'beta_schedule': self.beta_schedule_type,
                    'latent_dim': self.model.final.out_features
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch+1}")
                
        # Plot loss history
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(os.path.join(log_dir, 'training_loss.png'))
        plt.close()
        
        print(f"Training complete. Best loss: {best_loss:.6f}")
        return log_dir
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
    
    @torch.no_grad()
    def p_sample_loop(self, shape, device):
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step'):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i)
            
        return img

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def process_checkpoints_with_vae(vae_path, checkpoint_dir, device):
    # Load the VAE
    vae_checkpoint = torch.load(vae_path, map_location=device)
    
    weight_dim = vae_checkpoint['weight_dim']
    latent_dim = vae_checkpoint['latent_dim']
    hidden_dim = vae_checkpoint['hidden_dim']
    
    vae = WeightVAE(
        weight_dim=weight_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    
    # Find checkpoint files
    checkpoint_paths = []
    for root, _, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith('.pth'):
                checkpoint_paths.append(os.path.join(root, file))
                
    print(f"Found {len(checkpoint_paths)} checkpoint files")
    
    # Process each checkpoint through the VAE to get latent vectors
    latent_vectors = []
    
    for path in tqdm(checkpoint_paths, desc="Processing checkpoints"):
        # Load the checkpoint
        ckpt = torch.load(path, map_location='cpu')
        
        # Flatten the weights
        params = []
        for name, param in ckpt['state_dict'].items():
            params.append(param.flatten())
        weight_vector = torch.cat(params)
        
        # Encode through the VAE to get latent vector (without sampling)
        with torch.no_grad():
            weight_vector = weight_vector.to(device)
            mu, _ = vae.encode(weight_vector)
            # Use the mean vector directly (no sampling)
            latent_vectors.append(mu)
    
    # Stack all latent vectors into a single tensor
    latent_vectors = torch.stack(latent_vectors)
    
    print(f"Created {latent_vectors.shape[0]} latent vectors of dimension {latent_vectors.shape[1]}")
    # Return just the latent vectors, not the VAE
    return latent_vectors

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the pretrained VAE
    print(f"Loading VAE from {args.vae_checkpoint}")
    vae_checkpoint = torch.load(args.vae_checkpoint, map_location=device)
    
    # Get dimensions from the checkpoint
    latent_dim = vae_checkpoint.get('latent_dim', 32)
    hidden_dim = vae_checkpoint.get('hidden_dim', 128)
    weight_dim = vae_checkpoint.get('weight_dim')
    
    # Initialize VAE
    vae = WeightVAE(
        weight_dim=weight_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Load the VAE weights
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    
    # Process checkpoint directory with VAE to get latent vectors
    print(f"Processing checkpoints in {args.checkpoint_dir}")
    latent_vectors = process_checkpoints_with_vae(args.vae_checkpoint, args.checkpoint_dir, device)
    print(f"Generated {latent_vectors.shape[0]} latent vectors with dimension {latent_vectors.shape[1]}")
    
    # Initialize the latent diffusion model
    hidden_dims = [256, 256, 256]
    diffusion_model = LatentDiffusion(
        latent_dim=latent_dim,
        time_emb_dim=128,
        hidden_dims=hidden_dims
    ).to(device)
    
    # Train the diffusion model
    print("Training latent diffusion model")
    trainer = LatentDiffusionTrainer(
        vae=vae,
        diffusion_model=diffusion_model,
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule
    )
    
    log_dir = trainer.train(
        latent_vectors=latent_vectors,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=device,
        lr_decay_epochs=args.lr_decay_epochs,
        lr_decay_factor=args.lr_decay_factor
    )
    
    print(f"Training complete. Model saved in {log_dir}")
    return log_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a latent diffusion model on VAE latent space")
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                        help='Path to VAE checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='Toy_CNN',
                        help='Directory containing model checkpoints')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=500,
                        help='Number of diffusion timesteps')
    parser.add_argument('--beta_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='Beta schedule for diffusion process')
    parser.add_argument('--lr_decay_epochs', type=int, default=200,
                        help='Number of epochs after which to decay learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5,
                        help='Factor by which to decay learning rate')
    
    args = parser.parse_args()
    main(args) 