import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime

class WeightDataset(Dataset):
    def __init__(self, checkpoint_dir):
        self.checkpoint_paths = []
        print(f"\nScanning directory: {checkpoint_dir}")
        for root, _, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith('.pth'):
                    self.checkpoint_paths.append(os.path.join(root, file))
        print(f"Found {len(self.checkpoint_paths)} checkpoint files:")
        for path in self.checkpoint_paths[:5]:  # Print first 5 paths
            print(f"  - {path}")
        if len(self.checkpoint_paths) > 5:
            print(f"  ... and {len(self.checkpoint_paths)-5} more")
                    
    def __len__(self):
        return len(self.checkpoint_paths)
    
    def __getitem__(self, idx):
        ckpt = torch.load(self.checkpoint_paths[idx], map_location='cpu')
        # Flatten all parameters into a single vector
        params = []
        for name, param in ckpt['state_dict'].items():
            params.append(param.flatten())
            if idx == 0:  # Print parameter shapes for first item only
                print(f"Parameter {name}: {param.shape}, {param.numel()} elements")
        flattened = torch.cat(params)
        if idx == 0:  # Print total parameters for first item only
            print(f"Total flattened parameters: {flattened.shape}")
        return flattened

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.02)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class DiffusionTrainer:
    def __init__(self, model, timesteps=1000, beta_schedule='cosine'):
        self.model = model
        self.timesteps = timesteps
        self.beta_schedule_type = beta_schedule
        print(f"\nInitializing DiffusionTrainer with {timesteps} timesteps and {beta_schedule} schedule")
        
        # Define beta schedule
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-calculate diffusion parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        print("Diffusion parameters initialized:")
        print(f"Beta range: [{self.betas[0]:.6f}, {self.betas[-1]:.6f}]")
        
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)

        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    def train(self, dataset, batch_size=32, epochs=100, lr=1e-4, weight_decay=1e-5, device="cuda", save_best_only=True):
        print(f"\nStarting training with:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {lr}")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - Device: {device}")
        print(f"  - Dataset size: {len(dataset)}")
        print(f"  - Save best only: {save_best_only}")
        print(f"  - Beta schedule: {self.beta_schedule_type}")
        
        # Create log directory
        log_dir = f"training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = open(os.path.join(log_dir, "training_log.txt"), "w")
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Improved optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs, 
            eta_min=lr/10
        )
        
        # Track best loss
        best_loss = float('inf')
        best_epoch = -1
        no_improvement_count = 0
        patience = 30 
        
        for epoch in range(epochs):
            epoch_start_time = datetime.now()
            total_loss = 0
            max_loss = 0
            min_loss = float('inf')
            loss_std = []
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Improved timestep sampling - concentrate more on early and late timesteps
                # This helps better learn the beginning and end of the diffusion process
                if np.random.random() < 0.2:
                    # Focus on early timesteps (t < 100)
                    t = torch.randint(0, min(100, self.timesteps), (batch.shape[0],), device=device).long()
                elif np.random.random() < 0.4:
                    # Focus on late timesteps (t > timesteps-100)
                    t = torch.randint(max(0, self.timesteps-100), self.timesteps, (batch.shape[0],), device=device).long()
                else:
                    # Uniform random otherwise
                    t = torch.randint(0, self.timesteps, (batch.shape[0],), device=device).long()
                
                # Calculate loss
                loss = self.p_losses(batch, t)
                total_loss += loss.item()
                max_loss = max(max_loss, loss.item())
                min_loss = min(min_loss, loss.item())
                loss_std.append(loss.item())
                
                # Backprop
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
            
            # Step the learning rate scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Calculate statistics
            avg_loss = total_loss / len(dataloader)
            loss_std = np.std(loss_std) if loss_std else 0
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            
            # Log detailed statistics
            log_msg = (f"\nEpoch {epoch+1}/{epochs} Statistics:\n"
                      f"  Average Loss: {avg_loss:.6f}\n"
                      f"  Min Loss: {min_loss:.6f}\n"
                      f"  Max Loss: {max_loss:.6f}\n"
                      f"  Loss Std: {loss_std:.6f}\n"
                      f"  Learning Rate: {current_lr:.6f}\n"
                      f"  Time: {epoch_time:.2f}s\n")
            
            print(log_msg)
            log_file.write(log_msg)
            log_file.flush()
            
            # Save checkpoint if best loss
            improved = avg_loss < best_loss
            if improved:
                best_loss = avg_loss
                best_epoch = epoch
                no_improvement_count = 0
                print(f"New best loss achieved! Saving checkpoint...")
                
                # Save best model checkpoint
                checkpoint_path = os.path.join(log_dir, f'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'timesteps': self.timesteps,
                    'beta_schedule': self.beta_schedule_type,
                }, checkpoint_path)
                
                log_file.write(f"New best model saved: {checkpoint_path}\n")
                log_file.flush()
            else:
                no_improvement_count += 1
                print(f"No improvement for {no_improvement_count} epochs. Best loss: {best_loss:.6f} (epoch {best_epoch+1})")
                
                # Early stopping
                if no_improvement_count >= patience:
                    print(f"Early stopping after {patience} epochs without improvement")
                    break
            
            # Regular checkpoint saving (only if not save_best_only)
            if not save_best_only and (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(log_dir, f'diffusion_checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'timesteps': self.timesteps,
                    'beta_schedule': self.beta_schedule_type,
                }, checkpoint_path)
                print(f"Regular checkpoint saved: {checkpoint_path}")
        
        log_file.close()
        print(f"\nTraining completed. Best loss: {best_loss:.6f} at epoch {best_epoch+1}")
        print(f"Logs and checkpoints saved in {log_dir}") 