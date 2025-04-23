# The purpose of this code is to train a VAE on a dataset of TinyCNN checkpoints

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
from tqdm import tqdm
import numpy as np
import argparse
from datetime import datetime

from diffusion_model import TinyCNN
from vae_model import WeightVAE, flatten_weights, weights_to_model

class WeightDataset(Dataset):
    def __init__(self, checkpoint_dir):
        self.checkpoint_paths = []
        print(f"\nScanning directory: {checkpoint_dir}")
        for root, _, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith('.pth'):
                    self.checkpoint_paths.append(os.path.join(root, file))
        print(f"Found {len(self.checkpoint_paths)} checkpoint files")
        
        # Load the first checkpoint to get weight dimension
        if len(self.checkpoint_paths) > 0:
            ckpt = torch.load(self.checkpoint_paths[0], map_location='cpu')
            self.weight_vectors = torch.zeros(len(self.checkpoint_paths), self._get_weight_dim(ckpt))
            
            # Load all weights
            for i, path in enumerate(tqdm(self.checkpoint_paths, desc="Loading weights")):
                ckpt = torch.load(path, map_location='cpu')
                self.weight_vectors[i] = self._flatten_weights(ckpt)
        else:
            raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
    def _get_weight_dim(self, ckpt):
        # Calculate total weight dimension
        total_dim = 0
        for name, param in ckpt['state_dict'].items():
            total_dim += param.numel()
        return total_dim
    
    def _flatten_weights(self, ckpt):
        # Flatten all parameters into a single vector
        params = []
        for name, param in ckpt['state_dict'].items():
            params.append(param.flatten())
        return torch.cat(params)
    
    def __len__(self):
        return len(self.checkpoint_paths)
    
    def __getitem__(self, idx):
        return self.weight_vectors[idx]

def train_vae(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Calculate weight dimension
    dummy_model = TinyCNN()
    weight_dim = sum(p.numel() for p in dummy_model.parameters())
    print(f"TinyCNN parameters: {weight_dim}")
    
    # Create dataset
    dataset = WeightDataset(args.checkpoint_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize VAE
    vae = WeightVAE(
        weight_dim=weight_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epochs, gamma=args.lr_decay_factor)
    
    # Create log directory
    log_dir = f"vae_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Starts training
        vae.train()
        train_loss = 0
        recon_loss_total = 0
        kld_loss_total = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, log_var = vae(batch)
            
            # Calculate loss with KL weight annealing
            kld_weight = min(epoch / (args.epochs * 0.2), 1.0) * args.kl_weight
            loss, recon_loss, kld_loss = vae.loss_function(recon_batch, batch, mu, log_var, kld_weight)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kld_loss_total += kld_loss.item()
        
        # Apply learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate average loss
        avg_loss = train_loss / len(dataloader.dataset)
        avg_recon_loss = recon_loss_total / len(dataloader.dataset)
        avg_kld_loss = kld_loss_total / len(dataloader.dataset)
        
        # Print training results
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}, Recon Loss: {avg_recon_loss:.6f}, KLD Loss: {avg_kld_loss:.6f}, LR: {current_lr:.2e}")
        
        # Save model if it's the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(log_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'latent_dim': args.latent_dim,
                'weight_dim': weight_dim,
                'hidden_dim': args.hidden_dim
            }, checkpoint_path)
            print(f"New best model saved to {checkpoint_path}")
        
        # Generate sample model every 10 epochs
        if (epoch + 1) % 10 == 0:
            vae.eval()
            with torch.no_grad():
                # Sample a weight vector
                sampled_weights = vae.sample(1, device=device).squeeze(0)
                
                # Reconstruct model
                generated_model = weights_to_model(sampled_weights.cpu(), TinyCNN)
                
                # Save the generated model
                generated_path = os.path.join(log_dir, f"generated_epoch_{epoch+1}.pth")
                torch.save({
                    'state_dict': generated_model.state_dict(),
                    'epoch': epoch,
                }, generated_path)
    
    print(f"Training complete. Final loss: {best_loss:.6f}")
    return log_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE for weight generation")
    parser.add_argument('--checkpoint_dir', type=str, default='Toy_CNN', 
                        help='Directory containing model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=32, 
                        help='Dimension of the latent space')
    parser.add_argument('--hidden_dim', type=int, default=128, 
                        help='Dimension of hidden layers')
    parser.add_argument('--kl_weight', type=float, default=0.01, 
                        help='Weight for KL divergence loss')
    parser.add_argument('--lr_decay_epochs', type=int, default=200, 
                        help='Number of epochs after which to decay learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, 
                        help='Factor by which to decay learning rate')
    
    args = parser.parse_args()
    train_vae(args) 