import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets.cifar10 import get_cifar10_loaders, get_num_classes
from models.diffusion import UNet
from utils import (save_checkpoint, save_samples, setup_device,
                  get_noise_schedule, get_noisy_image)

def train_diffusion():
    # Hyperparameters
    batch_size = 128
    num_timesteps = 1000
    lr = 0.0001
    num_epochs = 200
    device = setup_device()

    # Create directories if they don't exist
    os.makedirs('checkpoints/diffusion', exist_ok=True)
    os.makedirs('results/diffusion', exist_ok=True)
    
    # Setup model
    num_classes = get_num_classes()
    model = UNet(n_channels=3, n_classes=num_classes).to(device)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    start_epoch = 101
    checkpoint_path = 'checkpoints/diffusion/checkpoint_epoch_100.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Get noise schedule
    betas = get_noise_schedule(num_timesteps).to(device)
    
    # Get data loaders
    train_loader, _ = get_cifar10_loaders(batch_size=batch_size)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        
        for i, (imgs, labels) in enumerate(train_loader):
            # Move data to device
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Sample timestep
            t = torch.randint(0, num_timesteps, (imgs.shape[0],), device=device).float()
            
            # Get noisy image and target noise
            noisy_imgs, target_noise = get_noisy_image(imgs, t, betas)
            
            # Forward pass
            optimizer.zero_grad()
            predicted_noise = model(noisy_imgs, t, labels)
            
            # Compute loss
            loss = criterion(predicted_noise, target_noise)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Print single line per epoch
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {start_epoch+epoch+1}/{start_epoch+num_epochs}, Loss: {avg_loss:.4f}')
        
        # Save checkpoint every 5 epochs
        if (start_epoch + epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, start_epoch + epoch + 1, f'checkpoints/diffusion/checkpoint_epoch_{start_epoch+epoch+1}.pth')
        
        # Save generated samples
        if (start_epoch+epoch + 1) % 5 == 0:
            with torch.no_grad():
                # Generate samples for each class
                samples = []
                for label in range(num_classes):
                    # Start from pure noise
                    x = torch.randn(1, 3, 32, 32).to(device)
                    labels_tensor = torch.tensor([label], device=device)
                    
                    # Denoising loop
                    for t in reversed(range(num_timesteps)):
                        t_tensor = torch.tensor([t], device=device).float()
                        predicted_noise = model(x, t_tensor, labels_tensor)
                        alpha = 1 - betas[t]
                        alpha_bar = torch.prod(1 - betas[:t+1])
                        x = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
                        if t > 0:
                            x += torch.sqrt(betas[t]) * torch.randn_like(x)
                    
                    samples.append(x)
                
                # Stack and save samples
                samples = torch.cat(samples, dim=0)
                labels = torch.arange(num_classes).to(device)
                save_samples(samples, labels, start_epoch+epoch+1,
                           f'results/diffusion/samples_epoch_{start_epoch+epoch+1}.png')

def sample(model, num_samples=10, num_timesteps=1000):
    """Generate samples from trained model"""
    device = next(model.parameters()).device
    betas = get_noise_schedule(num_timesteps).to(device)
    
    samples = []
    for label in range(num_samples):
        # Start from pure noise
        x = torch.randn(1, 3, 32, 32).to(device)
        labels_tensor = torch.tensor([label], device=device)
        
        # Denoising loop
        for t in reversed(range(num_timesteps)):
            t_tensor = torch.tensor([t], device=device).float()
            predicted_noise = model(x, t_tensor, labels_tensor)
            alpha = 1 - betas[t]
            alpha_bar = torch.prod(1 - betas[:t+1])
            x = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
            if t > 0:
                x += torch.sqrt(betas[t]) * torch.randn_like(x)
        
        samples.append(x)
    
    return torch.cat(samples, dim=0)

if __name__ == '__main__':
    train_diffusion() 