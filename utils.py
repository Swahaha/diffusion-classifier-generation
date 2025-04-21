import os
import torch
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def save_samples(images, labels, epoch, path, nrow=8):
    """Save generated samples as a grid"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Denormalize images
    images = (images + 1) / 2
    
    # Create grid and save
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    plt.figure(figsize=(nrow, nrow))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(path)
    plt.close()

def setup_device():
    """Setup device (GPU if available)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def get_noise_schedule(num_timesteps=1000):
    """Get noise schedule for diffusion model"""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, num_timesteps)

def get_noisy_image(x_0, t, betas):
    """Add noise to image at timestep t"""
    noise = torch.randn_like(x_0)
    batch_size = x_0.shape[0]
    
    # Convert t to integer indices
    t_int = t.long()
    
    # Compute alpha_bar for each timestep in the batch
    alpha_bar = torch.ones(batch_size, device=x_0.device)
    for i in range(batch_size):
        alpha_bar[i] = torch.prod(1 - betas[:t_int[i]+1])
    
    # Reshape alpha_bar to match x_0 dimensions
    alpha_bar = alpha_bar.view(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
    return x_t, noise

def plot_results(generator, test_loader, device, epoch, save_dir='results'):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set generator to evaluation mode
    generator.eval()
    
    # Get a batch of real images
    real_images, real_labels = next(iter(test_loader))
    real_images = real_images.to(device)
    real_labels = real_labels.to(device)
    
    # Generate fake images
    noise = torch.randn(real_images.size(0), 100, device=device)
    fake_labels = torch.randint(0, 10, (real_images.size(0),), device=device)
    fake_images = generator(noise, fake_labels)
    
    # Denormalize images
    real_images = real_images * 0.5 + 0.5  # Scale from [-1, 1] to [0, 1]
    fake_images = fake_images * 0.5 + 0.5  # Scale from [-1, 1] to [0, 1]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot real images
    real_grid = make_grid(real_images.cpu(), nrow=8, padding=2, normalize=False)
    ax1.imshow(np.transpose(real_grid.numpy(), (1, 2, 0)))
    ax1.set_title('Real Images')
    ax1.axis('off')
    
    # Plot generated images
    fake_grid = make_grid(fake_images.cpu(), nrow=8, padding=2, normalize=False)
    ax2.imshow(np.transpose(fake_grid.numpy(), (1, 2, 0)))
    ax2.set_title('Generated Images')
    ax2.axis('off')
    
    # Add class labels below each image
    for i, (real_label, fake_label) in enumerate(zip(real_labels, fake_labels)):
        if i < 8:  # Only show labels for first row
            ax1.text(i * 32 + 16, 32 + 5, f'Class: {real_label.item()}', 
                    ha='center', va='top', color='white', fontsize=8)
            ax2.text(i * 32 + 16, 32 + 5, f'Class: {fake_label.item()}', 
                    ha='center', va='top', color='white', fontsize=8)
    
    plt.suptitle(f'Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}.png'))
    plt.close()
    
    # Set generator back to training mode
    generator.train() 