import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.gan import Generator
from utils import load_checkpoint
import os
from datetime import datetime

# Create timestamp for unique output directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join('analysis', f'gan_analysis_{timestamp}')
os.makedirs(output_dir, exist_ok=True)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load CIFAR-10 test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# CIFAR-10 class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Initialize generator
generator = Generator().to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Load checkpoint
checkpoint_path = 'checkpoints/gan/generator_100.pt'  # Update with your checkpoint path
checkpoint = torch.load(checkpoint_path)
generator.load_state_dict(checkpoint['model_state_dict'])
generator.eval()
print(f"Loaded generator from {checkpoint_path} (epoch {checkpoint['epoch']})")

def generate_samples(generator, num_samples=64, class_idx=None):
    """Generate samples from the generator"""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, 100, device=device)
        if class_idx is not None:
            labels = torch.full((num_samples,), class_idx, device=device)
        else:
            labels = torch.randint(0, 10, (num_samples,), device=device)
        samples = generator(noise, labels)
        samples = samples * 0.5 + 0.5  # Denormalize
    return samples.cpu(), labels.cpu()

def save_samples(samples, labels, title, filename, nrow=8):
    """Save a grid of samples to a file"""
    plt.figure(figsize=(15, 15))
    grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title(title)
    plt.axis('off')
    
    # Add class labels
    for i, label in enumerate(labels[:nrow]):
        plt.text(i * 32 + 16, 32 + 5, f'{classes[label.item()]}', 
                ha='center', va='top', color='white', fontsize=8)
    
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def save_comparison(generator, test_loader, num_samples=64):
    """Save comparison of real vs generated images"""
    # Get real images
    real_images, real_labels = next(iter(test_loader))
    real_images = real_images.to(device)
    real_labels = real_labels.to(device)
    
    # Generate fake images
    fake_images, fake_labels = generate_samples(generator, num_samples)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot real images
    real_grid = torchvision.utils.make_grid(real_images.cpu(), nrow=8, padding=2, normalize=True)
    ax1.imshow(real_grid.permute(1, 2, 0).numpy())
    ax1.set_title('Real Images')
    ax1.axis('off')
    
    # Plot generated images
    fake_grid = torchvision.utils.make_grid(fake_images, nrow=8, padding=2, normalize=False)
    ax2.imshow(fake_grid.permute(1, 2, 0).numpy())
    ax2.set_title('Generated Images')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'real_vs_generated.png'))
    plt.close()
    
    # Save class distributions to file
    with open(os.path.join(output_dir, 'class_distribution.txt'), 'w') as f:
        f.write("Real class distribution:\n")
        for i in range(10):
            count = (real_labels == i).sum().item()
            f.write(f"{classes[i]}: {count}\n")
        
        f.write("\nGenerated class distribution:\n")
        for i in range(10):
            count = (fake_labels == i).sum().item()
            f.write(f"{classes[i]}: {count}\n")

def analyze_training_progress(checkpoint_paths):
    """Analyze and save training progress from multiple checkpoints"""
    fig, axes = plt.subplots(len(checkpoint_paths), 2, figsize=(15, 5*len(checkpoint_paths)))
    
    for i, path in enumerate(checkpoint_paths):
        # Load checkpoint
        generator = Generator().to(device)
        checkpoint = torch.load(path)
        generator.load_state_dict(checkpoint['model_state_dict'])
        generator.eval()
        
        # Generate samples
        samples, labels = generate_samples(generator, num_samples=16)
        
        # Plot samples
        grid = torchvision.utils.make_grid(samples, nrow=4, padding=2, normalize=False)
        axes[i, 0].imshow(grid.permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f'Epoch {checkpoint["epoch"]}')
        axes[i, 0].axis('off')
        
        # Plot class distribution
        class_counts = torch.bincount(labels, minlength=10)
        axes[i, 1].bar(range(10), class_counts.numpy())
        axes[i, 1].set_title(f'Class Distribution (Epoch {checkpoint["epoch"]})')
        axes[i, 1].set_xticks(range(10))
        axes[i, 1].set_xticklabels(classes, rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_progress.png'))
    plt.close()

def main():
    print(f"Saving analysis results to: {output_dir}")
    
    # Generate and save samples for each class
    for class_idx in range(10):
        samples, labels = generate_samples(generator, num_samples=64, class_idx=class_idx)
        save_samples(samples, labels, f'Generated {classes[class_idx]}s', 
                    f'generated_{classes[class_idx]}.png')
        print(f"Saved generated samples for class: {classes[class_idx]}")
    
    # Save real vs generated comparison
    save_comparison(generator, test_loader)
    print("Saved real vs generated comparison")
    
    # Analyze training progress using available checkpoints
    checkpoint_paths = [
        'checkpoints/gan/generator_10.pt',
        'checkpoints/gan/generator_50.pt',
        'checkpoints/gan/generator_100.pt'
    ]
    analyze_training_progress(checkpoint_paths)
    print("Saved training progress analysis")

if __name__ == "__main__":
    main() 