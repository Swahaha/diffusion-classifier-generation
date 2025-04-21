import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import os
from datetime import datetime
from models.diffusion import UNet
from utils import get_noise_schedule
import pickle

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# CIFAR-10 class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Load CIFAR-10 training data
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data'].reshape(len(batch[b'data']), 3, 32, 32)
        labels = batch[b'labels']
        return data, labels

def load_cifar10_data(cifar_dir):
    images = []
    labels = []
    for i in range(1, 6):
        batch_path = os.path.join(cifar_dir, f"data_batch_{i}")
        batch_images, batch_labels = load_cifar_batch(batch_path)
        images.append(batch_images)
        labels.extend(batch_labels)
    return np.concatenate(images), np.array(labels)

cifar_dir = "data/cifar-10-batches-py"
real_images, real_labels = load_cifar10_data(cifar_dir)
print(f"Loaded real CIFAR-10: {real_images.shape[0]} images")

# Load diffusion model
model = UNet(n_channels=3, n_classes=10).to(device)
checkpoint_path = './diffusion_checkpoints/checkpoint_epoch_300.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded model from {checkpoint_path} (epoch {checkpoint['epoch']})")

# Sampling function
def sample_from_model(model, num_samples=100, class_idx=0, num_timesteps=1000):
    model.eval()
    betas = get_noise_schedule(num_timesteps).to(device)

    with torch.no_grad():
        x = torch.randn(num_samples, 3, 32, 32, device=device)
        labels = torch.full((num_samples,), class_idx, device=device)

        for t in reversed(range(num_timesteps)):
            t_tensor = torch.tensor([t], device=device).float().repeat(num_samples)
            predicted_noise = model(x, t_tensor, labels)
            alpha = 1 - betas[t]
            alpha_bar = torch.prod(1 - betas[:t+1])
            x = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
            if t > 0:
                x += torch.sqrt(betas[t]) * torch.randn_like(x)

        x = x * 0.5 + 0.5  # Denormalize to [0,1]
        x = torch.clamp(x, 0, 1)
    return x.cpu(), labels.cpu()

# Generate and collect synthetic data
generated_images = []
generated_labels = []

for class_idx in range(10):
    print(f"Generating 100 samples for class: {classes[class_idx]}")
    imgs, lbls = sample_from_model(model, num_samples=1250, class_idx=class_idx)
    imgs_np = (imgs.numpy() * 255).astype(np.uint8)  # Convert to [0,255]
    generated_images.append(imgs_np)
    generated_labels.extend([class_idx] * 100)

generated_images = np.concatenate(generated_images)
generated_labels = np.array(generated_labels)
print(f"Generated images: {generated_images.shape}")

# Convert real CIFAR-10 to match format [N, C, H, W]
real_images = real_images.astype(np.uint8)
print(f"Real images shape: {real_images.shape}")

# Combine real + generated data
all_images = np.concatenate([real_images, generated_images], axis=0)
all_labels = np.concatenate([real_labels, generated_labels], axis=0)
print(f"Total dataset: {all_images.shape[0]} images")

# Save combined dataset
output_file = f'combined_cifar10_with_diffusion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz'
np.savez(output_file, images=all_images, labels=all_labels)
print(f"Saved combined dataset to {output_file}")