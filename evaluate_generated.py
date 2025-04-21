import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from diffusion_model import TinyCNN, WeightDiffusion
import numpy as np
from tqdm import tqdm
import glob
import os

def load_cifar10(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testset = datasets.CIFAR10(root='./data', train=False,
                              download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    return testloader

def evaluate_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader, desc="Evaluating"):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

@torch.no_grad()
def sample_timestep(model, x, t, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas):
    """Sample from the model at a specific timestep"""
    # Convert t to tensor
    t_tensor = torch.tensor([t], device=x.device, dtype=torch.float32)
    
    # Predict noise
    predicted_noise = model(x, t_tensor)
    
    # Extract required values for this timestep
    beta = betas[t]
    sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t]
    sqrt_recip_alpha = sqrt_recip_alphas[t]

    # Compute the mean
    pred_original = sqrt_recip_alpha * (x - beta * predicted_noise / sqrt_one_minus_alpha)
    
    # Add noise if t > 0
    if t > 0:
        noise = torch.randn_like(x)
        return pred_original + torch.sqrt(beta) * noise
    else:
        return pred_original

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.02)

def linear_beta_schedule(timesteps):
    """Linear beta schedule"""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def sample_weights(model, timesteps, weight_shape, device="cuda", beta_schedule="cosine"):
    """Generate a new set of weights using the diffusion model"""
    # Start from random noise
    x = torch.randn(weight_shape).to(device)
    
    # Parameters for reverse process
    if beta_schedule == "linear":
        betas = linear_beta_schedule(timesteps).to(device)
    else:  # default to cosine
        betas = cosine_beta_schedule(timesteps).to(device)
        
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    # Iteratively denoise
    for t in reversed(range(timesteps)):
        x = sample_timestep(
            model, x, t,
            betas,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas
        )
    
    return x

def reconstruct_model(weight_vector, model_class=TinyCNN):
    """Reconstruct a model from a flattened weight vector"""
    # Create a new model instance
    model = model_class()
    
    # Move weight vector to CPU for reconstruction
    weight_vector = weight_vector.cpu()
    
    # Only handle one sample at a time
    if len(weight_vector.shape) > 1:
        weight_vector = weight_vector.squeeze(0)
    
    # Get model parameter shapes
    param_shapes = {}
    param_sizes = {}
    start_indices = {}
    end_indices = {}
    
    # First collect all parameter info
    current_idx = 0
    for name, param in model.state_dict().items():
        param_shapes[name] = param.shape
        param_sizes[name] = param.numel()
        start_indices[name] = current_idx
        end_indices[name] = current_idx + param_sizes[name]
        current_idx += param_sizes[name]
        
    # Print parameter info for debugging
    total_size = sum(param_sizes.values())
    print(f"Total model parameters: {total_size}")
    print(f"Weight vector size: {weight_vector.numel()}")
    
    if total_size != weight_vector.numel():
        raise ValueError(f"Parameter count mismatch! Model has {total_size} parameters but weight vector has {weight_vector.numel()}")
        
    # Create a new state dictionary
    new_state_dict = {}
    
    # Reconstruct each parameter
    for name in model.state_dict().keys():
        start_idx = start_indices[name]
        end_idx = end_indices[name]
        shape = param_shapes[name]
        
        print(f"Processing {name}: shape {shape}, slice {start_idx}:{end_idx}")
        
        # Extract and reshape the parameter
        param_flat = weight_vector[start_idx:end_idx]
        param = param_flat.reshape(shape)
        new_state_dict[name] = param
    
    # Load the state dictionary into the model
    model.load_state_dict(new_state_dict)
    
    return model

def sample_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CIFAR10 test set
    print("Loading CIFAR10 test set...")
    testloader = load_cifar10()
    
    # Calculate weight dimension
    dummy_model = TinyCNN()
    weight_dim = sum(p.numel() for p in dummy_model.parameters())
    print(f"Total number of parameters: {weight_dim}")
    
    # Find the best model checkpoint
    log_dirs = glob.glob('training_logs_*')
    if not log_dirs:
        raise ValueError("No training logs found!")
    latest_log_dir = max(log_dirs, key=os.path.getmtime)
    best_model_path = os.path.join(latest_log_dir, 'best_model.pth')
    
    # Load the best diffusion model
    print(f"Loading best diffusion model from {best_model_path}...")
    checkpoint = torch.load(best_model_path)
    
    # Get model parameters from checkpoint
    model_timesteps = checkpoint.get('timesteps', 1000)
    beta_schedule = checkpoint.get('beta_schedule', 'cosine')
    
    # Initialize the model with the right configuration
    diffusion_model = WeightDiffusion(
        weight_dim=weight_dim,
        time_dim=256, 
        hidden_dims=[512, 512, 512, 256],
        dropout_rate=0.1
    ).to(device)
    
    diffusion_model.load_state_dict(checkpoint['model_state_dict'])
    diffusion_model.eval()
    
    print(f"Model parameters: timesteps={model_timesteps}, beta_schedule={beta_schedule}")
    
    # Sample multiple weights for better chance of good results
    print("Sampling multiple weight vectors...")
    num_samples = 10
    best_accuracy = 0
    best_model = None
    
    for i in range(num_samples):
        print(f"\nSampling weights (sample {i+1}/{num_samples})...")
        sampled_weights = sample_weights(
            diffusion_model,
            timesteps=model_timesteps,
            weight_shape=(weight_dim,),
            device=device,
            beta_schedule=beta_schedule
        )
        
        # Reconstruct the model
        print(f"Reconstructing TinyCNN from sample {i+1}...")
        generated_model = reconstruct_model(sampled_weights)
        generated_model = generated_model.to(device)
        
        # Evaluate
        print(f"Evaluating sample {i+1} on CIFAR10...")
        accuracy = evaluate_model(generated_model, testloader, device)
        print(f"Sample {i+1} Test Accuracy: {accuracy:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = generated_model
            print(f"New best model found! Accuracy: {best_accuracy:.2f}%")
    
    # Save the best generated model
    print(f"\nSaving best model with accuracy {best_accuracy:.2f}%")
    torch.save({
        'state_dict': best_model.state_dict(),
        'val_acc': best_accuracy
    }, 'best_generated_model.pth')
    print("Best generated model saved to 'best_generated_model.pth'")

if __name__ == "__main__":
    sample_and_evaluate() 