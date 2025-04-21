import torch
import torch.nn.functional as F
from diffusion_model import WeightDiffusion, TinyCNN
import numpy as np

@torch.no_grad()
def sample_timestep(model, x, t, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas):
    """Sample from the model at a specific timestep"""
    # Predict noise
    predicted_noise = model(x, t)
    
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

def sample_weights(model, timesteps, weight_shape, device="cuda"):
    """Generate a new set of weights using the diffusion model"""
    # Start from random noise
    x = torch.randn(weight_shape).to(device)
    
    # Parameters for reverse process
    betas = torch.linspace(0.0001, 0.02, timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    # Iteratively denoise
    for t in reversed(range(timesteps)):
        t_tensor = torch.tensor([t], device=device)
        x = sample_timestep(
            model, x, t,
            betas,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas
        )
    
    return x

def reconstruct_model(weight_vector, model_class=TinyCNN):
    """Reconstruct a model from a flattened weight vector"""
    model = model_class()
    start_idx = 0
    
    # Dictionary to store reconstructed parameters
    state_dict = {}
    
    for name, param in model.state_dict().items():
        num_params = param.numel()
        param_slice = weight_vector[start_idx:start_idx + num_params]
        state_dict[name] = param_slice.reshape(param.shape)
        start_idx += num_params
    
    model.load_state_dict(state_dict)
    return model

def main():
    # Load the trained diffusion model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # First, calculate the dimension of the weight vector
    dummy_model = TinyCNN()
    weight_dim = sum(p.numel() for p in dummy_model.parameters())
    
    # Initialize and load the diffusion model
    diffusion_model = WeightDiffusion(weight_dim=weight_dim).to(device)
    checkpoint = torch.load('diffusion_checkpoint_epoch_100.pth')  # adjust epoch as needed
    diffusion_model.load_state_dict(checkpoint['model_state_dict'])
    diffusion_model.eval()
    
    # Sample new weights
    print("Sampling new weights...")
    sampled_weights = sample_weights(
        diffusion_model,
        timesteps=500,  # Match training timesteps
        weight_shape=(weight_dim,),
        device=device
    )
    
    # Reconstruct the model
    print("Reconstructing model...")
    generated_model = reconstruct_model(sampled_weights)
    
    # Save the generated model
    torch.save({
        'state_dict': generated_model.state_dict(),
        'val_acc': None  # This would need to be evaluated
    }, 'generated_model.pth')
    
    print("Generated model saved to 'generated_model.pth'")

if __name__ == "__main__":
    main() 