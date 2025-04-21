import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_model import TinyCNN

class WeightVAE(nn.Module):
    def __init__(self, weight_dim, latent_dim=64, hidden_dim=256):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(weight_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Mean and variance for latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, weight_dim)
        )
        
        self.weight_dim = weight_dim
        self.latent_dim = latent_dim
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def sample(self, num_samples=1, device="cuda"):
        """Generate sample weight vectors from random latent vectors"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
    
    def loss_function(self, recon_x, x, mu, log_var, kld_weight=0.001):
        """VAE loss function: reconstruction + KL divergence losses"""
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + kld_weight * kld_loss, recon_loss, kld_loss
    
def flatten_weights(model):
    """Flatten model weights into a single vector"""
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)

def weights_to_model(weight_vector, model_class=TinyCNN):
    """Reconstruct a model from a flattened weight vector"""
    model = model_class()
    start_idx = 0
    
    for name, param in model.state_dict().items():
        num_params = param.numel()
        param_slice = weight_vector[start_idx:start_idx + num_params]
        model.state_dict()[name].copy_(param_slice.view(param.shape))
        start_idx += num_params
    
    return model