import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*8*8,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout_rate=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch),
            nn.GELU()
        )
        
        # Improved network with layer normalization and residual connections
        self.norm1 = nn.LayerNorm(in_ch)
        self.net1 = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.norm2 = nn.LayerNorm(out_ch)
        self.net2 = nn.Sequential(
            nn.Linear(out_ch, out_ch),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Projection for residual connection if dimensions don't match
        self.proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        # First residual block
        residual = self.proj(x)
        x = self.norm1(x)
        x = self.net1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(t)
        x = x + time_emb
        
        # Second residual block
        x = self.norm2(x) 
        x = self.net2(x)
        
        # Add residual connection
        return x + residual

class WeightDiffusion(nn.Module):
    def __init__(
        self,
        weight_dim,
        time_dim=256,
        hidden_dims=[512, 512, 512, 256],
        dropout_rate=0.1
    ):
        super().__init__()
        
        # Time embedding with improved capacity
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(time_dim * 2, time_dim),
            nn.GELU(),
        )

        # Main network with residual connections and more capacity
        self.net = nn.ModuleList([])
        in_dim = weight_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.net.append(Block(in_dim, hidden_dim, time_dim))
            in_dim = hidden_dim
            
        # Improved final layers with normalization
        self.norm = nn.LayerNorm(hidden_dims[-1])
        self.final = nn.Sequential(
            nn.Linear(hidden_dims[-1], weight_dim),
            nn.Tanh()  # Constrain output range for stability
        )
        
        # Initialize weights for better convergence
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, time):
        t = self.time_mlp(time)
        
        h = x
        for block in self.net:
            h = block(h, t)
            
        h = self.norm(h)
        return self.final(h) 