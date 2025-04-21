import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, embed_dim=100):
        super(Generator, self).__init__()
        
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        self.init_size = 4  # Initial size before upsampling
        self.l1 = nn.Linear(latent_dim + embed_dim, 128 * self.init_size ** 2)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Embed labels
        label_embedding = self.label_embedding(labels)
        
        # Concatenate noise and label embedding
        gen_input = torch.cat((noise, label_embedding), -1)
        
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes=10, embed_dim=100):
        super(Discriminator, self).__init__()
        
        # Image processing path
        self.conv_layers = nn.Sequential(
            # Input: 3xHxW (H,W can be 32 or 16)
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        )
        
        # Adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Calculate flattened size: 128 channels * 2 * 2 = 512
        self.flat_size = 128 * 2 * 2
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4)
        )
        
        # Label processing
        self.label_processor = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4)
        )
        
        # Combined processing
        self.combined_processor = nn.Sequential(
            nn.Linear(1024, 512),  # 512 + 512 = 1024
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4)
        )
        
        # Output heads
        self.adv_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.aux_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, img, labels):
        # Input validation
        assert img.dim() == 4, f"Expected 4D tensor (batch, channels, height, width), got {img.dim()}D"
        assert labels.dim() == 1, f"Expected 1D tensor (batch), got {labels.dim()}D"
        
        # Process image through conv layers
        features = self.conv_layers(img)
        
        # Use adaptive pooling to ensure consistent output size
        features = self.adaptive_pool(features)
        assert features.shape[1:] == (128, 2, 2), f"Expected features shape (batch, 128, 2, 2), got {features.shape}"
        
        # Flatten features
        features = features.view(features.size(0), -1)
        assert features.shape[1] == self.flat_size, f"Expected flattened size {self.flat_size}, got {features.shape[1]}"
        
        # Process features
        features = self.feature_processor(features)
        assert features.shape[1] == 512, f"Expected processed features size 512, got {features.shape[1]}"
        
        # Process labels
        label_embedding = self.label_embedding(labels)
        assert label_embedding.shape[1] == 100, f"Expected label embedding size 100, got {label_embedding.shape[1]}"
        
        label_features = self.label_processor(label_embedding)
        assert label_features.shape[1] == 512, f"Expected processed label features size 512, got {label_features.shape[1]}"
        
        # Combine features
        combined = torch.cat([features, label_features], dim=1)
        assert combined.shape[1] == 1024, f"Expected combined features size 1024, got {combined.shape[1]}"
        
        # Process combined features
        combined = self.combined_processor(combined)
        assert combined.shape[1] == 512, f"Expected processed combined features size 512, got {combined.shape[1]}"
        
        # Get outputs
        validity = self.adv_layer(combined)
        label = self.aux_layer(combined)
        
        assert validity.shape[1] == 1, f"Expected validity shape (batch, 1), got {validity.shape}"
        assert label.shape[1] == 10, f"Expected label shape (batch, 10), got {label.shape}"
        
        return validity, label 