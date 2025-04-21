import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets.cifar10 import get_cifar10_loaders, get_num_classes
from models.gan import Generator, Discriminator
from utils import save_checkpoint, save_samples, setup_device

def compute_gradient_penalty(D, real_samples, fake_samples, labels, device):
    """Calculates the gradient penalty loss for WGAN-GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates, _ = D(interpolates, labels)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_gan():
    # Hyperparameters
    batch_size = 128
    latent_dim = 100
    lr = 0.0002
    num_epochs = 1000
    n_critic = 5  # Number of discriminator updates per generator update
    lambda_gp = 10  # Gradient penalty coefficient
    device = setup_device()
    
    # Setup models
    num_classes = get_num_classes()
    generator = Generator(latent_dim=latent_dim, num_classes=num_classes).to(device)
    discriminator = Discriminator(num_classes=num_classes).to(device)
    
    # Setup optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Loss functions
    auxiliary_loss = nn.CrossEntropyLoss()
    
    # Get data loaders
    train_loader, _ = get_cifar10_loaders(batch_size=batch_size)
    
    # Training loop
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for i, (imgs, labels) in enumerate(progress_bar):
            batch_size = imgs.shape[0]
            
            # Configure input
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Sample noise and labels
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            
            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)
            
            # Real images
            real_validity, real_aux = discriminator(real_imgs, labels)
            # Fake images
            fake_validity, fake_aux = discriminator(gen_imgs.detach(), gen_labels)
            
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_imgs.data, gen_imgs.data, labels, device
            )
            
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            # Auxiliary loss
            d_aux_loss = (auxiliary_loss(real_aux, labels) + auxiliary_loss(fake_aux, gen_labels)) / 2
            
            # Total discriminator loss
            d_loss = d_loss + d_aux_loss
            
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                
                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)
                
                # Loss measures generator's ability to fool the discriminator
                validity, pred_label = discriminator(gen_imgs, gen_labels)
                g_loss = -torch.mean(validity) + auxiliary_loss(pred_label, gen_labels)
                
                g_loss.backward()
                optimizer_G.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': d_loss.item(),
                'G_loss': g_loss.item() if i % n_critic == 0 else g_loss.item()
            })
        
        # Save generated samples
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                # Generate samples for each class
                z = torch.randn(num_classes, latent_dim).to(device)
                labels = torch.arange(num_classes).to(device)
                gen_imgs = generator(z, labels)
                save_samples(gen_imgs, labels, epoch+1, 
                           f'results/gan/samples_epoch_{epoch+1}.png')
            
            # Save model checkpoints
            save_checkpoint(generator, optimizer_G, epoch+1, 
                          f'checkpoints/gan/generator_{epoch+1}.pt')
            save_checkpoint(discriminator, optimizer_D, epoch+1, 
                          f'checkpoints/gan/discriminator_{epoch+1}.pt')

if __name__ == '__main__':
    train_gan() 