import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets.cifar10 import get_cifar10_loaders, get_num_classes
from models.gan import Generator, Discriminator
from utils import save_checkpoint, save_samples, setup_device

def train_gan():
    # Hyperparameters
    batch_size = 128
    latent_dim = 100
    lr = 0.0002
    num_epochs = 100
    device = setup_device()
    
    # Setup models
    num_classes = get_num_classes()
    generator = Generator(latent_dim=latent_dim, num_classes=num_classes).to(device)
    discriminator = Discriminator(num_classes=num_classes).to(device)
    
    # Setup optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()
    
    # Get data loaders
    train_loader, _ = get_cifar10_loaders(batch_size=batch_size)
    
    # Training loop
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for i, (imgs, labels) in enumerate(progress_bar):
            batch_size = imgs.shape[0]
            
            # Adversarial ground truths
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)
            
            # Configure input
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Sample noise and labels
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            
            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)
            
            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(gen_imgs, gen_labels)
            g_loss = 0.5 * (adversarial_loss(validity, valid) + 
                          auxiliary_loss(pred_label, gen_labels))
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs, labels)
            d_real_loss = (adversarial_loss(real_pred, valid) + 
                         auxiliary_loss(real_aux, labels)) / 2
            
            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = (adversarial_loss(fake_pred, fake) + 
                         auxiliary_loss(fake_aux, gen_labels)) / 2
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': d_loss.item(),
                'G_loss': g_loss.item()
            })
        
        # Save generated samples
        if (epoch + 1) % 5 == 0:
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