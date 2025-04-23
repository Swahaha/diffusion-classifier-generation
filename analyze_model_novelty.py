# The purpose of this code is to analyze the novelty of a generated model compared to a training dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import random

from diffusion_model import TinyCNN
from vae_model import WeightVAE, weights_to_model
from vae_diffusion import LatentDiffusion, LatentDiffusionTrainer

def load_cifar10(batch_size=128, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=train,
                             download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                          shuffle=False, num_workers=2)
    return dataloader, dataset.classes

def extract_model_features(model, dataloader, device):
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Extracting features"):
            images = data[0].to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
    
    return np.concatenate(all_features, axis=0)

def compute_similarity_metrics(train_features, generated_features):
    metrics = {}
    
    # Cosine similarity between feature distributions
    train_mean = np.mean(train_features, axis=0)
    generated_mean = np.mean(generated_features, axis=0)
    metrics['cosine_similarity'] = cosine_similarity(
        train_mean.reshape(1, -1), 
        generated_mean.reshape(1, -1)
    )[0][0]
    
    # Wasserstein distance between feature distributions
    metrics['wasserstein_distance'] = wasserstein_distance(
        train_features.flatten(),
        generated_features.flatten()
    )
    
    # Feature diversity (variance ratio)
    train_var = np.var(train_features, axis=0)
    generated_var = np.var(generated_features, axis=0)
    metrics['variance_ratio'] = np.mean(generated_var) / np.mean(train_var)
    
    return metrics

def load_model_safely(checkpoint_path, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        if any(k.endswith(('.weight', '.bias')) for k in checkpoint.keys()):
            return checkpoint
    return checkpoint

def get_epoch_from_checkpoint(checkpoint_path):
    try:
        epoch = int(''.join(filter(str.isdigit, checkpoint_path.split('epoch')[-1])))
        return epoch
    except:
        return checkpoint_path

def analyze_generated_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load VAE and diffusion models
    vae_checkpoint = torch.load(args.vae_checkpoint, map_location=device)
    diffusion_checkpoint = torch.load(args.diffusion_checkpoint, map_location=device)
    
    # Initialize models
    vae = WeightVAE(
        weight_dim=vae_checkpoint['weight_dim'],
        latent_dim=vae_checkpoint['latent_dim'],
        hidden_dim=vae_checkpoint['hidden_dim']
    ).to(device)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    
    diffusion_model = LatentDiffusion(
        latent_dim=vae_checkpoint['latent_dim'],
        time_emb_dim=128,
        hidden_dims=[256, 256, 256]
    ).to(device)
    diffusion_model.load_state_dict(diffusion_checkpoint['model_state_dict'])
    diffusion_model.eval()
    
    trainer = LatentDiffusionTrainer(
        vae=vae,
        diffusion_model=diffusion_model,
        timesteps=diffusion_checkpoint['timesteps'],
        beta_schedule=diffusion_checkpoint['beta_schedule']
    )
    
    # Load training data
    print("Loading training data...")
    trainloader, _ = load_cifar10(batch_size=args.batch_size, train=True)
    
    # Find all reference model checkpoints and sort them
    if os.path.isdir(args.reference_model):
        all_checkpoints = glob.glob(os.path.join(args.reference_model, "*.pth"))
        all_checkpoints.sort(key=get_epoch_from_checkpoint)
        print(f"Found {len(all_checkpoints)} total checkpoints")
    else:
        raise ValueError("Please provide a directory containing model checkpoints")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(args.diffusion_checkpoint), "novelty_analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 10 samples and analyze each against 10 different checkpoints
    all_results = []
    
    for sample_idx in range(10):
        print(f"\nGenerating and analyzing sample {sample_idx + 1}/10")
        
        # Generate a model
        with torch.no_grad():
            latent_sample = trainer.p_sample_loop(
                shape=(1, vae_checkpoint['latent_dim']),
                device=device
            )
            weight_vector = vae.decode(latent_sample).squeeze(0)
        
        generated_model = weights_to_model(weight_vector.cpu(), TinyCNN)
        generated_model = generated_model.to(device)
        
        # Extract features from generated model
        print("Extracting features from generated model...")
        generated_features = extract_model_features(generated_model, trainloader, device)
        
        # Randomly select 10 checkpoints from dataset
        num_checkpoints = len(all_checkpoints)
        checkpoint_indices = np.linspace(0, num_checkpoints-1, 10, dtype=int)
        selected_checkpoints = [all_checkpoints[i] for i in checkpoint_indices]
        
        sample_results = []
        
        # Compare against each selected checkpoint
        for checkpoint_idx, checkpoint_path in enumerate(selected_checkpoints):
            print(f"Comparing with checkpoint {checkpoint_idx + 1}/10: {os.path.basename(checkpoint_path)}")
            
            # Load reference model
            reference_model = TinyCNN().to(device)
            state_dict = load_model_safely(checkpoint_path, device)
            reference_model.load_state_dict(state_dict)
            reference_model.eval()
            
            # Extract features from reference model
            train_features = extract_model_features(reference_model, trainloader, device)
            
            # Compute metrics
            metrics = compute_similarity_metrics(train_features, generated_features)
            metrics['checkpoint'] = os.path.basename(checkpoint_path)
            metrics['sample_id'] = sample_idx
            metrics['checkpoint_idx'] = checkpoint_idx
            sample_results.append(metrics)
        
        all_results.extend(sample_results)
        
        # Plot results for this sample
        sample_dir = os.path.join(output_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Plot metrics across checkpoints
        for metric in ['cosine_similarity', 'wasserstein_distance', 'variance_ratio']:
            values = [r[metric] for r in sample_results]
            checkpoint_names = [r['checkpoint'] for r in sample_results]
            
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(values)), values, 'o-')
            plt.xticks(range(len(values)), checkpoint_names, rotation=45)
            plt.title(f"{metric} across checkpoints for sample {sample_idx}")
            plt.ylabel(metric)
            plt.xlabel("Checkpoint")
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, f"{metric}_progression.png"))
            plt.close()
    
    # Save all results to CSV
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)
    
    # Create summary visualizations
    plt.figure(figsize=(15, 5))
    
    # Plot distribution of metrics across all samples and checkpoints
    for i, metric in enumerate(['cosine_similarity', 'wasserstein_distance', 'variance_ratio']):
        plt.subplot(1, 3, i+1)
        sns.boxplot(data=df, x='sample_id', y=metric)
        plt.title(f"{metric} distribution")
        plt.xlabel("Sample ID")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_distributions.png"))
    plt.close()
    
    # Save summary statistics
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("=== Novelty Analysis Summary ===\n\n")
        
        for metric in ['cosine_similarity', 'wasserstein_distance', 'variance_ratio']:
            f.write(f"\n{metric}:\n")
            f.write(f"  Overall Mean: {df[metric].mean():.4f}\n")
            f.write(f"  Overall Std: {df[metric].std():.4f}\n")
            f.write(f"  Overall Min: {df[metric].min():.4f}\n")
            f.write(f"  Overall Max: {df[metric].max():.4f}\n")
            
            # Per-sample statistics
            f.write("\n  Per-sample statistics:\n")
            for sample_id in range(10):
                sample_data = df[df['sample_id'] == sample_id][metric]
                f.write(f"    Sample {sample_id}:\n")
                f.write(f"      Mean: {sample_data.mean():.4f}\n")
                f.write(f"      Std: {sample_data.std():.4f}\n")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze novelty of generated models compared to training data")
    parser.add_argument("--vae-checkpoint", type=str, required=True, help="Path to the VAE checkpoint")
    parser.add_argument("--diffusion-checkpoint", type=str, required=True, help="Path to the diffusion model checkpoint")
    parser.add_argument("--reference-model", type=str, required=True, help="Path to directory containing reference model checkpoints")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for feature extraction")
    
    args = parser.parse_args()
    analyze_generated_models(args) 