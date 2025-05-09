# The purpose of this code is to analyze the predictions of two TinyCNN models, 
# one of them is a generated model and the other is a randomly selected checkpoint from a trained model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import os
import random
from diffusion_model import TinyCNN
import seaborn as sns
from scipy.stats import gaussian_kde

def load_model_safely(checkpoint_path, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        print("Weights-only loading failed, attempting legacy loading...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        if any(k.endswith(('.weight', '.bias')) for k in checkpoint.keys()):
            return checkpoint
    return checkpoint

def get_random_checkpoint(checkpoint_dir="CNN_checkpoints/run_1"):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    return random.choice(checkpoints)

def analyze_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=False,
                             download=True, transform=transform)
    classes = dataset.classes
    
    # Load the generated model
    model1 = TinyCNN().to(device)
    model1.load_state_dict(load_model_safely(args.model1, device))
    model1.eval()
    
    # Load random model from the training set
    model2_path = get_random_checkpoint()
    print(f"Randomly selected checkpoint for model 2: {model2_path}")
    model2 = TinyCNN().to(device)
    model2.load_state_dict(load_model_safely(model2_path, device))
    model2.eval()
    
    # Sample size for analysis
    sample_size = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # Initialize storage for predictions
    all_preds1 = []
    all_preds2 = []
    all_probs1 = []
    all_probs2 = []
    all_labels = []
    
    # Get predictions
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions from both models
            probs1 = F.softmax(model1(images), dim=1)
            probs2 = F.softmax(model2(images), dim=1)
            
            preds1 = torch.argmax(probs1, dim=1)
            preds2 = torch.argmax(probs2, dim=1)
            
            all_preds1.extend(preds1.cpu().numpy())
            all_preds2.extend(preds2.cpu().numpy())
            all_probs1.extend(probs1.cpu().numpy())
            all_probs2.extend(probs2.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds1 = np.array(all_preds1)
    all_preds2 = np.array(all_preds2)
    all_probs1 = np.array(all_probs1)
    all_probs2 = np.array(all_probs2)
    all_labels = np.array(all_labels)
    
    # Create output directory if it doesn't exist
    output_dir = "analysis_2_models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Class Distribution
    plt.figure(figsize=(12, 6))
    x = np.arange(10)
    width = 0.35
    
    plt.bar(x - width/2, np.bincount(all_preds1, minlength=10), width, 
            label=f"Generated Model", alpha=0.7)
    plt.bar(x + width/2, np.bincount(all_preds2, minlength=10), width, 
            label=f"Training Set", alpha=0.7)
    
    plt.title("Class Distribution Comparison")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save class distribution plot
    class_dist_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(class_dist_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Class distribution plot saved to {class_dist_path}")
    
    # Confidence Distribution 
    plt.figure(figsize=(12, 6))
    confidences1 = np.max(all_probs1, axis=1)
    confidences2 = np.max(all_probs2, axis=1)
    
    sns.histplot(confidences1, bins=20, kde=True, 
                label=f"Generated Model", 
                alpha=0.5, color='blue')
    sns.histplot(confidences2, bins=20, kde=True, 
                label=f"Training Set", 
                alpha=0.5, color='red')
    
    plt.title("Confidence Distribution Comparison")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    
    # Save confidence distribution plot
    conf_dist_path = os.path.join(output_dir, "confidence_distribution.png")
    plt.savefig(conf_dist_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Confidence distribution plot saved to {conf_dist_path}")
    
    # Class-wise Confidence
    plt.figure(figsize=(12, 6))
    x = np.arange(10)
    width = 0.35
    
    class_confidences1 = []
    class_confidences2 = []
    for i in range(10):
        mask1 = all_preds1 == i
        mask2 = all_preds2 == i
        class_confidences1.append(confidences1[mask1].mean() if np.any(mask1) else 0)
        class_confidences2.append(confidences2[mask2].mean() if np.any(mask2) else 0)
    
    plt.bar(x - width/2, class_confidences1, width, 
            label=f"Generated Model", alpha=0.7)
    plt.bar(x + width/2, class_confidences2, width, 
            label=f"Training Set", alpha=0.7)
    
    plt.title("Average Confidence by Class")
    plt.xlabel("Class")
    plt.ylabel("Average Confidence")
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save class-wise confidence plot
    class_conf_path = os.path.join(output_dir, "class_confidence.png")
    plt.savefig(class_conf_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Class-wise confidence plot saved to {class_conf_path}")
    
    # Print statistics to console
    print("\nModel Statistics:")
    print(f"Model 1 ({os.path.basename(args.model1)}):")
    print(f"Accuracy: {(all_preds1 == all_labels).mean()*100:.2f}%")
    print(f"Average Confidence: {confidences1.mean()*100:.2f}%")
    print(f"Confidence Std: {confidences1.std()*100:.2f}%")
    
    print(f"\nModel 2 ({os.path.basename(model2_path)}):")
    print(f"Accuracy: {(all_preds2 == all_labels).mean()*100:.2f}%")
    print(f"Average Confidence: {confidences2.mean()*100:.2f}%")
    print(f"Confidence Std: {confidences2.std()*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze predictions of two TinyCNN models")
    parser.add_argument("--model1", type=str, required=True, help="Path to first model checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="CNN_checkpoints/run_1", 
                      help="Directory containing checkpoints for random selection (default: CNN_checkpoints/run_1)")
    
    args = parser.parse_args()
    analyze_models(args) 