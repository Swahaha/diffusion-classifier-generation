import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import glob
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from diffusion_model import TinyCNN
from vae_model import WeightVAE, weights_to_model
from vae_diffusion import LatentDiffusion, LatentDiffusionTrainer, extract

def load_cifar10(batch_size=128):
    """Load CIFAR10 test dataset with normalization"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testset = datasets.CIFAR10(root='./data', train=False,
                              download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    return testloader, testset.classes

def evaluate_model(model, testloader, device):
    """Evaluate model on test set and return predictions and ground truth"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data in tqdm(testloader, desc="Evaluating"):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    return accuracy, all_preds, all_labels, all_probs

def plot_results(all_preds, all_labels, class_names, output_dir):
    """Create visualizations of the model performance"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # Class distribution
    class_dist = np.bincount(all_preds, minlength=len(class_names))
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_dist)
    plt.title("Distribution of Predictions")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_distribution.png"))
    plt.close()
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Plot precision, recall, and F1 score
    metrics = ["precision", "recall", "f1-score"]
    for metric in metrics:
        values = [report[class_name][metric] for class_name in class_names]
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, values)
        plt.title(f"{metric.capitalize()} by Class")
        plt.xlabel("Class")
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_by_class.png"))
        plt.close()

def sample_and_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find the VAE checkpoint
    vae_checkpoint_path = args.vae_checkpoint
    print(f"Loading VAE from: {vae_checkpoint_path}")
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    
    # Create the VAE model
    weight_dim = vae_checkpoint['weight_dim']
    latent_dim = vae_checkpoint['latent_dim']
    hidden_dim = vae_checkpoint['hidden_dim']
    
    vae = WeightVAE(
        weight_dim=weight_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    
    # Find the diffusion model checkpoint
    diffusion_checkpoint_path = args.diffusion_checkpoint
    print(f"Loading diffusion model from: {diffusion_checkpoint_path}")
    diffusion_checkpoint = torch.load(diffusion_checkpoint_path, map_location=device)
    
    # Create the diffusion model
    diffusion_model = LatentDiffusion(
        latent_dim=latent_dim,
        time_emb_dim=128,
        hidden_dims=[256, 256, 256]
    ).to(device)
    
    diffusion_model.load_state_dict(diffusion_checkpoint['model_state_dict'])
    diffusion_model.eval()
    
    # Create the diffusion trainer
    trainer = LatentDiffusionTrainer(
        vae=vae,
        diffusion_model=diffusion_model,
        timesteps=diffusion_checkpoint['timesteps'],
        beta_schedule=diffusion_checkpoint['beta_schedule']
    )
    
    # Load CIFAR10 test dataset
    print("Loading CIFAR10 test dataset...")
    testloader, class_names = load_cifar10(batch_size=args.batch_size)
    
    # Generate and evaluate multiple models
    print(f"Generating and evaluating {args.num_samples} models...")
    
    best_accuracy = 0
    best_model = None
    best_preds = None
    best_labels = None
    best_probs = None
    best_sample_idx = -1
    
    all_accuracies = []
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(diffusion_checkpoint_path), "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(args.num_samples):
        print(f"\nSample {i+1}/{args.num_samples}")
        
        # Sample latent vectors from the diffusion model
        with torch.no_grad():
            # Generate latent vectors from diffusion model
            latent_sample = trainer.p_sample_loop(
                shape=(1, latent_dim),
                device=device
            )
            
            # Decode the latent vector to get weight vector
            weight_vector = vae.decode(latent_sample).squeeze(0)
        
        # Reconstruct the TinyCNN model
        model = weights_to_model(weight_vector.cpu(), TinyCNN)
        model = model.to(device)
        
        # Evaluate the model
        accuracy, preds, labels, probs = evaluate_model(model, testloader, device)
        all_accuracies.append(accuracy)
        
        print(f"Sample {i+1} Accuracy: {accuracy:.2f}%")
        
        # Save this model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_preds = preds
            best_labels = labels
            best_probs = probs
            best_sample_idx = i
            
            # Save the best model
            torch.save({
                'state_dict': model.state_dict(),
                'accuracy': accuracy,
                'sample_idx': i
            }, os.path.join(output_dir, "best_model.pth"))
            
            print(f"New best model! Accuracy: {best_accuracy:.2f}%")
    
    # Visualize the distribution of accuracies
    plt.figure(figsize=(10, 6))
    plt.hist(all_accuracies, bins=20, alpha=0.7)
    plt.axvline(best_accuracy, color='r', linestyle='--', label=f'Best Accuracy: {best_accuracy:.2f}%')
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Count")
    plt.title("Distribution of Model Accuracies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_distribution.png"))
    plt.close()
    
    # Print overall statistics
    print("\n=== Overall Statistics ===")
    print(f"Number of samples: {args.num_samples}")
    print(f"Average accuracy: {np.mean(all_accuracies):.2f}%")
    print(f"Median accuracy: {np.median(all_accuracies):.2f}%")
    print(f"Min accuracy: {np.min(all_accuracies):.2f}%")
    print(f"Max accuracy: {np.max(all_accuracies):.2f}%")
    print(f"Std dev: {np.std(all_accuracies):.2f}%")
    
    # Plot detailed results for the best model
    if best_model is not None:
        print(f"\n=== Best Model (Sample {best_sample_idx+1}) ===")
        print(f"Accuracy: {best_accuracy:.2f}%")
        
        # Create detailed plots for the best model
        plot_results(best_preds, best_labels, class_names, output_dir)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(best_labels, best_preds, target_names=class_names))
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VAE+Diffusion generated models")
    parser.add_argument("--vae-checkpoint", type=str, required=True, help="Path to the VAE checkpoint")
    parser.add_argument("--diffusion-checkpoint", type=str, required=True, help="Path to the diffusion model checkpoint")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for generation")
    parser.add_argument("--time-emb-dim", type=int, default=128, help="Time embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    
    args = parser.parse_args()
    sample_and_evaluate(args) 