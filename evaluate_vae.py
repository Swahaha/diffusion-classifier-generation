# The purpose of this code is to evaluate the performance of a VAE on a dataset of TinyCNN checkpoints

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

def load_cifar10(batch_size=128):
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
    if args.vae_checkpoint:
        vae_checkpoint_path = args.vae_checkpoint
    else:
        vae_dirs = glob.glob('vae_logs_*')
        if not vae_dirs:
            raise ValueError("No VAE logs found! Please train the VAE first or specify a checkpoint path.")
        latest_vae_dir = max(vae_dirs, key=os.path.getmtime)
        vae_checkpoint_path = os.path.join(latest_vae_dir, 'best_model.pth')
    
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
    output_dir = os.path.join(os.path.dirname(vae_checkpoint_path), "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(args.num_samples):
        print(f"\nSample {i+1}/{args.num_samples}")
        
        # Sample a weight vector from the VAE
        with torch.no_grad():
            sampled_weights = vae.sample(1, device).squeeze(0)
        
        # Reconstruct the TinyCNN model
        model = weights_to_model(sampled_weights.cpu(), TinyCNN)
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
    parser = argparse.ArgumentParser(description="Evaluate models generated from a VAE")
    parser.add_argument('--vae_checkpoint', type=str, default='',
                        help='Path to the VAE checkpoint (default: use latest)')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to generate and evaluate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    sample_and_evaluate(args)