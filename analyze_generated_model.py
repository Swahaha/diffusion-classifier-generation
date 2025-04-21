import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from diffusion_model import TinyCNN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import argparse
import os
from sklearn.metrics import confusion_matrix, classification_report

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
    """Evaluate model on test set and return predictions and true labels"""
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
    
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    return accuracy, all_preds, all_labels, all_probs

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir='.'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_class_distribution(y_true, y_pred, class_names, output_dir='.'):
    """Plot distribution of true vs predicted classes"""
    true_counts = np.bincount(y_true, minlength=len(class_names))
    pred_counts = np.bincount(y_pred, minlength=len(class_names))
    
    df = pd.DataFrame({
        'True': true_counts,
        'Predicted': pred_counts
    }, index=class_names)
    
    plt.figure(figsize=(12, 6))
    df.plot(kind='bar')
    plt.title('Class Distribution (True vs Predicted)')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
    return df

def plot_prediction_confidence(all_probs, all_preds, all_labels, class_names, output_dir='.'):
    """Plot prediction confidence for correct and incorrect predictions"""
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Get confidence for each prediction
    confidences = [prob[pred] for prob, pred in zip(all_probs, all_preds)]
    
    # Separate correct and incorrect predictions
    correct_mask = all_preds == all_labels
    correct_conf = np.array(confidences)[correct_mask]
    incorrect_conf = np.array(confidences)[~correct_mask]
    
    plt.figure(figsize=(10, 6))
    plt.hist(correct_conf, alpha=0.5, bins=20, label='Correct Predictions')
    plt.hist(incorrect_conf, alpha=0.5, bins=20, label='Incorrect Predictions')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Prediction Confidence Distribution')
    plt.grid(linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    plt.close()
    
    # Average confidence per class
    class_conf = []
    for i in range(len(class_names)):
        class_mask = all_labels == i
        if np.sum(class_mask) > 0:
            avg_conf = np.mean([all_probs[j][all_preds[j]] for j in range(len(all_preds)) if class_mask[j]])
            class_conf.append(avg_conf)
        else:
            class_conf.append(0)
    
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_conf)
    plt.xlabel('Class')
    plt.ylabel('Average Confidence')
    plt.title('Average Prediction Confidence by Class')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'class_confidence.png'))
    plt.close()

def load_model_safely(model_path, device):
    """Load model with fallback options for PyTorch compatibility"""
    try:
        # First try with default settings
        checkpoint = torch.load(model_path, map_location=device)
        return checkpoint
    except Exception as e:
        print(f"Failed to load with default settings: {e}")
        try:
            # Try with weights_only=False for PyTorch 2.6+ compatibility
            print("Attempting to load with weights_only=False...")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            return checkpoint
        except Exception as e2:
            print(f"Failed to load with weights_only=False: {e2}")
            # Try the most permissive method
            print("Attempting to load with pickle module directly...")
            import pickle
            with open(model_path, 'rb') as f:
                checkpoint = pickle.load(f)
            return checkpoint

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if specified
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CIFAR10 test set
    print("Loading CIFAR10 test set...")
    testloader, class_names = load_cifar10(batch_size=args.batch_size)
    
    # Load the generated model
    print(f"Loading model from: {args.model_path}")
    checkpoint = load_model_safely(args.model_path, device)
    
    # Create and initialize the model
    model = TinyCNN().to(device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        acc_key = 'val_acc' if 'val_acc' in checkpoint else 'accuracy'
        prev_acc = checkpoint.get(acc_key, 'N/A')
    else:
        # If checkpoint is just the state dict
        model.load_state_dict(checkpoint)
        prev_acc = 'N/A'
    
    print(f"Evaluating model (previously reported accuracy: {prev_acc})")
    
    # Evaluate the model
    accuracy, all_preds, all_labels, all_probs = evaluate_model(model, testloader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Generate and save plots
    print("\nGenerating analysis plots...")
    plot_confusion_matrix(all_labels, all_preds, class_names, output_dir)
    class_dist_df = plot_class_distribution(all_labels, all_preds, class_names, output_dir)
    plot_prediction_confidence(all_probs, all_preds, all_labels, class_names, output_dir)
    
    # Check if model is biased towards specific classes
    pred_dist = np.bincount(all_preds, minlength=len(class_names))
    max_class = np.argmax(pred_dist)
    max_percent = 100 * pred_dist[max_class] / len(all_preds)
    
    print("\nClass Distribution Analysis:")
    print(class_dist_df)
    print(f"\nMost predicted class: {class_names[max_class]} ({max_percent:.2f}% of predictions)")
    
    # Check if there's significant bias (>50% predictions for a single class)
    if max_percent > 50:
        print(f"WARNING: Model shows significant bias toward class '{class_names[max_class]}'")
    
    print("\nAnalysis complete! Plots saved to:")
    print(f"- {os.path.join(output_dir, 'confusion_matrix.png')}")
    print(f"- {os.path.join(output_dir, 'class_distribution.png')}")
    print(f"- {os.path.join(output_dir, 'confidence_distribution.png')}")
    print(f"- {os.path.join(output_dir, 'class_confidence.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a generated model on CIFAR10")
    parser.add_argument('--model_path', type=str, default='generated_model.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    main(args) 