# The purpose of this code is to compare the predictions of two TinyCNN models, 
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

def get_one_image_per_class(dataset):
    class_images = {}
    class_indices = {}
    
    for idx, (image, label) in enumerate(dataset):
        if label not in class_images and len(class_images) < 10:
            class_images[label] = image
            class_indices[label] = idx
        if len(class_images) == 10:
            break
    
    # Sort by class
    sorted_images = [class_images[i] for i in range(10)]
    sorted_indices = [class_indices[i] for i in range(10)]
    return sorted_images, sorted_indices

def denormalize_image(image):
    image = image.clone()
    image = image * torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    image = image + torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    return image.permute(1, 2, 0).numpy()

def get_prediction_text(probs, classes, label):
    pred_class = torch.argmax(probs).item()
    confidence = probs[pred_class].item() * 100
    text = f"{classes[pred_class]}\n{confidence:.1f}%"
    color = 'green' if pred_class == label else 'red'
    return text, color

def compare_models(args):
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
    
    # Get one image per class
    images, indices = get_one_image_per_class(dataset)
        
        # Load model 1 (specified model)
    model1 = TinyCNN().to(device)
    model1.load_state_dict(load_model_safely(args.model1, device))
    model1.eval()
    
    # Load random model from the training set
    model2_path = get_random_checkpoint()
    print(f"Randomly selected checkpoint for model 2: {model2_path}")
    model2 = TinyCNN().to(device)
    model2.load_state_dict(load_model_safely(model2_path, device))
    model2.eval()
    
    # Create figure
    fig = plt.figure(figsize=(7, 10))
    
    gs = plt.GridSpec(5, 3, width_ratios=[1, 1, 1])
    gs.update(wspace=0.0, hspace=0.5) 
    
    # Process each image
    with torch.no_grad():
        for i in range(5):
            image = images[i+2]
            label = i+2 
            
            # Get predictions
            image_tensor = image.unsqueeze(0).to(device)
            probs1 = F.softmax(model1(image_tensor), dim=1)[0]
            probs2 = F.softmax(model2(image_tensor), dim=1)[0]
            
            # Model 1 predictions (left)
            ax_left = plt.subplot(gs[i, 0])
            text1, color1 = get_prediction_text(probs1, classes, label)
            
            ax_left.text(1.05, 0.5, text1,
                        horizontalalignment='right',
                        verticalalignment='center',
                        color=color1,
                        fontsize=10,
                        transform=ax_left.transAxes)
            ax_left.axis('off')
            
            # Image (center)
            ax_center = plt.subplot(gs[i, 1])
            ax_center.imshow(denormalize_image(image))
            ax_center.axis('off')
            ax_center.set_title(f"True Class: {classes[label]}")
            
            # Model 2 predictions (right)
            ax_right = plt.subplot(gs[i, 2])
            text2, color2 = get_prediction_text(probs2, classes, label)
            ax_right.text(-0.05, 0.5, text2,
                         horizontalalignment='left',
                         verticalalignment='center',
                         color=color2,
                         fontsize=10,
                         transform=ax_right.transAxes)
            ax_right.axis('off')
    
    # Add model names as column headers
    fig.text(0.40, 0.87, "Generated Model", 
             ha='right', va='bottom')
    fig.text(0.60, 0.87, f"Sample from Training Set", 
             ha='left', va='bottom')
    
    # Save the figure
    output_path = "model_comparison.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Comparison saved to {output_path}")
    
    # Print accuracy summary
    print("\nAccuracy Summary:")
    correct1 = correct2 = 0
    with torch.no_grad():
        for image, label in zip(images, range(10)):
            image_tensor = image.unsqueeze(0).to(device)
            pred1 = torch.argmax(model1(image_tensor)).item()
            pred2 = torch.argmax(model2(image_tensor)).item()
            
            correct1 += (pred1 == label)
            correct2 += (pred2 == label)
    
    print(f"Model 1 accuracy: {correct1/10*100:.1f}%")
    print(f"Model 2 (random) accuracy: {correct2/10*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare predictions of two TinyCNN models")
    parser.add_argument("--model1", type=str, required=True, help="Path to first model checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="CNN_checkpoints/run_1", 
                      help="Directory containing checkpoints for random selection (default: CNN_checkpoints/run_1)")
    
    args = parser.parse_args()
    compare_models(args) 