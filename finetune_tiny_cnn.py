#!/usr/bin/env python3
# finetune_tiny_cnn.py
# Script to finetune a TinyCNN checkpoint on CIFAR-10 for 5 epochs

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

# Define TinyCNN model (same as in the repository)
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

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

def finetune_model():
    # Configuration
    CHECKPOINT_PATH = "Current-best.pth"  # Path to the checkpoint
    OUTPUT_PATH = "finetuned_tiny_cnn.pth"
    NUM_EPOCHS = 5
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_set = CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    val_set = CIFAR10(root="./data", train=False, download=True, transform=transform_val)
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    model = TinyCNN().to(device)
    
    # Load checkpoint using the safe loader from analyze_generated_model.py
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = load_model_safely(CHECKPOINT_PATH, device)
        
        # Handle different checkpoint formats as done in analyze_generated_model.py
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            acc_key = 'val_acc' if 'val_acc' in checkpoint else 'accuracy'
            prev_acc = checkpoint.get(acc_key, 'N/A')
            print(f"Loaded model with accuracy: {prev_acc}")
        else:
            # If checkpoint is just the state dict
            model.load_state_dict(checkpoint)
            print("Loaded model (accuracy info not available)")
    else:
        print(f"Checkpoint {CHECKPOINT_PATH} not found. Starting with a fresh model.")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        progress_bar = tqdm(train_loader, desc="Training")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': train_loss / (progress_bar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': val_loss / (progress_bar.n + 1),
                    'acc': 100. * val_correct / val_total
                })
        
        # Print epoch summary
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        print(f"Training: Loss={train_loss/len(train_loader):.4f}, Accuracy={train_acc:.2f}%")
        print(f"Validation: Loss={val_loss/len(val_loader):.4f}, Accuracy={val_acc:.2f}%")
        
        # Save model if it has the best accuracy so far
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"New best accuracy: {best_acc:.2f}%. Saving model...")
            torch.save({
                'state_dict': model.state_dict(),
                'accuracy': val_acc,
                'epoch': epoch + 1
            }, OUTPUT_PATH)
    
    print(f"Finetuning completed. Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    finetune_model() 