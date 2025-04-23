#!/usr/bin/env python3
# Check the sparsity of a checkpoint with the same format

import torch
import numpy as np
import os
import argparse
from diffusion_model import TinyCNN  # Import TinyCNN from the repository

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

def analyze_checkpoint(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = load_model_safely(checkpoint_path, device)
    
    # Print checkpoint keys
    print("\nCheckpoint keys:")
    if isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            print(f"- {key}")
    else:
        print("Checkpoint is not a dictionary")
        
    # Check if state_dict is in the checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("\nState dict keys:")
        for key in state_dict.keys():
            print(f"- {key}")
    
    # Analyze sparsity
    print("\nAnalyzing sparsity:")
    
    # Try to load into TinyCNN model
    try:
        model = TinyCNN().to(device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            state_dict = checkpoint['state_dict']
        else:
            model.load_state_dict(checkpoint)
            state_dict = checkpoint
            
        total_params = 0
        zero_params = 0
        
        for name, param in state_dict.items():
            param_array = param.cpu().numpy().flatten()
            zeros = np.sum(param_array == 0)
            total = param_array.size
            
            sparsity = zeros / total * 100 if total > 0 else 0
            print(f"{name}: shape={param.shape}, size={total}, zeros={zeros}, sparsity={sparsity:.2f}%")
            
            total_params += total
            zero_params += zeros
        
        overall_sparsity = zero_params / total_params * 100 if total_params > 0 else 0
        print(f"\nOverall sparsity: {overall_sparsity:.2f}% ({zero_params}/{total_params} parameters are zero)")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        if isinstance(state_dict, dict):
            print("\nAnalyzing state_dict directly:")
            total_params = 0
            zero_params = 0
            
            for name, param in state_dict.items():
                if hasattr(param, 'cpu') and hasattr(param, 'numpy'):
                    param_array = param.cpu().numpy().flatten()
                    zeros = np.sum(param_array == 0)
                    total = param_array.size
                    
                    sparsity = zeros / total * 100 if total > 0 else 0
                    print(f"{name}: shape={param.shape}, size={total}, zeros={zeros}, sparsity={sparsity:.2f}%")
                    
                    total_params += total
                    zero_params += zeros
            
            overall_sparsity = zero_params / total_params * 100 if total_params > 0 else 0
            print(f"\nOverall sparsity: {overall_sparsity:.2f}% ({zero_params}/{total_params} parameters are zero)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze checkpoint sparsity")
    parser.add_argument('--checkpoint_path', type=str, default='4_run0_save056_epoch0106.pth',
                        help='Path to the checkpoint file')
    
    args = parser.parse_args()
    analyze_checkpoint(args.checkpoint_path) 