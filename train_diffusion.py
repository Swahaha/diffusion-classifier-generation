import torch
import argparse
from diffusion_model import WeightDiffusion, TinyCNN
from diffusion_trainer import WeightDataset, DiffusionTrainer
import os

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Starting Diffusion Training ===")
    print(f"Using device: {device}")
    
    # Verify checkpoint directory exists
    if not os.path.exists(args.checkpoint_dir):
        raise ValueError(f"Checkpoint directory {args.checkpoint_dir} does not exist!")
    
    # Calculate weight dimension from TinyCNN
    dummy_model = TinyCNN()
    weight_dim = sum(p.numel() for p in dummy_model.parameters())
    print(f"Total number of parameters: {weight_dim}")
    
    # Create dataset from checkpoints
    print(f"\nLoading checkpoints from: {args.checkpoint_dir}")
    dataset = WeightDataset(args.checkpoint_dir)
    print(f"Found {len(dataset)} checkpoint files")
    
    # Initialize improved diffusion model with stronger architecture
    print("\nInitializing diffusion model...")
    model = WeightDiffusion(
        weight_dim=weight_dim,
        time_dim=256,
        hidden_dims=[512, 512, 512, 256],
        dropout_rate=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Diffusion model parameters: {total_params:,}")
    
    # Initialize trainer with improved beta schedule
    print("\nInitializing trainer...")
    trainer = DiffusionTrainer(
        model, 
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule
    )
    
    # Train the model with adjusted parameters
    print("\nStarting training...")
    print(f"Training parameters:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Weight decay: {args.weight_decay}")
    print(f"  - Dropout rate: {args.dropout}")
    print(f"  - Timesteps: {args.timesteps}")
    print(f"  - Beta schedule: {args.beta_schedule}")
    print(f"  - Save strategy: Every improvement (best model only)")
    
    trainer.train(
        dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        save_best_only=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the diffusion model')
    parser.add_argument('--checkpoint_dir', type=str, default='Toy_CNN',
                        help='Directory containing the checkpoints')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate in the model')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--beta_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='Type of beta schedule to use')
    
    args = parser.parse_args()
    main(args) 