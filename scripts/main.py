#!/usr/bin/env python3
"""
Main script for training a DCGAN to generate realistic faces.
This script provides a complete pipeline for training, evaluation, and sample generation.
"""

# Fix OpenMP runtime conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import argparse
import sys
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import Generator, Discriminator, weights_init
from dataset import get_dataloader
from trainer import GANTrainer

def setup_device():
    """
    Setup device (GPU/CPU) for training.
    Returns:
        Device string and whether CUDA is available
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print("Using CPU for training")
    
    return device

def create_models(latent_dim=100, ngf=48, ndf=48, channels=3):
    """
    Create optimized Generator and Discriminator models.
    Args:
        latent_dim: Dimension of latent space
        ngf: Number of generator features (reduced for optimization)
        ndf: Number of discriminator features (reduced for optimization)
        channels: Number of image channels
    Returns:
        Tuple of (generator, discriminator)
    """
    print("Creating optimized Generator and Discriminator models...")
    
    # Create models with reduced features
    generator = Generator(latent_dim=latent_dim, ngf=ngf, channels=channels)
    discriminator = Discriminator(ndf=ndf, channels=channels)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Print model information
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    return generator, discriminator

def train_gan(args):
    """
    Main training function.
    Args:
        args: Command line arguments
    """
    print("=" * 60)
    print("DCGAN Training for Face Generation")
    print("=" * 60)
    
    # Setup device
    device = setup_device()
    
    # Create data loader
    print(f"\nLoading dataset from: {args.dataset_path}")
    dataloader = get_dataloader(
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create models
    generator, discriminator = create_models(
        latent_dim=args.latent_dim,
        ngf=args.ngf,
        ndf=args.ndf,
        channels=args.channels
    )
    
    # Create trainer
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        lr=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        latent_dim=args.latent_dim
    )
    
    # Load checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        else:
            print(f"Checkpoint not found: {args.resume}")
            return
    
    # Start training
    print(f"\nStarting training for {args.epochs} epochs...")
    trainer.train(
        num_epochs=args.epochs,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval
    )
    
    # Plot training history
    print("\nGenerating training history plots...")
    trainer.plot_training_history()
    
    # Generate interpolation
    print("Generating interpolation samples...")
    trainer.generate_interpolation()
    
    print("\nTraining completed successfully!")

def generate_samples(args):
    """
    Generate samples from a trained model.
    Args:
        args: Command line arguments
    """
    print("=" * 60)
    print("Generating Samples from Trained GAN")
    print("=" * 60)
    
    # Setup device
    device = setup_device()
    
    # Create generator
    generator, _ = create_models(
        latent_dim=args.latent_dim,
        ngf=args.ngf,
        channels=args.channels
    )
    
    # Load trained generator
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    with torch.no_grad():
        noise = torch.randn(args.num_samples, args.latent_dim, device=device)
        fake_images = generator(noise)
    
    # Save samples
    from dataset import save_sample_images
    os.makedirs('generated_samples', exist_ok=True)
    save_sample_images(fake_images, 'generated_samples', 0, args.num_samples)
    
    print("Samples generated and saved to 'generated_samples' directory")

def main():
    """
    Main function with argument parsing.
    """
    parser = argparse.ArgumentParser(description='DCGAN Training for Face Generation')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training parser
    train_parser = subparsers.add_parser('train', help='Train the GAN')
    train_parser.add_argument('--dataset-path', type=str, required=True,
                             help='Path to dataset directory')
    train_parser.add_argument('--dataset-type', type=str, default='custom',
                             choices=['custom', 'celeba'],
                             help='Type of dataset')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size for training (optimized)')
    train_parser.add_argument('--image-size', type=int, default=64,
                             help='Size of input images')
    train_parser.add_argument('--latent-dim', type=int, default=100,
                             help='Dimension of latent space')
    train_parser.add_argument('--ngf', type=int, default=48,
                             help='Number of generator features (optimized)')
    train_parser.add_argument('--ndf', type=int, default=48,
                             help='Number of discriminator features (optimized)')
    train_parser.add_argument('--channels', type=int, default=3,
                             help='Number of image channels')
    train_parser.add_argument('--learning-rate', type=float, default=0.0002,
                             help='Learning rate')
    train_parser.add_argument('--beta1', type=float, default=0.5,
                             help='Beta1 for Adam optimizer')
    train_parser.add_argument('--beta2', type=float, default=0.999,
                             help='Beta2 for Adam optimizer')
    train_parser.add_argument('--num-workers', type=int, default=2,
                             help='Number of data loader workers')
    train_parser.add_argument('--save-interval', type=int, default=20,
                             help='Interval for saving checkpoints (optimized)')
    train_parser.add_argument('--sample-interval', type=int, default=10,
                             help='Interval for generating samples (optimized)')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Path to checkpoint to resume from')
    
    # Generate parser
    gen_parser = subparsers.add_parser('generate', help='Generate samples')
    gen_parser.add_argument('--checkpoint', type=str, required=True,
                           help='Path to trained model checkpoint')
    gen_parser.add_argument('--num-samples', type=int, default=16,
                           help='Number of samples to generate')
    gen_parser.add_argument('--latent-dim', type=int, default=100,
                           help='Dimension of latent space')
    gen_parser.add_argument('--ngf', type=int, default=48,
                           help='Number of generator features (optimized)')
    gen_parser.add_argument('--channels', type=int, default=3,
                           help='Number of image channels')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_gan(args)
    elif args.command == 'generate':
        generate_samples(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 