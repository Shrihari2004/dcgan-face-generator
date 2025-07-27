#!/usr/bin/env python3
"""
Simple script to generate samples from trained GAN model.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import Generator
from dataset import denormalize

# Fix OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def generate_samples(checkpoint_path, num_samples=16, save_dir='generated_samples'):
    """
    Generate samples from a trained GAN model.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        num_samples: Number of samples to generate
        save_dir: Directory to save generated images
    """
    print(f"Generating {num_samples} samples from {checkpoint_path}")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create generator
    generator = Generator(latent_dim=100, ngf=64, channels=3)
    generator.to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Generate samples
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, 100, device=device)
        fake_images = generator(noise)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save individual images
    for i in range(num_samples):
        img = denormalize(fake_images[i]).permute(1, 2, 0).cpu().numpy()
        plt.imsave(os.path.join(save_dir, f'sample_{i:03d}.png'), img)
    
    # Create grid
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(min(num_samples, 16)):
        img = denormalize(fake_images[i]).permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'samples_grid.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated {num_samples} samples saved to '{save_dir}' directory")
    print(f"Grid image saved as '{save_dir}/samples_grid.png'")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate samples from trained GAN')
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoints/checkpoint_epoch_10.pth',
                       help='Path to checkpoint file')
    parser.add_argument('--num-samples', type=int, default=16,
                       help='Number of samples to generate')
    parser.add_argument('--save-dir', type=str, default='generated_samples',
                       help='Directory to save generated images')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Available checkpoints:")
        if os.path.exists('outputs/checkpoints'):
            for file in os.listdir('outputs/checkpoints'):
                if file.endswith('.pth'):
                    print(f"  - outputs/checkpoints/{file}")
        return
    
    generate_samples(args.checkpoint, args.num_samples, args.save_dir)

if __name__ == '__main__':
    main() 