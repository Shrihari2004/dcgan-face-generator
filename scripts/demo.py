#!/usr/bin/env python3
"""
Demo script for DCGAN Face Generation.
This script provides quick examples of how to use the GAN implementation.
"""

# Fix OpenMP runtime conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import Generator, Discriminator, weights_init
from dataset import get_dataloader, denormalize, save_sample_images
from trainer import GANTrainer
from utils import create_synthetic_dataset, visualize_latent_space

def demo_synthetic_dataset():
    """
    Demo: Create and train on synthetic dataset.
    """
    print("=" * 60)
    print("Demo: Training on Synthetic Dataset")
    print("=" * 60)
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    create_synthetic_dataset(save_dir='data/synthetic', num_images=1000, image_size=64)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loader
    dataloader = get_dataloader(
        dataset_path='data/synthetic',
        dataset_type='custom',
        image_size=64,
        batch_size=32,
        num_workers=2
    )
    
    # Create models
    generator = Generator(latent_dim=100, ngf=64, channels=3)
    discriminator = Discriminator(ndf=64, channels=3)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Create trainer
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        lr=0.0002,
        latent_dim=100
    )
    
    # Train for a few epochs
    print("Training for 10 epochs...")
    trainer.train(num_epochs=10, save_interval=5, sample_interval=2)
    
    # Generate final samples
    print("Generating final samples...")
    trainer.generate_samples(10)
    
    print("Demo completed! Check the 'samples' directory for generated images.")

def demo_model_architecture():
    """
    Demo: Show model architecture and parameters.
    """
    print("=" * 60)
    print("Demo: Model Architecture")
    print("=" * 60)
    
    # Create models
    generator = Generator(latent_dim=100, ngf=64, channels=3)
    discriminator = Discriminator(ndf=64, channels=3)
    
    # Print model information
    print("Generator Architecture:")
    print(generator)
    print(f"\nGenerator Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    print("\n" + "="*40 + "\n")
    
    print("Discriminator Architecture:")
    print(discriminator)
    print(f"\nDiscriminator Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    total_params = sum(p.numel() for p in generator.parameters()) + \
                  sum(p.numel() for p in discriminator.parameters())
    print(f"\nTotal Parameters: {total_params:,}")

def demo_sample_generation():
    """
    Demo: Generate samples from untrained model (random noise).
    """
    print("=" * 60)
    print("Demo: Sample Generation (Untrained Model)")
    print("=" * 60)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create generator
    generator = Generator(latent_dim=100, ngf=64, channels=3)
    generator.to(device)
    generator.eval()
    
    # Generate random samples
    print("Generating random samples from untrained model...")
    with torch.no_grad():
        noise = torch.randn(16, 100, device=device)
        fake_images = generator(noise)
    
    # Save samples
    os.makedirs('demo_samples', exist_ok=True)
    save_sample_images(fake_images, 'demo_samples', 0, 16)
    
    print("Random samples saved to 'demo_samples' directory")
    print("Note: These will look like random noise since the model is untrained.")

def demo_latent_space_interpolation():
    """
    Demo: Show latent space interpolation.
    """
    print("=" * 60)
    print("Demo: Latent Space Interpolation")
    print("=" * 60)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create generator
    generator = Generator(latent_dim=100, ngf=64, channels=3)
    generator.to(device)
    generator.eval()
    
    # Create two random noise vectors
    noise1 = torch.randn(1, 100, device=device)
    noise2 = torch.randn(1, 100, device=device)
    
    # Interpolate between them
    print("Generating interpolation between two random noise vectors...")
    interpolated_images = []
    
    with torch.no_grad():
        for alpha in np.linspace(0, 1, 10):
            interpolated_noise = alpha * noise1 + (1 - alpha) * noise2
            fake_image = generator(interpolated_noise)
            interpolated_images.append(fake_image[0])
    
    # Create interpolation visualization
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i, img in enumerate(interpolated_images):
        img = denormalize(img).permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Step {i+1}')
    
    plt.tight_layout()
    os.makedirs('demo_samples', exist_ok=True)
    plt.savefig('demo_samples/interpolation_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Interpolation saved to 'demo_samples/interpolation_demo.png'")

def demo_data_loading():
    """
    Demo: Show data loading and preprocessing.
    """
    print("=" * 60)
    print("Demo: Data Loading and Preprocessing")
    print("=" * 60)
    
    # Create synthetic dataset if it doesn't exist
    if not os.path.exists('data/synthetic'):
        print("Creating synthetic dataset...")
        create_synthetic_dataset(save_dir='data/synthetic', num_images=100, image_size=64)
    
    # Load dataset
    print("Loading dataset...")
    dataloader = get_dataloader(
        dataset_path='data/synthetic',
        dataset_type='custom',
        image_size=64,
        batch_size=16,
        num_workers=2
    )
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {dataloader.batch_size}")
    
    # Show a few samples
    print("\nLoading a batch of images...")
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")
    print(f"Image range: [{batch.min():.3f}, {batch.max():.3f}]")
    
    # Visualize batch
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(min(8, batch.size(0))):
        img = denormalize(batch[i]).permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Image {i+1}')
    
    plt.tight_layout()
    os.makedirs('demo_samples', exist_ok=True)
    plt.savefig('demo_samples/data_samples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Data samples saved to 'demo_samples/data_samples.png'")

def main():
    """
    Main demo function.
    """
    print("DCGAN Face Generation - Demo Script")
    print("=" * 60)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    
    # Run demos
    demos = [
        ("Model Architecture", demo_model_architecture),
        ("Data Loading", demo_data_loading),
        ("Sample Generation", demo_sample_generation),
        ("Latent Space Interpolation", demo_latent_space_interpolation),
        ("Synthetic Dataset Training", demo_synthetic_dataset),
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{i}. {name}")
    
    print("\n0. Run all demos")
    
    try:
        choice = input("\nSelect a demo to run (0-5): ").strip()
        
        if choice == "0":
            # Run all demos
            for name, demo_func in demos:
                print(f"\n{'='*20} {name} {'='*20}")
                demo_func()
        elif choice in ["1", "2", "3", "4", "5"]:
            # Run specific demo
            idx = int(choice) - 1
            name, demo_func = demos[idx]
            demo_func()
        else:
            print("Invalid choice. Running all demos...")
            for name, demo_func in demos:
                print(f"\n{'='*20} {name} {'='*20}")
                demo_func()
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError during demo: {e}")
    
    print("\nDemo completed!")

if __name__ == '__main__':
    main() 