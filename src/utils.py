import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
import urllib.request
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

def download_celeba_dataset(save_dir='data/celeba', num_images=10000):
    """
    Download a subset of CelebA dataset for training.
    Args:
        save_dir: Directory to save the dataset
        num_images: Number of images to download (max 10000 for demo)
    """
    print("Downloading CelebA dataset...")
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # CelebA dataset URLs (small subset for demo)
    base_url = "https://github.com/ndb796/CelebA-HQ-Dataset/raw/master/data"
    
    # Download a small subset for demonstration
    for i in range(min(num_images, 1000)):
        try:
            url = f"{base_url}/{i:05d}.jpg"
            filename = os.path.join(save_dir, f"{i:05d}.jpg")
            
            if not os.path.exists(filename):
                urllib.request.urlretrieve(url, filename)
                
            if (i + 1) % 100 == 0:
                print(f"Downloaded {i + 1} images...")
                
        except Exception as e:
            print(f"Error downloading image {i}: {e}")
            continue
    
    print(f"Downloaded {num_images} images to {save_dir}")

def create_synthetic_dataset(save_dir='data/synthetic', num_images=1000, image_size=64):
    """
    Create a synthetic dataset for testing purposes.
    Args:
        save_dir: Directory to save synthetic images
        num_images: Number of synthetic images to create
        image_size: Size of synthetic images
    """
    print("Creating synthetic dataset...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_images):
        # Create random synthetic image
        img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        
        # Add some structure to make it more interesting
        # Create random geometric shapes
        for _ in range(np.random.randint(3, 8)):
            color = np.random.randint(0, 255, 3)
            x, y = np.random.randint(0, image_size, 2)
            radius = np.random.randint(5, 20)
            cv2.circle(img, (x, y), radius, color.tolist(), -1)
        
        # Save image
        filename = os.path.join(save_dir, f"synthetic_{i:04d}.jpg")
        cv2.imwrite(filename, img)
    
    print(f"Created {num_images} synthetic images in {save_dir}")

def calculate_fid_score(real_images, fake_images, device='cuda'):
    """
    Calculate Fréchet Inception Distance (FID) score.
    Note: This is a simplified version. For production, use the official FID implementation.
    Args:
        real_images: Tensor of real images
        fake_images: Tensor of fake images
        device: Device to run calculations on
    Returns:
        FID score (lower is better)
    """
    try:
        from pytorch_fid import fid_score
        return fid_score.calculate_fid_given_paths([real_images, fake_images], 
                                                 batch_size=50, device=device)
    except ImportError:
        print("pytorch-fid not installed. Skipping FID calculation.")
        return None

def evaluate_model(generator, discriminator, dataloader, device='cuda', num_samples=1000):
    """
    Evaluate the trained GAN model.
    Args:
        generator: Trained generator model
        discriminator: Trained discriminator model
        dataloader: Data loader for real images
        device: Device to run evaluation on
        num_samples: Number of samples to evaluate
    Returns:
        Dictionary containing evaluation metrics
    """
    generator.eval()
    discriminator.eval()
    
    real_scores = []
    fake_scores = []
    
    with torch.no_grad():
        # Evaluate on real images
        for i, real_images in enumerate(dataloader):
            if i * dataloader.batch_size >= num_samples:
                break
                
            real_images = real_images.to(device)
            real_outputs = discriminator(real_images)
            real_scores.extend(torch.sigmoid(real_outputs).cpu().numpy().flatten())
        
        # Generate and evaluate fake images
        for i in range(0, num_samples, dataloader.batch_size):
            batch_size = min(dataloader.batch_size, num_samples - i)
            noise = torch.randn(batch_size, generator.latent_dim, device=device)
            fake_images = generator(noise)
            fake_outputs = discriminator(fake_images)
            fake_scores.extend(torch.sigmoid(fake_outputs).cpu().numpy().flatten())
    
    # Calculate metrics
    real_mean = np.mean(real_scores)
    fake_mean = np.mean(fake_scores)
    real_std = np.std(real_scores)
    fake_std = np.std(fake_scores)
    
    # Calculate accuracy (assuming threshold of 0.5)
    real_accuracy = np.mean(np.array(real_scores) > 0.5)
    fake_accuracy = np.mean(np.array(fake_scores) < 0.5)
    total_accuracy = (real_accuracy + fake_accuracy) / 2
    
    return {
        'real_score_mean': real_mean,
        'fake_score_mean': fake_mean,
        'real_score_std': real_std,
        'fake_score_std': fake_std,
        'real_accuracy': real_accuracy,
        'fake_accuracy': fake_accuracy,
        'total_accuracy': total_accuracy
    }

def visualize_latent_space(generator, device='cuda', save_path='latent_space.png'):
    """
    Visualize the latent space by interpolating between points.
    Args:
        generator: Trained generator model
        device: Device to run on
        save_path: Path to save visualization
    """
    generator.eval()
    
    # Create a grid of latent vectors
    n_points = 8
    z1 = torch.randn(1, generator.latent_dim, device=device)
    z2 = torch.randn(1, generator.latent_dim, device=device)
    
    # Interpolate between them
    images = []
    for i in range(n_points):
        for j in range(n_points):
            alpha_i = i / (n_points - 1)
            alpha_j = j / (n_points - 1)
            
            # Interpolate in 2D space
            z_interp = (1 - alpha_i) * (1 - alpha_j) * z1 + \
                      alpha_i * (1 - alpha_j) * z2 + \
                      (1 - alpha_i) * alpha_j * torch.randn_like(z1) + \
                      alpha_i * alpha_j * torch.randn_like(z1)
            
            with torch.no_grad():
                fake_image = generator(z_interp)
                images.append(fake_image[0])
    
    # Create grid
    fig, axes = plt.subplots(n_points, n_points, figsize=(12, 12))
    for i in range(n_points):
        for j in range(n_points):
            idx = i * n_points + j
            img = denormalize(images[idx]).permute(1, 2, 0).cpu().numpy()
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Latent space visualization saved to {save_path}")

def denormalize(tensor):
    """
    Denormalize tensor from [-1, 1] to [0, 1] range.
    Args:
        tensor: Normalized tensor
    Returns:
        Denormalized tensor
    """
    return (tensor + 1) / 2

def save_model_summary(generator, discriminator, save_path='model_summary.txt'):
    """
    Save a detailed summary of the model architecture.
    Args:
        generator: Generator model
        discriminator: Discriminator model
        save_path: Path to save summary
    """
    with open(save_path, 'w') as f:
        f.write("DCGAN Model Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Generator Architecture:\n")
        f.write("-" * 30 + "\n")
        f.write(str(generator))
        f.write(f"\n\nGenerator Parameters: {sum(p.numel() for p in generator.parameters()):,}\n")
        
        f.write("\nDiscriminator Architecture:\n")
        f.write("-" * 30 + "\n")
        f.write(str(discriminator))
        f.write(f"\n\nDiscriminator Parameters: {sum(p.numel() for p in discriminator.parameters()):,}\n")
        
        total_params = sum(p.numel() for p in generator.parameters()) + \
                      sum(p.numel() for p in discriminator.parameters())
        f.write(f"\nTotal Parameters: {total_params:,}\n")
    
    print(f"Model summary saved to {save_path}")

def create_training_config(config_path='config.json'):
    """
    Create a default training configuration file.
    Args:
        config_path: Path to save configuration
    """
    import json
    
    config = {
        "training": {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "beta2": 0.999,
            "save_interval": 10,
            "sample_interval": 5
        },
        "model": {
            "latent_dim": 100,
            "ngf": 64,
            "ndf": 64,
            "channels": 3,
            "image_size": 64
        },
        "data": {
            "dataset_type": "custom",
            "num_workers": 2
        },
        "system": {
            "device": "auto",
            "mixed_precision": False
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Default configuration saved to {config_path}")

def load_config(config_path='config.json'):
    """
    Load training configuration from file.
    Args:
        config_path: Path to configuration file
    Returns:
        Configuration dictionary
    """
    import json
    
    if not os.path.exists(config_path):
        create_training_config(config_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config 