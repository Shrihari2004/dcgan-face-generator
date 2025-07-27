import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from torchvision.datasets import ImageFolder, CelebA

class FaceDataset(Dataset):
    """
    Custom dataset class for face images.
    Supports both CelebA and custom image folders.
    """
    def __init__(self, root_dir, transform=None, image_size=64):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        
        # Get list of image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(
                [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                 if f.lower().endswith(ext.replace('*', ''))]
            )
        
        print(f"Found {len(self.image_files)} images in {root_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image

def get_transforms(image_size=64):
    """
    Get data transformations for training.
    Args:
        image_size: Size to resize images to
    Returns:
        Composition of transforms
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def get_celeba_dataset(root_dir, image_size=64, batch_size=64, num_workers=2):
    """
    Load CelebA dataset with proper transformations.
    Args:
        root_dir: Directory containing CelebA images
        image_size: Size to resize images to
        batch_size: Batch size for data loader
        num_workers: Number of worker processes
    Returns:
        DataLoader for CelebA dataset
    """
    transform = get_transforms(image_size)
    
    # Create dataset
    dataset = ImageFolder(
        root=root_dir,
        transform=transform
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Only use pin_memory if CUDA is available
    )
    
    return dataloader

def get_custom_dataset(root_dir, image_size=64, batch_size=64, num_workers=2):
    """
    Load custom dataset from image folder.
    Args:
        root_dir: Directory containing images
        image_size: Size to resize images to
        batch_size: Batch size for data loader
        num_workers: Number of worker processes
    Returns:
        DataLoader for custom dataset
    """
    transform = get_transforms(image_size)
    
    # Create dataset
    dataset = FaceDataset(
        root_dir=root_dir,
        transform=transform,
        image_size=image_size
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Only use pin_memory if CUDA is available
    )
    
    return dataloader

def get_dataloader(dataset_path, dataset_type='custom', image_size=64, 
                  batch_size=64, num_workers=2):
    """
    Get data loader based on dataset type.
    Args:
        dataset_path: Path to dataset
        dataset_type: Type of dataset ('celeba' or 'custom')
        image_size: Size to resize images to
        batch_size: Batch size for data loader
        num_workers: Number of worker processes
    Returns:
        DataLoader for the specified dataset
    """
    if dataset_type.lower() == 'celeba':
        return get_celeba_dataset(
            dataset_path, image_size, batch_size, num_workers
        )
    else:
        return get_custom_dataset(
            dataset_path, image_size, batch_size, num_workers
        )

def denormalize(tensor):
    """
    Denormalize tensor from [-1, 1] to [0, 1] range for visualization.
    Args:
        tensor: Normalized tensor
    Returns:
        Denormalized tensor
    """
    return (tensor + 1) / 2

def save_sample_images(images, save_path, epoch, num_images=16):
    """
    Save a grid of sample images.
    Args:
        images: Tensor of images to save
        save_path: Directory to save images
        epoch: Current epoch number
        num_images: Number of images to save
    """
    import matplotlib.pyplot as plt
    
    # Denormalize images
    images = denormalize(images[:num_images])
    
    # Create grid
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'samples_epoch_{epoch}.png'))
    plt.close() 