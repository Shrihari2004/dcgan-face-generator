import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import glob
from PIL import Image
import io
from models import Generator, Discriminator, weights_init, compress_model
from dataset import save_sample_images, denormalize

class OptimizedGANTrainer:
    """
    Optimized GAN trainer with compression and size reduction features.
    Maintains performance while reducing storage requirements.
    """
    def __init__(self, generator, discriminator, dataloader, device='cuda',
                 lr=0.0002, beta1=0.5, beta2=0.999, latent_dim=100, config=None):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.dataloader = dataloader
        self.device = device
        self.latent_dim = latent_dim
        self.config = config or {}
        
        # Apply model compression if specified
        if self.config.get('optimization', {}).get('compression_ratio', 1.0) < 1.0:
            compression_ratio = self.config['optimization']['compression_ratio']
            self.generator = compress_model(self.generator, compression_ratio)
            self.discriminator = compress_model(self.discriminator, compression_ratio)
            print(f"Applied model compression with ratio: {compression_ratio}")
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(beta1, beta2)
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        self.real_scores = []
        self.fake_scores = []
        
        # TensorBoard writer
        self.writer = SummaryWriter('logs/gan_training')
        
        # Create directories for saving
        self.save_dir = 'outputs/checkpoints'
        self.sample_dir = 'outputs/samples'
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Fixed noise for consistent sample generation
        self.fixed_noise = torch.randn(64, latent_dim, device=device)
        
        # Checkpoint management
        self.max_checkpoints = self.config.get('optimization', {}).get('max_checkpoints', 3)
        
        print(f"Training on device: {device}")
        print(f"Dataset size: {len(dataloader.dataset)}")
        print(f"Batch size: {dataloader.batch_size}")
        print(f"Max checkpoints to keep: {self.max_checkpoints}")
        
    def train_step(self, real_images):
        """
        Single training step for both generator and discriminator.
        Args:
            real_images: Batch of real images
        Returns:
            Dictionary containing losses and scores
        """
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        
        # Labels for real and fake images
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real images
        real_outputs = self.discriminator(real_images)
        d_real_loss = self.criterion(real_outputs, real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise)
        fake_outputs = self.discriminator(fake_images.detach())
        d_fake_loss = self.criterion(fake_outputs, fake_labels)
        
        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        # Generate fake images again (since discriminator was updated)
        fake_outputs = self.discriminator(fake_images)
        g_loss = self.criterion(fake_outputs, real_labels)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        # Calculate scores for monitoring
        real_score = torch.sigmoid(real_outputs).mean().item()
        fake_score = torch.sigmoid(fake_outputs).mean().item()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'real_score': real_score,
            'fake_score': fake_score
        }
    
    def compress_image(self, image_tensor, quality=85):
        """
        Compress image tensor to reduce storage size.
        Args:
            image_tensor: Image tensor to compress
            quality: JPEG quality (1-100)
        Returns:
            Compressed image tensor
        """
        # Convert tensor to PIL Image
        img = denormalize(image_tensor).permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        
        # Compress using JPEG
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality, optimize=True)
        buffer.seek(0)
        
        # Convert back to tensor
        compressed_img = Image.open(buffer)
        compressed_tensor = torch.from_numpy(np.array(compressed_img)).float() / 255.0
        compressed_tensor = compressed_tensor.permute(2, 0, 1)
        
        return compressed_tensor
    
    def save_compressed_samples(self, fake_images, epoch, batch_idx=None):
        """
        Save compressed sample images to reduce storage.
        Args:
            fake_images: Generated images tensor
            epoch: Current epoch number
            batch_idx: Current batch index (optional)
        """
        quality = self.config.get('optimization', {}).get('sample_quality', 85)
        
        # Create a grid of images
        grid_size = int(np.ceil(np.sqrt(fake_images.size(0))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        axes = axes.flatten()
        
        for i, img in enumerate(fake_images):
            if i < len(axes):
                # Compress image
                compressed_img = self.compress_image(img, quality)
                img_np = compressed_img.permute(1, 2, 0).numpy()
                axes[i].imshow(img_np)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(fake_images.size(0), len(axes)):
            axes[i].axis('off')
        
        # Save with compression
        if batch_idx is not None:
            save_path = os.path.join(self.sample_dir, f'samples_e{epoch}_b{batch_idx}.jpg')
        else:
            save_path = os.path.join(self.sample_dir, f'samples_epoch_{epoch}.jpg')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', quality=quality, optimize=True)
        plt.close()
    
    def cleanup_old_checkpoints(self):
        """
        Remove old checkpoints to maintain storage limits.
        """
        checkpoint_files = glob.glob(os.path.join(self.save_dir, 'checkpoint_epoch_*.pth'))
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Keep only the most recent checkpoints
        if len(checkpoint_files) > self.max_checkpoints:
            files_to_remove = checkpoint_files[:-self.max_checkpoints]
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    print(f"Removed old checkpoint: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")
    
    def train(self, num_epochs, save_interval=20, sample_interval=10):
        """
        Main training loop with optimizations.
        Args:
            num_epochs: Number of epochs to train
            save_interval: Interval for saving checkpoints
            sample_interval: Interval for generating samples
        """
        print("Starting optimized training...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_g_losses = []
            epoch_d_losses = []
            epoch_real_scores = []
            epoch_fake_scores = []
            
            # Training loop with progress bar
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch_idx, real_images in enumerate(pbar):
                # Train step
                results = self.train_step(real_images)
                
                # Store results
                epoch_g_losses.append(results['g_loss'])
                epoch_d_losses.append(results['d_loss'])
                epoch_real_scores.append(results['real_score'])
                epoch_fake_scores.append(results['fake_score'])
                
                # Update progress bar
                pbar.set_postfix({
                    'G_Loss': f"{results['g_loss']:.4f}",
                    'D_Loss': f"{results['d_loss']:.4f}",
                    'Real_Score': f"{results['real_score']:.4f}",
                    'Fake_Score': f"{results['fake_score']:.4f}"
                })
                
                # Generate samples periodically (less frequently)
                if batch_idx % 200 == 0:  # Reduced from 100
                    self.generate_samples(epoch, batch_idx)
            
            # Calculate epoch averages
            avg_g_loss = np.mean(epoch_g_losses)
            avg_d_loss = np.mean(epoch_d_losses)
            avg_real_score = np.mean(epoch_real_scores)
            avg_fake_score = np.mean(epoch_fake_scores)
            
            # Store history
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            self.real_scores.append(avg_real_score)
            self.fake_scores.append(avg_fake_score)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/Generator', avg_g_loss, epoch)
            self.writer.add_scalar('Loss/Discriminator', avg_d_loss, epoch)
            self.writer.add_scalar('Score/Real', avg_real_score, epoch)
            self.writer.add_scalar('Score/Fake', avg_fake_score, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"Generator Loss: {avg_g_loss:.4f}")
            print(f"Discriminator Loss: {avg_d_loss:.4f}")
            print(f"Real Score: {avg_real_score:.4f}")
            print(f"Fake Score: {avg_fake_score:.4f}")
            
            # Save checkpoints (less frequently)
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
                self.cleanup_old_checkpoints()
            
            # Generate samples (less frequently)
            if (epoch + 1) % sample_interval == 0:
                self.generate_samples(epoch + 1)
        
        # Final save
        self.save_checkpoint(num_epochs, is_final=True)
        self.writer.close()
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        
    def generate_samples(self, epoch, batch_idx=None):
        """
        Generate and save compressed sample images.
        Args:
            epoch: Current epoch number
            batch_idx: Current batch index (optional)
        """
        self.generator.eval()
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise)
            
            # Save compressed samples
            self.save_compressed_samples(fake_images, epoch, batch_idx)
        
        self.generator.train()
    
    def save_checkpoint(self, epoch, is_final=False):
        """
        Save training checkpoint with quantization if enabled.
        Args:
            epoch: Current epoch number
            is_final: Whether this is the final checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'real_scores': self.real_scores,
            'fake_scores': self.fake_scores,
        }
        
        if is_final:
            filename = 'checkpoint_final.pth'
        else:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        # Apply quantization if enabled
        if self.config.get('optimization', {}).get('quantization', False):
            checkpoint = {k: v.half() if isinstance(v, torch.Tensor) else v 
                         for k, v in checkpoint.items()}
        
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load training checkpoint.
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        self.real_scores = checkpoint['real_scores']
        self.fake_scores = checkpoint['fake_scores']
        
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def plot_training_history(self, save_path='training_history.png'):
        """
        Plot training history including losses and scores.
        Args:
            save_path: Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Generator Loss
        ax1.plot(self.g_losses, label='Generator Loss', color='blue')
        ax1.set_title('Generator Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Discriminator Loss
        ax2.plot(self.d_losses, label='Discriminator Loss', color='red')
        ax2.set_title('Discriminator Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Real vs Fake Scores
        ax3.plot(self.real_scores, label='Real Score', color='green')
        ax3.plot(self.fake_scores, label='Fake Score', color='orange')
        ax3.set_title('Real vs Fake Scores')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True)
        
        # Loss Ratio
        loss_ratio = [g/d if d > 0 else 0 for g, d in zip(self.g_losses, self.d_losses)]
        ax4.plot(loss_ratio, label='G/D Loss Ratio', color='purple')
        ax4.set_title('Generator/Discriminator Loss Ratio')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Ratio')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', optimize=True)
        plt.close()
        print(f"Training history saved to {save_path}")
    
    def generate_interpolation(self, num_steps=10, save_path='interpolation.png'):
        """
        Generate interpolation between two random noise vectors.
        Args:
            num_steps: Number of interpolation steps
            save_path: Path to save the interpolation
        """
        self.generator.eval()
        
        # Generate two random noise vectors
        noise1 = torch.randn(1, self.latent_dim, device=self.device)
        noise2 = torch.randn(1, self.latent_dim, device=self.device)
        
        # Interpolate between them
        interpolated_images = []
        for alpha in np.linspace(0, 1, num_steps):
            interpolated_noise = alpha * noise1 + (1 - alpha) * noise2
            with torch.no_grad():
                fake_image = self.generator(interpolated_noise)
                interpolated_images.append(fake_image)
        
        # Create interpolation grid
        fig, axes = plt.subplots(1, num_steps, figsize=(2*num_steps, 2))
        for i, img in enumerate(interpolated_images):
            img = denormalize(img[0]).permute(1, 2, 0).cpu().numpy()
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f'Step {i+1}')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', optimize=True)
        plt.close()
        print(f"Interpolation saved to {save_path}")
        
        self.generator.train()

# Backward compatibility
GANTrainer = OptimizedGANTrainer 