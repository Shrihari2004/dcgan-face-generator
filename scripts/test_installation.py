#!/usr/bin/env python3
"""
Test script to verify DCGAN installation and basic functionality.
"""

# Fix OpenMP runtime conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from models import Generator, Discriminator, weights_init
        print("✓ models.py imported successfully")
    except ImportError as e:
        print(f"✗ Error importing models.py: {e}")
        return False
    
    try:
        from dataset import get_dataloader, denormalize
        print("✓ dataset.py imported successfully")
    except ImportError as e:
        print(f"✗ Error importing dataset.py: {e}")
        return False
    
    try:
        from trainer import GANTrainer
        print("✓ trainer.py imported successfully")
    except ImportError as e:
        print(f"✗ Error importing trainer.py: {e}")
        return False
    
    try:
        from utils import create_synthetic_dataset
        print("✓ utils.py imported successfully")
    except ImportError as e:
        print(f"✗ Error importing utils.py: {e}")
        return False
    
    return True

def test_models():
    """Test model creation and forward pass."""
    print("\nTesting model creation...")
    
    try:
        from models import Generator, Discriminator, weights_init
        
        # Create models
        generator = Generator(latent_dim=100, ngf=64, channels=3)
        discriminator = Discriminator(ndf=64, channels=3)
        
        # Initialize weights
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        
        print(f"✓ Generator created with {sum(p.numel() for p in generator.parameters()):,} parameters")
        print(f"✓ Discriminator created with {sum(p.numel() for p in discriminator.parameters()):,} parameters")
        
        # Test forward pass
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator.to(device)
        discriminator.to(device)
        
        # Generate fake images
        noise = torch.randn(4, 100, device=device)
        fake_images = generator(noise)
        print(f"✓ Generator forward pass successful, output shape: {fake_images.shape}")
        
        # Test discriminator
        real_images = torch.randn(4, 3, 64, 64, device=device)
        real_outputs = discriminator(real_images)
        fake_outputs = discriminator(fake_images)
        print(f"✓ Discriminator forward pass successful")
        print(f"  Real outputs shape: {real_outputs.shape}")
        print(f"  Fake outputs shape: {fake_outputs.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing models: {e}")
        return False

def test_dataset():
    """Test dataset creation and loading."""
    print("\nTesting dataset functionality...")
    
    try:
        from utils import create_synthetic_dataset
        from dataset import get_dataloader
        
        # Create synthetic dataset
        create_synthetic_dataset(save_dir='data/test', num_images=10, image_size=64)
        print("✓ Synthetic dataset created")
        
        # Load dataset
        dataloader = get_dataloader(
            dataset_path='data/test',
            dataset_type='custom',
            image_size=64,
            batch_size=4,
            num_workers=0  # Use 0 for testing
        )
        
        print(f"✓ DataLoader created, dataset size: {len(dataloader.dataset)}")
        
        # Test loading a batch
        batch = next(iter(dataloader))
        print(f"✓ Batch loaded successfully, shape: {batch.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing dataset: {e}")
        return False

def test_training_setup():
    """Test training setup without actually training."""
    print("\nTesting training setup...")
    
    try:
        from models import Generator, Discriminator, weights_init
        from dataset import get_dataloader
        from trainer import GANTrainer
        
        # Create models
        generator = Generator(latent_dim=100, ngf=64, channels=3)
        discriminator = Discriminator(ndf=64, channels=3)
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        
        # Create small dataset
        from utils import create_synthetic_dataset
        create_synthetic_dataset(save_dir='data/test_train', num_images=20, image_size=64)
        
        # Create dataloader
        dataloader = get_dataloader(
            dataset_path='data/test_train',
            dataset_type='custom',
            image_size=64,
            batch_size=4,
            num_workers=0
        )
        
        # Create trainer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            dataloader=dataloader,
            device=device,
            lr=0.0002,
            latent_dim=100
        )
        
        print("✓ Trainer created successfully")
        
        # Test single training step
        batch = next(iter(dataloader))
        results = trainer.train_step(batch)
        print("✓ Training step completed successfully")
        print(f"  Generator loss: {results['g_loss']:.4f}")
        print(f"  Discriminator loss: {results['d_loss']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing training setup: {e}")
        return False

def test_gpu():
    """Test GPU availability and functionality."""
    print("\nTesting GPU availability...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test GPU tensor operations
        try:
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            z = torch.mm(x, y)
            print("✓ GPU tensor operations successful")
        except Exception as e:
            print(f"✗ GPU tensor operations failed: {e}")
            return False
    else:
        print("⚠ CUDA not available, using CPU")
    
    return True

def cleanup():
    """Clean up test files."""
    print("\nCleaning up test files...")
    
    test_dirs = ['data/test', 'data/test_train']
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
            print(f"✓ Removed {dir_path}")

def main():
    """Run all tests."""
    print("DCGAN Installation Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Test", test_models),
        ("Dataset Test", test_dataset),
        ("Training Setup Test", test_training_setup),
        ("GPU Test", test_gpu),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your DCGAN installation is working correctly.")
        print("\nYou can now:")
        print("1. Run the demo: python demo.py")
        print("2. Train a model: python main.py train --dataset-path your_data --epochs 100")
        print("3. Generate samples: python main.py generate --checkpoint checkpoints/checkpoint_final.pth")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    # Cleanup
    cleanup()
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
