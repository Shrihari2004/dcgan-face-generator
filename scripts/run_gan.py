#!/usr/bin/env python3
"""
Quick runner script for DCGAN with all fixes applied.
This script handles common issues and provides a simple interface.
"""

import os
import sys
import subprocess

def setup_environment():
    """Setup environment to avoid common issues."""
    # Fix OpenMP runtime conflict
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Set matplotlib backend to avoid GUI issues
    os.environ['MPLBACKEND'] = 'Agg'
    
    print("Environment setup completed:")
    print("✓ OpenMP conflict fix applied")
    print("✓ Matplotlib backend set to Agg")

def run_training(dataset_path, epochs=50, batch_size=32, image_size=64):
    """Run GAN training with default settings."""
    setup_environment()
    
    print(f"\nStarting GAN training...")
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}")
    
    # Build command
    cmd = [
        sys.executable, os.path.join('scripts', 'main.py'), 'train',
        '--dataset-path', dataset_path,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--image-size', str(image_size),
        '--save-interval', '10',
        '--sample-interval', '5'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error: {e}")
        return False
    
    return True

def run_demo():
    """Run the demo script."""
    setup_environment()
    
    print("\nRunning GAN demo...")
    
    try:
        subprocess.run([sys.executable, os.path.join('scripts', 'demo.py')], check=True)
        print("\n✓ Demo completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Demo failed with error: {e}")
        return False
    
    return True

def run_test():
    """Run the installation test."""
    setup_environment()
    
    print("\nRunning installation test...")
    
    try:
        subprocess.run([sys.executable, os.path.join('scripts', 'test_installation.py')], check=True)
        print("\n✓ Test completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Test failed with error: {e}")
        return False
    
    return True

def main():
    """Main function with simple interface."""
    print("DCGAN Quick Runner")
    print("=" * 40)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_gan.py train <dataset_path> [epochs] [batch_size]")
        print("  python run_gan.py demo")
        print("  python run_gan.py test")
        print("\nExamples:")
        print("  python run_gan.py train data/faces")
        print("  python run_gan.py train data/faces 100 64")
        print("  python run_gan.py demo")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'train':
        if len(sys.argv) < 3:
            print("Error: Please provide dataset path")
            return
        
        dataset_path = sys.argv[2]
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 32
        
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path '{dataset_path}' does not exist")
            print("Please create a folder with your face images or use the demo to create synthetic data")
            return
        
        run_training(dataset_path, epochs, batch_size)
        
    elif command == 'demo':
        run_demo()
        
    elif command == 'test':
        run_test()
        
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, demo, test")

if __name__ == '__main__':
    main() 