#!/usr/bin/env python3
"""
Main entry point for DCGAN Face Generation.
This script provides a clean interface to all GAN functionality.
"""

import os
import sys
import argparse
import json

# Fix OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def load_config(config_path='configs/config.json'):
    """Load configuration from JSON file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def main():
    """Main function with organized interface."""
    parser = argparse.ArgumentParser(description='DCGAN Face Generation - Organized Interface')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the GAN')
    train_parser.add_argument('--dataset-path', type=str, required=True,
                             help='Path to dataset directory')
    train_parser.add_argument('--config', type=str, default='configs/config.json',
                             help='Path to configuration file')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    train_parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate samples')
    gen_parser.add_argument('--checkpoint', type=str, 
                           default='outputs/checkpoints/checkpoint_epoch_10.pth',
                           help='Path to checkpoint file')
    gen_parser.add_argument('--num-samples', type=int, default=16,
                           help='Number of samples to generate')
    gen_parser.add_argument('--output-dir', type=str, default='outputs/generated_samples',
                           help='Output directory for generated samples')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demo')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run installation test')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show project information')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Load config
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.epochs:
            config['training']['epochs'] = args.epochs
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
        
        # Build command
        cmd = [
            sys.executable, 'scripts/main.py', 'train',
            '--dataset-path', args.dataset_path,
            '--epochs', str(config['training']['epochs']),
            '--batch-size', str(config['training']['batch_size']),
            '--image-size', str(config['model']['image_size']),
            '--learning-rate', str(config['training']['learning_rate']),
            '--save-interval', str(config['training']['save_interval']),
            '--sample-interval', str(config['training']['sample_interval'])
        ]
        
        print("Starting GAN training with configuration:")
        print(f"  Dataset: {args.dataset_path}")
        print(f"  Epochs: {config['training']['epochs']}")
        print(f"  Batch Size: {config['training']['batch_size']}")
        print(f"  Image Size: {config['model']['image_size']}")
        print(f"  Learning Rate: {config['training']['learning_rate']}")
        
        import subprocess
        subprocess.run(cmd, check=True)
        
    elif args.command == 'generate':
        cmd = [
            sys.executable, 'scripts/generate_samples.py',
            '--checkpoint', args.checkpoint,
            '--num-samples', str(args.num_samples),
            '--save-dir', args.output_dir
        ]
        
        print(f"Generating {args.num_samples} samples from {args.checkpoint}")
        import subprocess
        subprocess.run(cmd, check=True)
        
    elif args.command == 'demo':
        print("Running interactive demo...")
        import subprocess
        subprocess.run([sys.executable, 'scripts/demo.py'], check=True)
        
    elif args.command == 'test':
        print("Running installation test...")
        import subprocess
        subprocess.run([sys.executable, 'scripts/test_installation.py'], check=True)
        
    elif args.command == 'info':
        show_project_info()
        
    else:
        parser.print_help()
        print("\nQuick Examples:")
        print("  python run.py train data/faces")
        print("  python run.py generate --num-samples 64")
        print("  python run.py demo")
        print("  python run.py test")

def show_project_info():
    """Show project information and structure."""
    print("DCGAN Face Generation - Project Information")
    print("=" * 50)
    
    print("\n📁 Project Structure:")
    print("  src/                    - Core source code")
    print("    ├── models.py         - Generator and Discriminator")
    print("    ├── dataset.py        - Data loading and preprocessing")
    print("    ├── trainer.py        - Training loop and utilities")
    print("    └── utils.py          - Utility functions")
    
    print("\n  scripts/               - Executable scripts")
    print("    ├── main.py           - Main training script")
    print("    ├── demo.py           - Interactive demo")
    print("    ├── generate_samples.py - Sample generation")
    print("    └── run_gan.py        - Quick runner")
    
    print("\n  configs/               - Configuration files")
    print("    └── config.json       - Training parameters")
    
    print("\n  outputs/               - Model outputs")
    print("    ├── checkpoints/      - Trained models")
    print("    ├── samples/          - Training progress")
    print("    └── generated_samples/ - Generated images")
    
    print("\n  logs/                  - Training logs")
    print("    └── gan_training/     - TensorBoard logs")
    
    print("\n  data/                  - Training datasets")
    
    print("\n🚀 Quick Start:")
    print("  1. python run.py test          # Test installation")
    print("  2. python run.py demo          # Run demo")
    print("  3. python run.py train data/faces  # Train on your data")
    print("  4. python run.py generate      # Generate samples")

if __name__ == '__main__':
    main() 