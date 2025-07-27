#!/usr/bin/env python3
"""
Optimized DCGAN Face Generator - Main Entry Point
Reduced size with maintained performance through compression and optimization.
"""

import sys
import os
import argparse
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_config():
    """Load configuration from config file."""
    config_path = 'configs/config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def show_info():
    """Display project information and optimization details."""
    config = load_config()
    
    print("=" * 60)
    print("🎯 OPTIMIZED DCGAN FACE GENERATOR")
    print("=" * 60)
    print("📦 Project Size: Optimized for minimal storage footprint")
    print("🚀 Performance: Maintained accuracy with reduced resources")
    print("💾 Compression: Automatic image and model compression")
    print("🧹 Cleanup: Intelligent checkpoint and sample management")
    print("=" * 60)
    
    if config:
        print("\n📋 Current Configuration:")
        print(f"   • Model Features: {config.get('model', {}).get('ngf', 48)} (optimized)")
        print(f"   • Batch Size: {config.get('training', {}).get('batch_size', 32)} (optimized)")
        print(f"   • Checkpoints: Max {config.get('optimization', {}).get('max_checkpoints', 3)}")
        print(f"   • Image Quality: {config.get('optimization', {}).get('sample_quality', 85)}%")
        print(f"   • Compression: {config.get('optimization', {}).get('compression_ratio', 0.75)}x")
    
    print("\n🔧 Available Commands:")
    print("   • python run.py train <dataset_path>     - Train optimized model")
    print("   • python run.py generate --num-samples N  - Generate samples")
    print("   • python run.py demo                     - Interactive demo")
    print("   • python run.py cleanup                  - Reduce project size")
    print("   • python run.py test                     - Test installation")
    print("   • python run.py info                     - Show this info")
    print("=" * 60)

def run_cleanup():
    """Run the cleanup script to reduce project size."""
    print("🧹 Running project cleanup...")
    os.system("python scripts/cleanup.py")

def run_test():
    """Test the installation and optimization features."""
    print("🧪 Testing optimized installation...")
    os.system("python scripts/test_installation.py")

def run_demo():
    """Run the interactive demo."""
    print("🎮 Starting interactive demo...")
    os.system("python scripts/demo.py")

def run_train(dataset_path, **kwargs):
    """Run training with optimized settings."""
    print(f"🚀 Starting optimized training on: {dataset_path}")
    
    # Build command with optimized defaults
    cmd = f"python scripts/main.py train --dataset-path {dataset_path}"
    
    # Add optimization flags
    config = load_config()
    if config:
        training_config = config.get('training', {})
        model_config = config.get('model', {})
        
        cmd += f" --batch-size {training_config.get('batch_size', 32)}"
        cmd += f" --ngf {model_config.get('ngf', 48)}"
        cmd += f" --ndf {model_config.get('ndf', 48)}"
        cmd += f" --save-interval {training_config.get('save_interval', 20)}"
        cmd += f" --sample-interval {training_config.get('sample_interval', 10)}"
    
    # Add any additional kwargs
    for key, value in kwargs.items():
        if value is not None:
            cmd += f" --{key.replace('_', '-')} {value}"
    
    print(f"Command: {cmd}")
    os.system(cmd)

def run_generate(num_samples=64, **kwargs):
    """Generate samples with optimized settings."""
    print(f"🎨 Generating {num_samples} optimized samples...")
    
    # Find the latest checkpoint
    checkpoint_dir = "outputs/checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            
            cmd = f"python scripts/main.py generate --checkpoint {checkpoint_path} --num-samples {num_samples}"
            
            # Add optimization flags
            config = load_config()
            if config:
                model_config = config.get('model', {})
                cmd += f" --ngf {model_config.get('ngf', 48)}"
            
            print(f"Command: {cmd}")
            os.system(cmd)
        else:
            print("❌ No checkpoints found. Please train a model first.")
    else:
        print("❌ Checkpoints directory not found. Please train a model first.")

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Optimized DCGAN Face Generator - Reduced size with maintained performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py train data/faces                    # Train on face dataset
  python run.py train data/faces --epochs 200       # Train for 200 epochs
  python run.py generate --num-samples 100          # Generate 100 samples
  python run.py cleanup                             # Reduce project size
  python run.py demo                                # Interactive demo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    subparsers.add_parser('info', help='Show project information and optimization details')
    
    # Test command
    subparsers.add_parser('test', help='Test installation and optimization features')
    
    # Demo command
    subparsers.add_parser('demo', help='Run interactive demo')
    
    # Cleanup command
    subparsers.add_parser('cleanup', help='Reduce project size through cleanup')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train optimized model')
    train_parser.add_argument('dataset_path', help='Path to dataset directory')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size for training')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate samples')
    gen_parser.add_argument('--num-samples', type=int, default=64, 
                           help='Number of samples to generate')
    
    args = parser.parse_args()
    
    if args.command == 'info':
        show_info()
    elif args.command == 'test':
        run_test()
    elif args.command == 'demo':
        run_demo()
    elif args.command == 'cleanup':
        run_cleanup()
    elif args.command == 'train':
        run_train(args.dataset_path, 
                 epochs=args.epochs,
                 batch_size=args.batch_size,
                 learning_rate=args.learning_rate)
    elif args.command == 'generate':
        run_generate(num_samples=args.num_samples)
    else:
        show_info()

if __name__ == '__main__':
    main() 