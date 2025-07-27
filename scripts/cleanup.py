#!/usr/bin/env python3
"""
Cleanup script to reduce project size while maintaining functionality.
Removes old checkpoints, compresses images, and optimizes storage.
"""

import os
import glob
import shutil
import json
from PIL import Image
import argparse

def compress_image_file(file_path, quality=85):
    """
    Compress an image file to reduce size.
    Args:
        file_path: Path to the image file
        quality: JPEG quality (1-100)
    """
    try:
        with Image.open(file_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save with compression
            img.save(file_path, 'JPEG', quality=quality, optimize=True)
            print(f"Compressed: {file_path}")
    except Exception as e:
        print(f"Failed to compress {file_path}: {e}")

def cleanup_checkpoints(checkpoint_dir, max_checkpoints=3):
    """
    Remove old checkpoints keeping only the most recent ones.
    Args:
        checkpoint_dir: Directory containing checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Remove old checkpoints
    if len(checkpoint_files) > max_checkpoints:
        files_to_remove = checkpoint_files[:-max_checkpoints]
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                print(f"Removed old checkpoint: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")

def cleanup_samples(samples_dir, max_samples=20):
    """
    Remove old sample images keeping only the most recent ones.
    Args:
        samples_dir: Directory containing sample images
        max_samples: Maximum number of sample files to keep
    """
    if not os.path.exists(samples_dir):
        return
    
    # Find all sample files
    sample_files = glob.glob(os.path.join(samples_dir, 'samples_*.png'))
    sample_files.extend(glob.glob(os.path.join(samples_dir, 'samples_*.jpg')))
    sample_files.sort(key=os.path.getmtime, reverse=True)
    
    # Remove old samples
    if len(sample_files) > max_samples:
        files_to_remove = sample_files[max_samples:]
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                print(f"Removed old sample: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")

def compress_existing_images(directory, quality=85):
    """
    Compress all images in a directory.
    Args:
        directory: Directory containing images
        quality: JPEG quality for compression
    """
    if not os.path.exists(directory):
        return
    
    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    
    for file_path in image_files:
        compress_image_file(file_path, quality)

def remove_cache_directories():
    """
    Remove cache directories to free up space.
    """
    cache_dirs = ['__pycache__', '.pytest_cache', '.mypy_cache']
    
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name in cache_dirs:
                cache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(cache_path)
                    print(f"Removed cache directory: {cache_path}")
                except Exception as e:
                    print(f"Failed to remove {cache_path}: {e}")

def cleanup_logs(logs_dir, max_log_size_mb=100):
    """
    Clean up log files that are too large.
    Args:
        logs_dir: Directory containing logs
        max_log_size_mb: Maximum log file size in MB
    """
    if not os.path.exists(logs_dir):
        return
    
    max_size_bytes = max_log_size_mb * 1024 * 1024
    
    for root, dirs, files in os.walk(logs_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                if os.path.getsize(file_path) > max_size_bytes:
                    # Truncate large log files
                    with open(file_path, 'w') as f:
                        f.write(f"# Log file truncated due to size limit ({max_log_size_mb}MB)\n")
                    print(f"Truncated large log file: {file_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

def get_directory_size(directory):
    """
    Calculate the total size of a directory.
    Args:
        directory: Directory path
    Returns:
        Size in bytes
    """
    total_size = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
            except OSError:
                pass
    return total_size

def print_size_summary():
    """
    Print a summary of directory sizes.
    """
    directories = ['outputs', 'logs', 'data', 'src', 'scripts', 'configs']
    
    print("\n=== Directory Size Summary ===")
    total_size = 0
    
    for directory in directories:
        if os.path.exists(directory):
            size = get_directory_size(directory)
            size_mb = size / (1024 * 1024)
            total_size += size
            print(f"{directory:12}: {size_mb:8.2f} MB")
    
    total_mb = total_size / (1024 * 1024)
    print(f"{'TOTAL':12}: {total_mb:8.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='Cleanup and optimize DCGAN project size')
    parser.add_argument('--max-checkpoints', type=int, default=3, 
                       help='Maximum number of checkpoints to keep')
    parser.add_argument('--max-samples', type=int, default=20,
                       help='Maximum number of sample images to keep')
    parser.add_argument('--image-quality', type=int, default=85,
                       help='JPEG quality for image compression (1-100)')
    parser.add_argument('--max-log-size', type=int, default=100,
                       help='Maximum log file size in MB')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    print("=== DCGAN Project Cleanup ===")
    print(f"Max checkpoints: {args.max_checkpoints}")
    print(f"Max samples: {args.max_samples}")
    print(f"Image quality: {args.image_quality}")
    print(f"Max log size: {args.max_log_size}MB")
    print(f"Dry run: {args.dry_run}")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print("No files will be modified.")
    
    # Print initial size summary
    print("\n=== Initial Size Summary ===")
    print_size_summary()
    
    if not args.dry_run:
        # Cleanup operations
        print("\n=== Starting Cleanup ===")
        
        # Cleanup checkpoints
        cleanup_checkpoints('outputs/checkpoints', args.max_checkpoints)
        
        # Cleanup samples
        cleanup_samples('outputs/samples', args.max_samples)
        
        # Compress existing images
        print("\n=== Compressing Images ===")
        compress_existing_images('outputs/samples', args.image_quality)
        compress_existing_images('outputs/generated_samples', args.image_quality)
        
        # Remove cache directories
        print("\n=== Removing Cache Directories ===")
        remove_cache_directories()
        
        # Cleanup logs
        print("\n=== Cleaning Logs ===")
        cleanup_logs('logs', args.max_log_size)
        
        print("\n=== Cleanup Complete ===")
    
    # Print final size summary
    print("\n=== Final Size Summary ===")
    print_size_summary()

if __name__ == '__main__':
    main() 