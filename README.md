# DCGAN Face Generation

A comprehensive implementation of Deep Convolutional Generative Adversarial Networks (DCGAN) for generating realistic face images using PyTorch.

## Features

- **DCGAN Architecture**: Implements the original DCGAN paper architecture
- **Face Generation**: Specialized for generating realistic human faces
- **Comprehensive Training**: Full training pipeline with checkpoint saving and resuming
- **Visualization**: Real-time loss curves, sample generation, and latent space exploration
- **GPU Support**: Automatic GPU detection and utilization
- **Flexible Dataset**: Support for custom datasets and CelebA
- **Evaluation Metrics**: Built-in model evaluation and FID score calculation

## Project Structure

```
Market analyser/
├── src/                    # Core source code
│   ├── models.py          # Generator and Discriminator architectures
│   ├── dataset.py         # Dataset loading and preprocessing
│   ├── trainer.py         # Training loop and utilities
│   └── utils.py           # Utility functions and helpers
├── scripts/               # Executable scripts
│   ├── main.py           # Main training script
│   ├── demo.py           # Interactive demo
│   ├── generate_samples.py # Sample generation
│   └── run_gan.py        # Quick runner
├── configs/               # Configuration files
│   └── config.json       # Training parameters
├── outputs/               # Model outputs
│   ├── checkpoints/      # Trained models
│   ├── samples/          # Training progress
│   └── generated_samples/ # Generated images
├── logs/                  # Training logs
│   └── gan_training/     # TensorBoard logs
├── data/                  # Training datasets
├── run.py                # Main entry point
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Market-analyser
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
   ```

## Quick Start

### 1. Prepare Your Dataset

Place your face images in a directory structure like this:
```
data/
└── faces/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 2. Train the Model (Recommended)

**Use the new organized interface:**

```bash
# Basic training (recommended)
python run.py train data/faces

# Custom training with config
python run.py train data/faces --epochs 200 --batch-size 32

# Run demo first to test everything
python run.py demo
```

**Or use the legacy scripts:**

```bash
# Basic training
python scripts/main.py train --dataset-path data/faces --epochs 100

# Advanced training with custom parameters
python scripts/main.py train \
    --dataset-path data/faces \
    --epochs 200 \
    --batch-size 32 \
    --image-size 128 \
    --learning-rate 0.0001 \
    --save-interval 5 \
    --sample-interval 2
```

### 3. Generate Samples

```bash
# Generate samples from trained model
python run.py generate --num-samples 64

# Generate with custom checkpoint
python run.py generate \
    --checkpoint outputs/checkpoints/checkpoint_epoch_10.pth \
    --num-samples 100
```

## Usage Examples

### Training with Different Configurations

```bash
# Small dataset, quick training
python main.py train \
    --dataset-path data/small_faces \
    --epochs 50 \
    --batch-size 16 \
    --image-size 64

# Large dataset, high quality
python main.py train \
    --dataset-path data/large_faces \
    --epochs 500 \
    --batch-size 128 \
    --image-size 128 \
    --ngf 128 \
    --ndf 128

# Resume training from checkpoint
python main.py train \
    --dataset-path data/faces \
    --epochs 100 \
    --resume checkpoints/checkpoint_epoch_50.pth
```

### Using CelebA Dataset

```bash
# Download and use CelebA dataset
python -c "from utils import download_celeba_dataset; download_celeba_dataset()"

# Train on CelebA
python main.py train \
    --dataset-path data/celeba \
    --dataset-type celeba \
    --epochs 100
```

## Model Architecture

### Generator
- **Input**: Random noise vector (100-dimensional)
- **Architecture**: Linear → Reshape → 4 Transposed Convolution layers
- **Output**: 64×64×3 RGB image
- **Activation**: ReLU + BatchNorm (hidden), Tanh (output)

### Discriminator
- **Input**: 64×64×3 RGB image
- **Architecture**: 4 Convolution layers → Linear
- **Output**: Single probability value
- **Activation**: LeakyReLU + BatchNorm (hidden), Sigmoid (output)

## Training Process

1. **Data Preprocessing**: Images are resized, normalized to [-1, 1]
2. **Training Loop**: Alternating generator and discriminator updates
3. **Loss Functions**: Binary Cross-Entropy with Logits
4. **Optimization**: Adam optimizer with β₁=0.5, β₂=0.999
5. **Monitoring**: Real-time loss curves and sample generation

## Output Files

During training, the following files are generated:

- `checkpoints/`: Model checkpoints for resuming training
- `samples/`: Generated image samples during training
- `runs/gan_training/`: TensorBoard logs
- `training_history.png`: Training loss curves
- `interpolation.png`: Latent space interpolation
- `latent_space.png`: 2D latent space visualization

## Evaluation Metrics

The model provides several evaluation metrics:

- **Generator Loss**: Measures how well the generator fools the discriminator
- **Discriminator Loss**: Measures discriminator's ability to classify real vs fake
- **Real/Fake Scores**: Average discriminator scores for real and generated images
- **FID Score**: Fréchet Inception Distance (requires additional setup)

## Advanced Features

### Latent Space Exploration

```python
from utils import visualize_latent_space
from models import Generator

# Load trained generator
generator = Generator()
generator.load_state_dict(torch.load('checkpoint.pth')['generator_state_dict'])

# Visualize latent space
visualize_latent_space(generator, save_path='latent_space.png')
```

### Model Evaluation

```python
from utils import evaluate_model

# Evaluate trained model
metrics = evaluate_model(generator, discriminator, dataloader)
print(f"Total Accuracy: {metrics['total_accuracy']:.4f}")
```

### Custom Dataset Creation

```python
from utils import create_synthetic_dataset

# Create synthetic dataset for testing
create_synthetic_dataset(save_dir='data/synthetic', num_images=1000)
```

## Configuration

Create a custom configuration file:

```python
from utils import create_training_config

# Create default config
create_training_config('my_config.json')
```

## Troubleshooting

### Common Issues

1. **OpenMP Runtime Error**:
   ```
   OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
   ```
   **Solution**: Use the quick runner script:
   ```bash
   python run_gan.py train data/faces
   ```
   Or set environment variable manually:
   ```bash
   set KMP_DUPLICATE_LIB_OK=TRUE  # Windows
   export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/Mac
   ```

2. **Pin Memory Warning**:
   ```
   'pin_memory' argument is set as true but no accelerator is found
   ```
   **Solution**: This is now automatically fixed. The warning should not appear.

3. **CUDA Out of Memory**:
   - Reduce batch size: `--batch-size 16`
   - Reduce image size: `--image-size 64`
   - Use CPU: Set `device='cpu'` in trainer

4. **Training Instability**:
   - Adjust learning rate: `--learning-rate 0.0001`
   - Increase discriminator training: Modify trainer code
   - Use gradient clipping: Add to optimizer

5. **Poor Quality Results**:
   - Increase training epochs: `--epochs 500`
   - Use larger model: `--ngf 128 --ndf 128`
   - Improve dataset quality and size

### Performance Tips

- **GPU Memory**: Use mixed precision training for large models
- **Data Loading**: Increase `num_workers` for faster data loading
- **Checkpointing**: Save checkpoints frequently to avoid losing progress
- **Monitoring**: Use TensorBoard for real-time training monitoring

## Requirements

- **Python**: 3.7+
- **PyTorch**: 1.9.0+
- **CUDA**: 10.2+ (optional, for GPU acceleration)
- **Memory**: 8GB+ RAM, 4GB+ GPU memory recommended

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Original DCGAN paper: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
- PyTorch team for the excellent deep learning framework
- CelebA dataset creators for the face dataset

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{dcgan-face-generation,
  title={DCGAN Face Generation Implementation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/dcgan-face-generation}
}
``` 