# Installation Guide

## Prerequisites
- Python 3.8 or higher
- PyTorch 1.9.0 or higher
- CUDA (optional, for GPU acceleration)

## Quick Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dcgan-face-generator.git
   cd dcgan-face-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test installation**
   ```bash
   python run.py test
   ```

## Detailed Installation

### GPU Support (Optional)
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

## Verification
Run the test script to verify everything is working:
```bash
python run.py test
```
