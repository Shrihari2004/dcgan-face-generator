# Usage Guide

## Quick Start

1. **Prepare your data**
   ```bash
   # Extract your training data
   unzip training_data.zip -d data/
   ```

2. **Train the model**
   ```bash
   python run.py train data/faces
   ```

3. **Generate samples**
   ```bash
   python run.py generate --num-samples 64
   ```

## Advanced Usage

### Custom Configuration
Edit `configs/config.json` to customize training parameters:
```json
{
    "training": {
        "epochs": 200,
        "batch_size": 32,
        "learning_rate": 0.0002
    }
}
```

### Training Options
```bash
# Train with custom parameters
python run.py train data/faces --epochs 200 --batch-size 16

# Resume training from checkpoint
python run.py train data/faces --resume outputs/checkpoints/checkpoint_epoch_50.pth
```

### Sample Generation
```bash
# Generate different numbers of samples
python run.py generate --num-samples 100

# Use specific checkpoint
python run.py generate --checkpoint outputs/checkpoints/checkpoint_final.pth
```

## Optimization Features

### Automatic Cleanup
```bash
# Clean up old files
python run.py cleanup

# Monitor project size
python scripts/cleanup.py --dry-run
```

### Size Optimization
The project includes automatic optimization features:
- Model compression
- Image compression
- Checkpoint management
- Cache cleanup
