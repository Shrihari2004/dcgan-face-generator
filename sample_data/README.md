# Sample Data Directory

This directory contains sample data for the DCGAN Face Generator.

## Structure
```
sample_data/
├── sample_config.json    # Sample configuration file
└── README.md            # This file
```

## Usage
1. Extract your training data to this directory
2. Update the configuration in `sample_config.json`
3. Run training: `python run.py train sample_data/`

## Data Format
- Images should be in common formats (JPG, PNG, etc.)
- Recommended size: 64x64 pixels
- RGB format preferred
