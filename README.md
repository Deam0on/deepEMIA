# Deep-Learning (primarily Electron Microscopy) Image Analysis Tool

A modular, end-to-end computer vision pipeline for scientific image analysis, featuring dataset management, model training, evaluation, inference, and a Streamlit-based web interface. The project is designed for extensibility, reproducibility, security, and integration with Google Cloud Storage.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Security Features](#security-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Interactive CLI Wizard](#interactive-cli-wizard)
  - [Command Line Interface](#command-line-interface)
  - [Web Interface](#web-interface)
- [First-Time Setup](#first-time-setup)
- [Security Considerations](#security-considerations)
- [Error Handling and Reliability](#error-handling-and-reliability)
- [License](#license)
- [Support](#support)

## Overview

This repository provides a robust pipeline for computer vision tasks, including:

- Dataset preparation and splitting
- Model training and quantization (Detectron2, PyTorch)
- Model evaluation and metrics reporting
- Inference on new images with advanced measurement and analysis
- Interactive CLI wizard for easy operation
- Web-based interface for streamlined workflow management
- Integration with Google Cloud Storage for scalable data management

The pipeline is suitable for scientific and industrial applications requiring reproducible, automated image analysis with enterprise-grade security features.

## Features

- **Dataset Management**: Prepare, split, and register datasets in custom or COCO format. Supports per-image JSON and COCO-style annotations.
- **Model Training**: Train instance segmentation models with Detectron2, including CPU-optimized and quantized models for efficient inference.
- **Hyperparameter Optimization**: Automated hyperparameter tuning using Optuna with configuration persistence.
- **Evaluation**: COCO-style evaluation with metrics export and optional visualization.
- **Interactive CLI Wizard**: User-friendly command-line interface with guided workflows.
- **Advanced Measurements**: Geometric analysis with optional contrast distribution measurements for particle analysis.
- **Security Features**: Secure password handling, path validation, and input sanitization.
- **Error Handling**: Comprehensive error handling with retry logic and detailed logging.

## Security Features

### Authentication
- **Secure Password Handling**: Uses environment variable hashes instead of plaintext passwords
- **Environment Variable**: Set `ADMIN_PASSWORD_HASH` environment variable with SHA256 hash of your admin password

### File Security
- **Path Validation**: All file operations validate paths to prevent directory traversal attacks
- **Safe File Operations**: Utility functions for secure file manipulation within allowed directories
- **Configuration Validation**: Schema validation for configuration files

### Network Security
- **Retry Logic**: Robust network operations with exponential backoff
- **Timeout Handling**: Configurable timeouts for all network operations
- **Error Recovery**: Graceful handling of network failures

## Installation

### Prerequisites
- Python 3.8 or higher
- Google Cloud SDK (for GCS integration)
- CUDA-compatible GPU (optional, for training acceleration)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/deepEMIA.git
   cd deepEMIA
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up security:
   ```bash
   # Generate admin password hash (replace 'your_secure_password' with your actual password)
   export ADMIN_PASSWORD_HASH=$(echo -n 'your_secure_password' | sha256sum | cut -d' ' -f1)
   ```

4. Run initial setup:
   ```bash
   python cli_main.py  # Use interactive wizard
   # or
   python main.py --task setup --dataset_name dummy
   ```

## Configuration

### Security Configuration
The project uses secure configuration management:

- **Password Security**: Admin passwords are stored as environment variable hashes
- **Path Validation**: All file paths are validated against allowed directories
- **Configuration Schema**: YAML configuration is validated against a defined schema

### Configuration File
Edit `config/config.yaml` for your environment:

```yaml
bucket: your-gcs-bucket-name
paths:
  main_script: "~/deepEMIA/main.py"
  split_dir: "~/split_dir"
  category_json: "~/deepEMIA/dataset_info.json"
  # ... other paths
measure_contrast_distribution: false  # Enable for particle analysis
rcnn_hyperparameters:
  default:
    R50:
      base_lr: 0.00025
      # ... other hyperparameters
```

## Usage

### Interactive CLI Wizard

The easiest way to use the pipeline is through the interactive CLI wizard:

```bash
python cli_main.py
```

This provides a guided interface for all operations:
- Dataset selection from available datasets
- Interactive parameter configuration  
- Error recovery and retry logic
- Progress tracking and ETA estimation

### Command Line Interface

For automated workflows, use the direct CLI:

```bash
# First-time setup
python main.py --task setup

# Prepare dataset
python main.py --task prepare --dataset_name polyhipes

# Train model with hyperparameter optimization
python main.py --task train --dataset_name polyhipes --rcnn combo --optimize --n-trials 10 --augment

# Evaluate model with visualizations
python main.py --task evaluate --dataset_name polyhipes --visualize --rcnn combo

# Run inference with custom threshold and multi-pass deduplication
python main.py --task inference --dataset_name polyhipes --threshold 0.7 --visualize --pass multi 5 --id
```

#### Available CLI Arguments

**Core Arguments:**

- `--task` *(required)*: Task to perform (`prepare`, `train`, `evaluate`, `inference`, `setup`)
- `--dataset_name` *(required)*: Name of the dataset (e.g., `polyhipes`)

**Model Configuration:**

- `--rcnn`: RCNN backbone (`50`, `101`, `combo`) - default: `101`
- `--threshold`: Inference confidence threshold - default: `0.65`
- `--dataset_format`: Dataset format (`json`, `coco`) - default: `json`

**Training Options:**

- `--augment`: Enable data augmentation during training
- `--optimize`: Run Optuna hyperparameter optimization
- `--n-trials`: Number of Optuna trials - default: `10`

**Visualization & Output:**

- `--visualize`: Save visualizations of predictions during evaluation/inference
- `--id`: Draw instance IDs on inference overlays

**Data Transfer:**

- `--download`: Download data from GCS before task - default: `True`
- `--upload`: Upload results to GCS after task - default: `True`

**Inference Modes:**

- `--pass`: Inference mode (`single` or `multi [max_iters]`) - default: `single`
  - `single`: One inference pass per image
  - `multi N`: Iterative deduplication up to N iterations (default: 10)

For complete usage information and examples:

```bash
python main.py --help
```

### Advanced Usage Patterns

#### Automated Batch Processing

```bash
# Process multiple datasets in sequence
for dataset in dataset1 dataset2 dataset3; do
    python main.py --task inference --dataset_name $dataset --threshold 0.7 --visualize
done
```

#### Hyperparameter Optimization Workflow

```bash
# 1. Prepare dataset
python main.py --task prepare --dataset_name polyhipes

# 2. Train with optimization (finds best parameters)
python main.py --task train --dataset_name polyhipes --optimize --n-trials 50 --augment

# 3. Evaluate optimized model
python main.py --task evaluate --dataset_name polyhipes --visualize

# 4. Run inference with optimized model
python main.py --task inference --dataset_name polyhipes --threshold 0.65 --pass multi 10
```

#### Performance Tuning

```bash
# For speed (small particles, quick results)
python main.py --task inference --dataset_name dataset --rcnn 50 --threshold 0.7 --pass single

# For accuracy (complex particles, research quality)
python main.py --task inference --dataset_name dataset --rcnn combo --threshold 0.5 --pass multi 15 --visualize
```

### Dataset Format Requirements

**Input Structure:**
```
DATASET/
├── your_dataset_name/
│   ├── image1.tif
│   ├── image1.json        # Annotation file (for training)
│   ├── image2.tif
│   ├── image2.json
│   └── ...
└── INFERENCE/             # For inference-only images
    ├── test_image1.tif
    ├── test_image2.tif
    └── ...
```

**Annotation Format (JSON):**
```json
{
  "imagePath": "image1.tif",
  "imageHeight": 1024,
  "imageWidth": 1024,
  "shapes": [
    {
      "label": "particle",
      "points": [[x1, y1], [x2, y2], ...],  // Polygon points
      "shape_type": "polygon"
    }
  ]
}
```

### Output Files

**Training Output:**
- `~/split_dir/`: Dataset splits and model files
- `~/logs/`: Training logs and metrics
- Model weights: `model_final.pth`, `model_quantized.pth`

**Inference Output:**
- `results.csv`: Measurements and particle properties
- `*_overlay.png`: Visualization images with detected particles
- `*_measurements.csv`: Per-image detailed measurements

**Measurement Columns:**
- `Area_um2`: Particle area in square micrometers
- `Perimeter_um`: Particle perimeter in micrometers
- `Major_Axis_um`: Length of major axis
- `Minor_Axis_um`: Length of minor axis
- `Aspect_Ratio`: Ratio of major to minor axis
- `Circularity`: Shape circularity measure (0-1)
- `Wavelength_nm`: Estimated wavelength (if color analysis enabled)
- `Confidence`: Detection confidence score
- `Image_Name`: Source image filename

### Web Interface

Launch the Streamlit web interface:

```bash
streamlit run gui/streamlit_gui.py
```

## Security Considerations

### Environment Variables
Set these environment variables for secure operation:

```bash
export ADMIN_PASSWORD_HASH="your_sha256_hash_here"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gcs-key.json"
```

### File Permissions
Ensure proper file permissions:
- Configuration files: 600 (owner read/write only)
- Log directories: 750 (owner full, group read/execute)
- Temporary directories: 700 (owner only)

### Network Security
- Use VPC networks for GCS access when possible
- Enable GCS audit logging
- Use service accounts with minimal required permissions

## Error Handling and Reliability

The pipeline includes comprehensive error handling:

- **Retry Logic**: Automatic retry with exponential backoff for network operations
- **Graceful Degradation**: Fallback options when services are unavailable
- **Detailed Logging**: Comprehensive logging for debugging and monitoring
- **Resource Management**: Proper cleanup of processes and file handles
- **Validation**: Input validation and sanitization throughout the pipeline

### Development Setup
1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   # Optional: pip install cerberus  # For enhanced configuration validation
   ```

2. Run security checks:
   ```bash
   # Validate configuration
   python -c "from src.utils.config import get_config; print('Config valid')"
   ```

3. Follow security best practices:
   - Never commit passwords or API keys
   - Use environment variables for sensitive data
   - Validate all user inputs
   - Test error handling paths

## License

Copyright (c) 2025 Filip Hládek. All rights reserved.

Unauthorized copying, distribution, modification, or commercial use of this software is strictly prohibited without express permission from the copyright holder.

## First-Time Setup

Before running any training or inference tasks, you should configure your environment using the setup task. This will prompt you to enter your Google Cloud Storage bucket name and scale bar ROI settings, and will save them to `config/config.yaml`.

### Quick Setup

```bash
# Interactive setup wizard (recommended)
python cli_main.py
# Select "setup" from the menu

# Or direct setup
python main.py --task setup --dataset_name dummy
```

### Configuration Details

The setup process will configure:

- **Google Cloud Storage bucket**: Your GCS bucket for data storage and model sharing
- **Scale bar ROI settings**: Region of interest for automatic scale bar detection in SEM images
- **Scalebar thresholds**: Intensity and proximity thresholds for scale bar detection
- **Measurement settings**: Whether to measure contrast distribution for particle analysis
- **RCNN hyperparameters**: Default training parameters for R50 and R101 backbones

Example configuration file structure:

```yaml
bucket: your-bucket-name
paths:
  main_script: "~/deepEMIA/main.py"
  split_dir: "~/split_dir"
  category_json: "~/deepEMIA/dataset_info.json"
  eta_file: "~/deepEMIA/config/eta_data.json"
  logs_dir: "~/logs"
  output_dir: "~/deepEMIA/output"
  local_dataset_root: "~"
scale_bar_rois:
  default:
    x_start_factor: 0.667    # Start position (fraction of image width)
    y_start_factor: 0.866    # Start position (fraction of image height)
    width_factor: 1          # ROI width (fraction of image width)
    height_factor: 0.067     # ROI height (fraction of image height)
scalebar_thresholds:
  intensity: 150             # Intensity threshold for scale bar detection
  proximity: 50              # Proximity threshold for grouping scale bar elements
measure_contrast_distribution: false  # Enable for detailed particle contrast analysis
rcnn_hyperparameters:
  default:
    R50:
      base_lr: 0.00025
      ims_per_batch: 2
      warmup_iters: 1000
      gamma: 0.1
      batch_size_per_image: 64
    R101:
      base_lr: 0.00025
      ims_per_batch: 2
      warmup_iters: 1000
      gamma: 0.1
      batch_size_per_image: 64
  best:                      # Reserved for hyperparameter optimization results
    R50: {}
    R101: {}
```

## Troubleshooting

### Common Issues

1. **"Configuration not found" error**
   - Run the setup task first: `python cli_main.py` and select "setup"

2. **"Failed to download dataset_info.json" error**
   - Ensure Google Cloud SDK is installed and configured
   - Check your bucket name in the configuration
   - Verify you have read permissions on the GCS bucket

3. **Import errors or missing dependencies**
   - Install requirements: `pip install -r requirements.txt`
   - For CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

4. **Training fails with CUDA out of memory**
   - Reduce batch size in hyperparameters
   - Use R50 backbone instead of R101
   - Enable CPU training (automatic fallback)

5. **Scale bar detection fails**
   - Adjust scale_bar_rois parameters in config
   - Check intensity and proximity thresholds
   - Ensure images have visible scale bars in the expected region

### Log Files

All operations are logged to `~/logs/` with timestamps:
- `system_YYYY-MM-DD_HH-MM-SS.log`: Detailed system logs
- Console output: Simplified progress information

## Support

For support and troubleshooting:

1. **Check the logs** in `~/logs/` for detailed error information
2. **Verify configuration** by running setup again
3. **Test with sample data** to isolate issues
4. **Check system requirements** (Python 3.8+, sufficient RAM/disk space)
5. **Review Google Cloud permissions** for your service account
6. **Open an issue** on GitHub with:
   - Error messages from logs
   - Configuration file (remove sensitive info)
   - System information (OS, Python version, GPU)
   - Steps to reproduce the issue
