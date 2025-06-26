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
- [Contributing](#contributing)
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
   git clone https://github.com/your-username/uw-com-vision.git
   cd uw-com-vision
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

## Contributing

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please:

1. Check the logs in `~/logs/` for error details
2. Verify your configuration with the schema validator
3. Ensure all environment variables are set correctly
4. Open an issue on GitHub with relevant log excerpts

```yaml
bucket: your-bucket-name
paths:
  main_script: "~/deepEMIA/main.py"
  split_dir: "~/split_dir"
  category_json: "~/deepEMIA/dataset_info.json"
  eta_file: "~/deepEMIA/config/eta_data.json"
  local_dataset_root: "~"
scale_bar_rois:
  default:
    x_start_factor: 0.667
    y_start_factor: 0.866
    width_factor: 1
    height_factor: 0.067
```

## First-Time Setup

Before running any training or inference tasks, you should configure your environment using the setup task. This will prompt you to enter your Google Cloud Storage bucket name and scale bar ROI settings, and will save them to `config/config.yaml`.

To run the setup:

```sh
python main.py --task setup
```

You will be prompted for:

- **Google Cloud Storage bucket name**: The name of your GCS bucket for data storage.
- **Scale bar ROI settings**: The region of interest for scale bar detection. You can press Enter to accept the defaults.

If a configuration file already exists, you will be asked whether you want to overwrite it.

After setup, you can proceed to use the other tasks (`prepare`, `train`, `evaluate`, `inference`) as described in the Command Line Interface section above.

1. Fork the repository
2. Create a feature branch
3. Commit your changes with clear messages
4. Push to your fork
5. Open a Pull Request describing your changes

Please review the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.
