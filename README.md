# deepEMIA# deepEMIA



Deep Learning Electron Microscopy Image Analysis Tool - A complete pipeline for scientific image segmentation and analysis.Deep Learning Electron Microscopy Image Analysis Tool - A complete pipeline for scientific image segmentation and analysis.



## Overview## Overview



deepEMIA provides an end-to-end computer vision pipeline for electron microscopy and scientific image analysis, featuring:deepEMIA provides an end-to-end computer vision pipeline for electron microscopy and scientific image analysis, featuring:



- Dataset management and preparation- Dataset management and preparation

- Model training with hyperparameter optimization- Model training with hyperparameter optimization

- COCO-metrics evaluation- COCO-metrics evaluation

- Automated inference with measurement extraction- Automated inference with measurement extraction

- Interactive CLI wizard and web interface- Interactive CLI wizard and web interface

- Google Cloud Storage integration- Google Cloud Storage integration



## Quick Start## Quick Start



### Installation### Installation



```bash```bash

git clone https://github.com/Deam0on/deepEMIA.gitgit clone https://github.com/Deam0on/deepEMIA.git

cd deepEMIAcd deepEMIA

pip install -r requirements.txtpip install -r requirements.txt

``````



### Setup### Setup



```bash```bash

python cli_main.py  # Interactive setup wizardpython cli_main.py  # Interactive setup wizard

``````



Or direct command:Or direct command:



```bash```bash

python main.py --task setup --dataset_name dummypython main.py --task setup --dataset_name dummy

``````



### Basic Usage### Basic Usage



```bash```bash

# Prepare dataset# Prepare dataset

python main.py --task prepare --dataset_name my_datasetpython main.py --task prepare --dataset_name my_dataset



# Train model with optimization# Train model with optimization

python main.py --task train --dataset_name my_dataset --optimize --n-trials 20 --augmentpython main.py --task train --dataset_name my_dataset --optimize --n-trials 20 --augment



# Run inference# Run inference

python main.py --task inference --dataset_name my_dataset --threshold 0.65 --visualizepython main.py --task inference --dataset_name my_dataset --threshold 0.65 --visualize

``````



### Web Interface### Web Interface



```bash```bash

streamlit run gui/streamlit_gui.pystreamlit run gui/streamlit_gui.py

``````



## Key Features## Key Features



- **Automated Pipeline**: Complete workflow from raw data to measurements- **Automated Pipeline**: Complete workflow from raw data to measurements

- **Hyperparameter Optimization**: Automatic tuning with Optuna- **Hyperparameter Optimization**: Automatic tuning with Optuna

- **Multi-Scale Inference**: Iterative detection with deduplication- **Multi-Scale Inference**: Iterative detection with deduplication

- **Scale Bar Detection**: Automatic unit conversion from scale bars- **Scale Bar Detection**: Automatic unit conversion from scale bars

- **Spatial Constraints**: Rule-based filtering (containment, overlap)- **Spatial Constraints**: Rule-based filtering (containment, overlap)

- **Cloud Integration**: Seamless GCS sync and archival- **Cloud Integration**: Seamless GCS sync and archival

- **Flexible Interfaces**: CLI wizard, direct CLI, and web GUI- **Flexible Interfaces**: CLI wizard, direct CLI, and web GUI



## Documentation## Documentation



Comprehensive documentation is available in the `docs/` directory:Comprehensive documentation is available in the `docs/` directory:



- [Getting Started](docs/getting-started.md) - Installation and setup- [Getting Started](docs/getting-started.md) - Installation and setup

- [User Guide](docs/user-guide.md) - Usage and workflows- [User Guide](docs/user-guide.md) - Usage and workflows

- [Configuration Reference](docs/configuration.md) - Config file details- [Configuration Reference](docs/configuration.md) - Config file details

- [API Reference](docs/api/overview.md) - Technical API docs- [API Reference](docs/api/overview.md) - Technical API docs

- [Architecture](docs/architecture/pipeline-overview.md) - System design- [Architecture](docs/architecture/pipeline-overview.md) - System design

- [Examples](docs/examples/basic-workflow.md) - Practical examples- [Examples](docs/examples/basic-workflow.md) - Practical examples

- [Contributing](docs/development/contributing.md) - Development guide- [Contributing](docs/development/contributing.md) - Development guide



## Requirements## Requirements



- Python 3.8+- Python 3.8+

- PyTorch with CUDA support (optional, for GPU acceleration)- PyTorch with CUDA support (optional, for GPU acceleration)

- Google Cloud SDK (for GCS integration)- Google Cloud SDK (for GCS integration)

- 16GB RAM minimum (32GB recommended for training)- 16GB RAM minimum (32GB recommended for training)



## Technology Stack## Technology Stack



- **Detectron2**: Instance segmentation- **Detectron2**: Instance segmentation

- **PyTorch**: Deep learning framework- **PyTorch**: Deep learning framework

- **Optuna**: Hyperparameter optimization- **Optuna**: Hyperparameter optimization

- **OpenCV/scikit-image**: Image processing- **OpenCV/scikit-image**: Image processing

- **Streamlit**: Web interface- **Streamlit**: Web interface

- **Google Cloud Storage**: Data management- **Google Cloud Storage**: Data management



## Output## Output



Inference generates:Inference generates:



- CSV files with geometric measurements (area, perimeter, axes, circularity)- CSV files with geometric measurements (area, perimeter, axes, circularity)

- Wavelength analysis from particle colors- Wavelength analysis from particle colors

- Visualization images with detected particles- Visualization images with detected particles

- Per-image and aggregate statistics- Per-image and aggregate statistics



## Common Commands## Common Commands



```bash```bash

# Interactive wizard (recommended for beginners)# Interactive wizard (recommended for beginners)

python cli_main.pypython cli_main.py



# Train with augmentation# Train with augmentation

python main.py --task train --dataset_name mydata --augmentpython main.py --task train --dataset_name mydata --augment



# Evaluate with visualizations# Evaluate with visualizations

python main.py --task evaluate --dataset_name mydata --visualizepython main.py --task evaluate --dataset_name mydata --visualize



# Multi-pass inference# Multi-pass inference

python main.py --task inference --dataset_name mydata --pass multi 10 --threshold 0.6python main.py --task inference --dataset_name mydata --pass multi 10 --threshold 0.6

``````



## Configuration## Configuration



Configuration is stored in `~/deepEMIA/config/config.yaml`. Key settings:Configuration is stored in `~/deepEMIA/config/config.yaml`. Key settings:



- GCS bucket name- GCS bucket name

- Scale bar detection ROIs- Scale bar detection ROIs

- Hyperparameters (default and optimized)- Hyperparameters (default and optimized)

- Inference settings (thresholds, stopping criteria)- Inference settings (thresholds, stopping criteria)

- Performance optimizations- Performance optimizations



See [Configuration Reference](docs/configuration.md) for details.See [Configuration Reference](docs/configuration.md) for details.



## Security## Security



- Environment variable password hashing- Environment variable password hashing

- Path validation for file operations- Path validation for file operations

- GCS service account authentication- GCS service account authentication

- No plaintext credentials- No plaintext credentials



## License## License



This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).



For commercial use, contact: [info.f@hladek.cz](mailto:info.f@hladek.cz)For commercial use, contact: info.f@hladek.cz



## Support## Support



- Documentation: See `docs/` directory- Documentation: See `docs/` directory

- Issues: [GitHub Issues](https://github.com/Deam0on/deepEMIA/issues)- Issues: [GitHub Issues](https://github.com/Deam0on/deepEMIA/issues)

- Logs: Check `~/logs/` for detailed error information- Logs: Check `~/logs/` for detailed error information



## Citation## Citation



If you use deepEMIA in your research, please cite:If you use deepEMIA in your research, please cite:



```bibtex```bibtex

@software{deepemia,@software{deepemia,

  title={deepEMIA: Deep Learning Electron Microscopy Image Analysis},  title={deepEMIA: Deep Learning Electron Microscopy Image Analysis},

  author={Hladek, F.},  author={Hladek, F.},

  year={2025},  year={2025},

  url={https://github.com/Deam0on/deepEMIA}  url={https://github.com/Deam0on/deepEMIA}

}}

``````



## Acknowledgments## Acknowledgments



Built with Detectron2 by Facebook AI Research (FAIR).Built with Detectron2 by Facebook AI Research (FAIR).



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
- **Hyperparameter Optimization**: Automated hyperparameter tuning using Optuna with dataset-specific configuration persistence and automatic fallback to global/default settings.
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

# Inference configuration
inference_settings:
  use_class_specific_inference: true  # Use class-specific processing
  iterative_stopping:
    min_total_masks: 10          # Minimum masks before considering early stop
    min_relative_increase: 0.25  # 25% minimum increase requirement
    max_consecutive_zero: 1      # Stop after N iterations with no new masks
    min_iterations: 2            # Always run at least this many iterations
  class_specific_settings:
    class_0:  # Large particles
      confidence_threshold: 0.5
      iou_threshold: 0.7
      min_size: 25
    class_1:  # Small particles  
      confidence_threshold: 0.3
      iou_threshold: 0.5
      min_size: 5
      use_multiscale: true

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
    - Early stopping based on configuration: `max_consecutive_zero`, `min_total_masks`, `min_relative_increase`
    - Configurable via `inference_settings.iterative_stopping` in config.yaml

For complete usage information and examples:

```bash
python main.py --help
```

### Advanced Usage Patterns

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

This project is licensed for non-commercial use only.  
For commercial use, please contact the author at [info.f@hladek.cz] to obtain permission.

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

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
- **RCNN hyperparameters**: Default training parameters for R50 and R101 backbones, with support for dataset-specific optimization

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
  # Default hyperparameters used for all datasets when no dataset-specific settings exist
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
  # Global best hyperparameters (used when no dataset-specific best exists)
  best:
    R50: {}
    R101: {}
  # Dataset-specific best hyperparameters (automatically created by optimization)
  # Example: best_polyhipes, best_my_dataset, etc.
  # best_polyhipes:
  #   R50:
  #     base_lr: 0.0001
  #     ims_per_batch: 4
  #     warmup_iters: 1500
  #     gamma: 0.15
  #     batch_size_per_image: 128
  #   R101:
  #     base_lr: 0.0001
  #     ims_per_batch: 2
  #     warmup_iters: 1200
  #     gamma: 0.12
  #     batch_size_per_image: 96
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

### Dataset-Specific Hyperparameters

The deepEMIA project supports dataset-specific hyperparameter optimization, allowing different datasets to use their own optimized training parameters while maintaining global defaults.

#### How It Works

1. **Default Parameters**: All datasets start with the default hyperparameters defined in `rcnn_hyperparameters.default`

2. **Global Best**: When hyperparameter optimization is run without dataset-specific context, results are saved to `rcnn_hyperparameters.best`

3. **Dataset-Specific Best**: When optimization is run for a specific dataset, results are saved to `rcnn_hyperparameters.best_<dataset_name>`

#### Training Parameter Priority

During training, the system uses the following priority order:

1. **Dataset-specific best** (`best_<dataset_name>`) - if available
2. **Global best** (`best`) - if available and dataset-specific doesn't exist
3. **Default parameters** (`default`) - fallback option

#### Running Hyperparameter Optimization

To optimize hyperparameters for a specific dataset:

```bash
# Optimize hyperparameters for a specific dataset
python main.py --task train --dataset_name polyhipes --optimize --n-trials 20

# Use the CLI wizard for guided optimization
python cli_main.py
# Select "train" -> choose dataset -> enable optimization
```

#### Example Configuration After Optimization

After running optimization for a dataset named "polyhipes", your configuration might look like:

```yaml
rcnn_hyperparameters:
  default:
    R50:
      base_lr: 0.00025
      ims_per_batch: 2
      warmup_iters: 1000
      gamma: 0.1
      batch_size_per_image: 64
  best_polyhipes:  # Automatically created by optimization
    R50:
      base_lr: 0.0001
      ims_per_batch: 4
      warmup_iters: 1500
      gamma: 0.15
      batch_size_per_image: 128
```

#### Benefits

- **Dataset-specific optimization**: Each dataset can have its own optimal hyperparameters
- **Automatic fallback**: If no dataset-specific parameters exist, the system gracefully falls back to global best or defaults
- **Reproducibility**: All optimized parameters are saved in the configuration file for reproducible results
- **Easy management**: No manual parameter tuning required - just run optimization and the system handles the rest
