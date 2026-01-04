# deepEMIA

Deep(learning) E(lectron) MI(croscopy) A(nalysis) is a comprehensive deep learning toolkit for electron microscopy and scientific image analysis. It offers a robust pipeline for dataset preparation, model training, hyperparameter optimization, evaluation, scalable inference, and automated measurement extraction, with both CLI and web interfaces.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [CLI](#cli)
  - [Web Interface](#web-interface)
- [Configuration](#configuration)
- [Dataset Format](#dataset-format)
- [Output](#output)
- [Security](#security)
- [Troubleshooting & Support](#troubleshooting--support)
- [License & Citation](#license--citation)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Dataset Management**: Preparation, splitting, and registration (custom/COCO) with dataset-specific configuration files.
- **Model Training & Optimization**: Supports Detectron2 backbones (R50, R101, combo) and Optuna-based hyperparameter tuning.
- **Evaluation**: COCO-style metrics and visualizations.
- **Inference**: Batch, iterative, multi-scale, and tile-based inference with measurement extraction (area, perimeter, axes, circularity, wavelength).
- **Model Ensemble**: Automatic multi-model ensemble for improved detection accuracy.
- **Spatial Constraints**: Configurable containment and overlap rules for class relationships.
- **Interfaces**: Interactive CLI wizard, direct CLI, and Streamlit web GUI.
- **Cloud Integration**: Google Cloud Storage (GCS) for scalable data management.
- **GPU Management**: Automatic GPU detection with interactive prompts and L4 GPU optimizations.
- **Security**: Environment-variable password hashing, path validation, and service account authentication.
- **Error Handling**: Retry logic, detailed logging, and configuration validation.
- **Performance**: Fast spatial filtering, tile-based processing, mixed precision, and memory-efficient inference.

---

## Installation

### Prerequisites

- Python 3.8+
- pip
- Google Cloud SDK (for GCS integration)
- (Optional) CUDA-compatible GPU

### Steps

```bash
git clone https://github.com/Deam0on/deepEMIA.git
cd deepEMIA
pip install -r requirements.txt
```

---

## Quick Start

### Initial Setup

```bash
# Interactive setup wizard
python cli_main.py
# Or direct setup
python main.py --task setup --dataset_name dummy
```

### Example Workflow

```bash
# Prepare dataset
python main.py --task prepare --dataset_name my_dataset

# Train with hyperparameter optimization
python main.py --task train --dataset_name my_dataset --optimize --n-trials 20 --augment

# Evaluate model
python main.py --task evaluate --dataset_name my_dataset --visualize

# Run inference
python main.py --task inference --dataset_name my_dataset --threshold 0.65 --visualize
```

---

## Usage

### CLI

All tasks can be run via CLI:

```bash
python main.py --task <prepare|train|evaluate|inference|setup> --dataset_name <name> [options]
```

**Key Options:**
- `--rcnn`: RCNN backbone (`50`, `101`, `combo`), default: `101`
- `--threshold`: Inference confidence threshold, default: `0.65`
- `--augment`: Enable data augmentation
- `--optimize`: Activate hyperparameter optimization (Optuna)
- `--n-trials`: Number of Optuna trials
- `--visualize`: Save visualizations
- `--id`: Draw instance IDs on visualizations
- `--draw-scalebar`: Draw scale bar detection debug info on output images
- `--no-gpu-check`: Skip GPU availability check (for automated scripts)
- `--verbosity`: Console logging level (`debug`, `info`, `warning`, `error`)
- `--download/--upload`: Sync data/results with GCS

For full argument list:
```bash
python main.py --help
```

### Web Interface

Launch the Streamlit GUI:

```bash
streamlit run gui_legacy/streamlit_gui.py
```

---

## Configuration

- **Main config**: `~/deepEMIA/config/config.yaml` (global defaults)
- **Dataset-specific configs**: `~/deepEMIA/config/datasets/<dataset_name>.yaml`
- **Example templates**: `~/deepEMIA/config/datasets.example/`

Set environment variables for security:
```bash
export ADMIN_PASSWORD_HASH="your_sha256_hash_here"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gcs-key.json"
```

Key settings include:
- GCS bucket name and paths
- Scale bar ROI and detection thresholds
- RCNN hyperparameters (default and per-dataset)
- Inference settings (class-specific, iterative stopping, ensemble, tile-based)
- Spatial constraints (containment/overlap rules)
- L4 GPU performance optimizations

See [docs/configuration.md](docs/configuration.md) for details.

---

## Dataset Format

**Structure:**
```
DATASET/
├── your_dataset_name/
│   ├── image1.tif
│   ├── image1.json
│   ├── image2.tif
│   ├── image2.json
│   └── ...
└── INFERENCE/
    ├── test_image1.tif
    └── ...
```

**Annotation Example:**
```json
{
  "imagePath": "image1.tif",
  "imageHeight": 1024,
  "imageWidth": 1024,
  "shapes": [
    {
      "label": "particle",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "polygon"
    }
  ]
}
```

---

## Output

- CSV with geometric measurements (`area`, `perimeter`, `axes`, `circularity`, `wavelength`)
- Overlay images with detected particles
- Logs and metrics in `~/logs/`
- Model weights: `model_final.pth`, `model_quantized.pth`

---

## Security

- Passwords stored as SHA256 hashes in environment variables
- Strict path validation for all file operations
- GCS service account authentication required
- No plaintext credentials stored

---

## Troubleshooting & Support

- Logs: `~/logs/`
- Common issues:
  - Config not found: run setup
  - GCS errors: check SDK, credentials, and permissions
  - CUDA OOM: lower batch size or use CPU
  - Scale bar issues: adjust ROI and thresholds
- For help: review logs, validate configuration, open [GitHub Issues](https://github.com/Deam0on/deepEMIA/issues)
- See the full [Documentation](docs/) for advanced topics and troubleshooting.

---

## License & Citation

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

For commercial use, contact: [info.f@hladek.cz](mailto:info.f@hladek.cz)

If used in research, please cite:

```bibtex
@software{deepemia,
  title={deepEMIA: Deep Learning Electron Microscopy Image Analysis},
  author={Hládek, Filip},
  year={2026},
  version={2.0.0},
  url={https://github.com/Deam0on/deepEMIA}
}
```

---

## Acknowledgments

Built with Detectron2 by Facebook AI Research (FAIR).
