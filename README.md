# Deep-Learning (primarily Electron Microscopy) Image Analysis Tool

A modular, end-to-end computer vision pipeline for scientific image analysis, featuring dataset management, model training, evaluation, inference, and a Streamlit-based web interface. The project is designed for extensibility, reproducibility, and integration with Google Cloud Storage.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Web Interface](#web-interface)
- [Google Cloud Storage Integration](#google-cloud-storage-integration)
- [Extending the Pipeline](#extending-the-pipeline)
- [Logs](#logs)
- [Web Interface Notes](#web-interface-notes)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## Overview

This repository provides a robust pipeline for computer vision tasks, including:

- Dataset preparation and splitting
- Model training and quantization (Detectron2, PyTorch)
- Model evaluation and metrics reporting
- Inference on new images with advanced measurement and analysis
- Web-based interface for streamlined workflow management
- Integration with Google Cloud Storage for scalable data management

The pipeline is suitable for scientific and industrial applications requiring reproducible, automated image analysis.

## Features

- **Dataset Management**: Prepare, split, and register datasets in custom or COCO format. Supports per-image JSON and COCO-style annotations.
- **Model Training**: Train instance segmentation models with Detectron2, including CPU-optimized and quantized models for efficient inference.
- **Evaluation**: COCO-style evaluation with metrics export and optional visualization.
- **Inference**: Batch inference with measurement extraction (e.g., scale bar detection, geometric analysis), result encoding, and CSV export.
- **Iterative Inference**: Optionally repeat mask prediction and deduplication until no significant new masks are found, improving recall for challenging images. Control this with `--pass multi [max_iters]`.
- **Web Interface**: Streamlit-based GUI for dataset management, task execution, progress monitoring, and result visualization/download.
- **Cloud Integration**: Automated upload/download of datasets and results to/from Google Cloud Storage.
- **ETA Tracking**: Automatic estimation and tracking of task durations for user feedback.
- **Extensibility**: Modular codebase for easy extension and adaptation to new tasks or data formats.

## Project Structure

```
.
├── main.py                      # Main pipeline script (CLI entry point)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── LICENSE                      # License file
├── config/
│   ├── config.yaml              # Main configuration file (paths, bucket, etc.)
│   └── eta_data.json            # Timing/ETA tracking data
├── gui/
│   ├── streamlit_gui.py         # Streamlit web interface
│   └── streamlit_functions.py   # Streamlit utility functions
├── src/
│   ├── data/
│   │   ├── datasets.py          # Dataset splitting and registration
│   │   ├── models.py            # Model loading and selection
│   │   └── custom_mapper.py     # Data augmentation and mapping
│   ├── functions/
│   │   ├── train_model.py       # Model training logic
│   │   ├── evaluate_model.py    # Model evaluation logic
│   │   └── inference.py         # Inference and measurement extraction
│   └── utils/
│       ├── eta_utils.py         # ETA/time tracking utilities
│       ├── gcs_utils.py         # Google Cloud Storage utilities
│       ├── mask_utils.py        # Mask encoding/decoding and postprocessing
│       ├── measurements.py      # Measurement and color utilities
│       └── scalebar_ocr.py      # Scale bar detection via OCR
└── .github/
    └── ISSUE_TEMPLATE/          # GitHub issue templates
```

## Installation

### Prerequisites

- Python 3.8 or newer
- Google Cloud SDK (for `gsutil` and authentication)
- [PyTorch](https://pytorch.org/) (CPU or CUDA version as appropriate)
- [Detectron2](https://github.com/facebookresearch/detectron2) (installed via requirements)
- Google Cloud credentials configured (for GCS access)

### Steps

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/deepEMIA.git
   cd deepEMIA
   ```

2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

3. **Configure Google Cloud Storage:**

   - Set up your GCS credentials (e.g., `gcloud auth application-default login`)
   - Edit `config/config.yaml` to set your bucket name and paths as needed.

4. **Prepare your dataset:**

   - Place your images and annotation files in the appropriate directory structure.
   - Update or create `dataset_info.json` to describe your datasets and classes.

## Configuration

All configuration is managed via [`config/config.yaml`](config/config.yaml):

- `bucket`: Name of your GCS bucket
- `paths`: Paths for scripts, splits, category info, ETA file, and dataset root
- `scale_bar_rois`: Default and per-dataset ROI settings for scale bar detection
- `admin_password`: Password for accessing admin features in the web interface. Placeholder for now, will be re-done with fernet.crypto

Example:

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

After setup, you can proceed to use the other tasks (`prepare`, `train`, `evaluate`, `inference`) as described below.

## Usage

### Command Line Interface

The main pipeline is controlled via [`main.py`](main.py):

For a full list of options and usage examples, run:

```sh
python main.py --help
```

#### Prepare Dataset

```sh
python main.py --task prepare --dataset_name <dataset_name>
```

#### Train Model

```sh
python main.py --task train --dataset_name <dataset_name> --dataset_format json
```

#### Evaluate Model

```sh
python main.py --task evaluate --dataset_name <dataset_name> --visualize
```

#### Run Inference

```sh
python main.py --task inference --dataset_name <dataset_name> --threshold 0.65 --visualize
```

**Additional options:**

- `--pass single` : (Default) Run a single inference pass per image.
- `--pass multi [max_iters]` : Run iterative inference, repeating mask prediction and deduplication until the number of unique masks increases by less than 10% (or your configured threshold), or until `max_iters` is reached. For example, `--pass multi 5` limits to 5 iterations (default is 10 if not specified).

**Examples:**

Run inference with iterative deduplication, up to 5 iterations:
```sh
python main.py --task inference --dataset_name <dataset_name> --rcnn combo --pass multi 5
```

Run inference with a single pass (default):
```sh
python main.py --task inference --dataset_name <dataset_name> --rcnn combo --pass single
```

### RCNN Backbone Selection

You can specify which Mask R-CNN backbone to use for both training and inference:

- `--rcnn 50` : Use ResNet-50 backbone (better for small particles)
- `--rcnn 101` : Use ResNet-101 backbone (better for large particles, default)
- `--rcnn combo` : Use both backbones; for training, both models are trained and saved; for inference, predictions from both models are merged and deduplicated using standard NMS (IoU=0.5).

**Examples:**

Train with both backbones:
```sh
python main.py --task train --dataset_name <dataset_name> --rcnn combo
```

Run inference with both backbones and merge results:
```sh
python main.py --task inference --dataset_name <dataset_name> --rcnn combo
```

When using `--rcnn combo` for inference, the pipeline will run both models on each image, merge their predictions, and remove duplicates using non-maximum suppression with the default IoU threshold. If `--pass multi` is used, this process is repeated until no significant new masks are found or the maximum number of iterations is reached.

### Data Augmentation

You can enable data augmentation during training by adding the `--augment` flag to your command. This will apply random flips, rotations, and brightness changes to both images and their annotations.

**Example:**

```sh
python main.py --task train --augment
```

If you omit `--augment`, training will use the original images without augmentation.

### Web Interface

A Streamlit-based web interface is provided for interactive use.

1. **Start the interface:**

   ```sh
   streamlit run gui/streamlit_gui.py
   ```

2. **Access in your browser:**

   - Navigate to `http://localhost:8501`
   - Log in with the admin password (default: `admin`)

3. **Features:**

   - Select and manage datasets
   - Run pipeline tasks (prepare, train, evaluate, inference)
   - Upload/download data to/from GCS
   - Monitor progress and ETA
   - Visualize and download results

## Google Cloud Storage Integration

- All data and results can be synchronized with your configured GCS bucket.
- Uploads and downloads are managed automatically via the pipeline and web interface.
- Archive folders are timestamped for reproducibility and traceability.

## Extending the Pipeline

- **Add new datasets:** Update `dataset_info.json` and place your data in the expected structure.
- **Add new tasks or models:** Implement new modules in `src/functions/` or `src/data/`.
- **Customize measurement/extraction:** Extend or modify `src/utils/measurements.py` for geometric and color measurements, or `src/utils/scalebar_ocr.py` for scale bar detection.
- **Web interface:** Add new features to `gui/streamlit_gui.py` and `gui/streamlit_functions.py`.

## Logs

All pipeline logs are saved in the directory specified by `logs_dir` in your config (default: `~/logs`). After each run, logs are uploaded to your GCS bucket for traceability.

## Web Interface Notes

- The RCNN backbone selector defaults to "Dual model (universal)" for best generalization.
- The upload section is now at the top for a more intuitive workflow.
- Only warnings and errors are shown in the error panel for clarity.
- The log viewer always shows the latest run log.

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes with clear messages
4. Push to your fork
5. Open a Pull Request describing your changes

Please review the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Support

For questions, bug reports, or feature requests, please open an issue in the [GitHub repository](https://github.com/yourusername/deepEMIA/issues).
