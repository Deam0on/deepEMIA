# UW Computer Vision Project

A comprehensive computer vision pipeline for image processing, model training, evaluation, and inference, with a user-friendly Streamlit interface.

## Project Overview

This project provides a complete pipeline for computer vision tasks, including:
- Dataset preparation and splitting
- Model training
- Model evaluation
- Inference on new images
- Web-based interface for easy interaction

## Project Structure

```
.
├── main.py                 # Main pipeline script
├── streamlit_gui.py        # Streamlit web interface
├── requirements.txt        # Project dependencies
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration file
└── src/                   # Source code
    ├── data/             # Data processing modules
    │   └── data_preparation.py
    ├── models/           # Model-related modules
    │   ├── train_model.py
    │   ├── evaluate_model.py
    │   └── inference.py
    └── utils/            # Utility modules
        ├── gcs_utils.py
        └── eta_utils.py
```

## Features

### Data Management
- Dataset preparation and splitting into train/test sets
- Google Cloud Storage (GCS) integration for data storage
- Support for various image formats (PNG, JPG, TIF)

### Model Pipeline
- Training pipeline with configurable parameters
- Model evaluation with visualization options
- Inference on new images with confidence thresholds
- Progress tracking and ETA estimation

### Web Interface
- User-friendly Streamlit-based interface
- Secure admin access
- Real-time progress monitoring
- Dataset management
- Result visualization and download

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/uw-com-vision.git
cd uw-com-vision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Google Cloud Storage:
- Set up GCS credentials
- Update the bucket name in `config/config.yaml`

## Usage

### Command Line Interface

The main pipeline can be run using `main.py` with the following tasks:

1. Prepare dataset:
```bash
python main.py --task prepare --dataset_name your_dataset
```

2. Train model:
```bash
python main.py --task train --dataset_name your_dataset
```

3. Evaluate model:
```bash
python main.py --task evaluate --dataset_name your_dataset --visualize
```

4. Run inference:
```bash
python main.py --task inference --dataset_name your_dataset --threshold 0.65 --visualize
```

### Web Interface

1. Start the Streamlit interface:
```bash
streamlit run streamlit_gui.py
```

2. Access the interface at `http://localhost:8501`
3. Log in with admin credentials
4. Use the interface to:
   - Upload and manage datasets
   - Run training and inference
   - View and download results

## Configuration

The project uses a YAML configuration file (`config/config.yaml`) for:
- File paths
- GCS bucket settings
- Model parameters
- Training configurations

## Dependencies

Key dependencies include:
- TensorFlow/PyTorch (for model training)
- Streamlit (for web interface)
- Google Cloud Storage
- OpenCV
- NumPy
- Pandas

See `requirements.txt` for the complete list.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the terms of the included LICENSE file.

## Support

For support, please open an issue in the GitHub repository.
