# Getting Started with deepEMIA

This guide will help you install and configure deepEMIA for first-time use.

## Prerequisites

Before installing deepEMIA, ensure you have:

- Python 3.8 or higher
- pip package manager
- Google Cloud SDK (for GCS integration)
- CUDA-compatible GPU (optional, for faster training)
- 16GB RAM minimum (32GB recommended for training)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Deam0on/deepEMIA.git
cd deepEMIA
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Google Cloud SDK

Follow the [official Google Cloud SDK installation guide](https://cloud.google.com/sdk/docs/install) for your operating system.

After installation, authenticate:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## Initial Configuration

### 1. Set Up Security

Generate an admin password hash for the web interface:

```bash
# Linux/Mac
export ADMIN_PASSWORD_HASH=$(echo -n 'your_secure_password' | sha256sum | cut -d' ' -f1)

# Windows PowerShell
$password = 'your_secure_password'
$hash = [System.Security.Cryptography.SHA256]::Create().ComputeHash([System.Text.Encoding]::UTF8.GetBytes($password))
$env:ADMIN_PASSWORD_HASH = [System.BitConverter]::ToString($hash).Replace('-','').ToLower()
```

For permanent configuration, add to your shell profile (.bashrc, .zshrc, or PowerShell profile).

### 2. Run Initial Setup

Use the interactive CLI wizard:

```bash
python cli_main.py
```

Select "setup" from the menu and provide:
- Google Cloud Storage bucket name
- Scale bar ROI settings (use defaults if unsure)
- Intensity and proximity thresholds

Or use the direct command:

```bash
python main.py --task setup
```

### 3. Verify Configuration

The setup process creates `~/deepEMIA/config/config.yaml`. Verify it contains:

```yaml
bucket: your-bucket-name
paths:
  main_script: "~/deepEMIA/main.py"
  split_dir: "~/split_dir"
  category_json: "~/deepEMIA/dataset_info.json"
  dataset_configs_dir: "~/deepEMIA/config/datasets"
  # ... other paths
```

Dataset-specific configurations can be created in `~/deepEMIA/config/datasets/`.

## Testing the Installation

### 1. Test Basic Functionality

```bash
python -c "from src.utils.config import get_config; print('Config loaded successfully')"
```

### 2. Test GCS Connection

```bash
gsutil ls gs://your-bucket-name/
```

### 3. Run a Simple Test

Try running the setup task again to ensure everything works:

```bash
python main.py --task setup --dataset_name test
```

## Directory Structure

After installation, your directory structure should look like:

```
deepEMIA/
├── config/
│   ├── config.yaml          # Main configuration (defaults)
│   ├── eta_data.json         # ETA tracking
│   ├── datasets/             # Dataset-specific configs
│   │   └── <dataset_name>.yaml
│   └── datasets.example/     # Example templates
│       ├── template.yaml
│       └── polyhipes_tommy.yaml
├── docs/                     # Documentation
├── gui_legacy/               # Legacy web interface
├── src/                      # Source code
│   ├── data/                 # Data handling
│   ├── functions/            # Core pipeline functions
│   └── utils/                # Utility functions
├── main.py                   # Main CLI entry point
├── cli_main.py               # Interactive CLI wizard
└── requirements.txt          # Dependencies
```

## Next Steps

Now that you have deepEMIA installed:

1. Read the [User Guide](user-guide.md) to learn how to use the tool
2. Review [Configuration](configuration.md) for advanced settings
3. Follow the [Basic Workflow Example](examples/basic-workflow.md)
4. Explore the [API Reference](api/overview.md) if developing

## Troubleshooting

### Import Errors

If you encounter import errors:

```bash
pip install -r requirements.txt --upgrade
```

### CUDA Not Available

If PyTorch doesn't detect your GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Install the correct CUDA version of PyTorch for your system.

### GCS Authentication Issues

If you have GCS authentication problems:

```bash
gcloud auth application-default login
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### Configuration File Not Found

Ensure the config directory exists in your home directory:

```bash
mkdir -p ~/deepEMIA/config
```

Then re-run the setup task.

## Getting Help

- Check the [User Guide](user-guide.md#troubleshooting) for common issues
- Review logs in `~/logs/` for detailed error information
- Open an issue on [GitHub](https://github.com/Deam0on/deepEMIA/issues)
