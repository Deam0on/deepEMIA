# GPU Availability Checking

## Overview

The deepEMIA project now includes automatic GPU availability checking to prevent silent failures when GPU/CUDA initialization fails on remote VMs or other environments. This feature is particularly important for L4 GPU instances where GPU initialization can fail intermittently.

## Features

- **Automatic GPU Detection**: Checks for CUDA-capable GPUs at pipeline start
- **Interactive Prompts**: User can choose to continue or abort when GPU is unavailable
- **Detailed Logging**: Comprehensive GPU information logging (model, VRAM, CUDA version)
- **Task-Specific Handling**: Different behavior for GPU-intensive vs. GPU-optional tasks
- **Non-Interactive Mode**: Support for automated/scripted execution

## How It Works

### Main Pipeline (`main.py`)

The GPU check is performed automatically before any GPU-intensive task:

```bash
# Normal execution - will prompt if no GPU detected
python main.py --task train --dataset_name polyhipes

# Automated execution - skip GPU check (for scripts)
python main.py --task train --dataset_name polyhipes --no-gpu-check
```

### Task Classification

**GPU-Required Tasks** (will prompt to continue):
- `train` - Model training
- `inference` - Running predictions
- `evaluate` - Model evaluation

**GPU-Optional Tasks** (no prompt):
- `prepare` - Dataset preparation
- `setup` - Configuration

### Interactive Mode

When GPU is not detected during an interactive session:

```
============================================================
WARNING: No GPU/CUDA device detected!
============================================================

This will significantly impact performance:
  • Training may be 10-50x slower
  • Inference may be 5-20x slower

Possible causes:
  • GPU drivers not installed
  • CUDA not installed or incompatible version
  • GPU initialization failed
  • Running on CPU-only instance

Recommendation: Check GPU status with 'nvidia-smi'
============================================================

Do you want to continue anyway? (yes/no):
```

User can choose:
- `yes`/`y` - Continue with CPU-only mode
- `no`/`n` - Abort execution

## Usage Examples

### 1. Training with GPU Check

```bash
# Standard training - will check GPU automatically
python main.py --task train --dataset_name polyhipes --rcnn 101

# Training with disabled GPU check (for automated scripts)
python main.py --task train --dataset_name polyhipes --rcnn 101 --no-gpu-check
```

### 2. Inference with GPU Check

```bash
# Inference with GPU check
python main.py --task inference --dataset_name polyhipes --threshold 0.65 --visualize

# Non-interactive inference (won't prompt, just warns)
python main.py --task inference --dataset_name polyhipes --no-gpu-check
```

### 3. Using the CLI Wizard

The interactive CLI wizard (`cli_main.py`) also includes GPU checking:

```bash
python cli_main.py
```

The wizard displays GPU information at startup but doesn't block execution.

## Programmatic Usage

### In Your Own Scripts

```python
from src.utils.gpu_check import check_gpu_availability, log_device_info

# Check and log GPU info
log_device_info()

# Interactive check (requires GPU)
if not check_gpu_availability(require_gpu=True, interactive=True):
    print("GPU not available, aborting")
    sys.exit(1)

# Non-interactive check (optional GPU)
check_gpu_availability(require_gpu=False, interactive=False)
```

### Function Parameters

#### `check_gpu_availability(require_gpu=False, interactive=True)`

**Parameters:**
- `require_gpu` (bool): If True, GPU is considered required for the task
- `interactive` (bool): If True, prompts user when GPU unavailable

**Returns:**
- `bool`: True if GPU available or user confirmed to continue, False otherwise

#### `log_device_info()`

Logs comprehensive GPU/CPU information:
- GPU model and count
- CUDA version
- VRAM capacity
- Compute capability
- PyTorch version

## Testing GPU Check

Test the GPU check functionality:

```bash
python test_gpu_check.py
```

This will:
1. Log device information
2. Test interactive GPU check
3. Test non-interactive GPU check

## Troubleshooting

### GPU Not Detected

If GPU is not detected but should be available:

1. **Check GPU Status**:
   ```bash
   nvidia-smi
   ```

2. **Verify CUDA Installation**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Check CUDA Version Compatibility**:
   - PyTorch: 1.10.0+cu113
   - CUDA: 11.3
   - Ensure CUDA version matches PyTorch requirements

4. **Reinstall GPU Drivers**:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install nvidia-driver-XXX
   ```

5. **Verify Environment**:
   ```bash
   python -c "import torch; print(f'CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}')"
   ```

### Automated/Scripted Execution

For automated workflows where prompts should be avoided:

```bash
# Disable GPU check entirely
python main.py --task train --dataset_name polyhipes --no-gpu-check

# Or use non-interactive mode programmatically
from src.utils.gpu_check import check_gpu_availability
check_gpu_availability(require_gpu=False, interactive=False)
```

## GPU Information Logged

When GPU is available, the following information is logged:

```
✓ GPU available: NVIDIA L4
  CUDA version: 11.3
  Device count: 1
  GPU 0: NVIDIA L4, 22.5GB VRAM
=== GPU Device Information ===
GPU 0:
  Name: NVIDIA L4
  Total Memory: 22.50GB
  Compute Capability: 8.9
  Multi-Processors: 58
PyTorch version: 1.10.0+cu113
CUDA available: True
CUDA version: 11.3
cuDNN version: 8200
```

When GPU is not available:

```
⚠ No GPU/CUDA device detected!
  This may significantly impact performance for training and inference.
  Possible causes:
    - GPU drivers not installed
    - CUDA not installed or incompatible version
    - GPU initialization failed
    - Running on CPU-only instance
```

## Integration with Existing Code

The GPU check is integrated at key entry points:

1. **main.py**: Before any task execution (except setup), controllable via `--no-gpu-check`
2. **train_model.py**: Automatic GPU detection and optimization
3. **inference.py**: L4-optimized inference with mixed precision support
4. **evaluate_model.py**: GPU-accelerated evaluation
5. **cli_main.py**: At CLI startup (non-blocking informational display)

## Performance Impact

### With GPU (NVIDIA L4)
- Training: ~30 minutes per epoch
- Inference: ~5-10 seconds per image

### Without GPU (CPU-only)
- Training: ~5-15 hours per epoch (10-50x slower)
- Inference: ~30-120 seconds per image (5-20x slower)

## Best Practices

1. **Always check GPU status** before long-running tasks
2. **Use `--no-gpu-check`** for automated scripts where intervention is not possible
3. **Monitor GPU health** with `nvidia-smi` regularly on VMs
4. **Set up alerts** for GPU initialization failures in production
5. **Consider CPU-only mode** only for testing or development

## Environment Variables

For automated deployments, you can set:

```bash
# Skip all GPU checks (use with caution)
export DEEPEMIA_SKIP_GPU_CHECK=1
```

Then in your script:
```python
import os
skip_check = os.environ.get('DEEPEMIA_SKIP_GPU_CHECK', '0') == '1'
```

## See Also

- [Training Pipeline Documentation](../architecture/training-pipeline.md)
- [Inference Pipeline Documentation](../architecture/inference-pipeline.md)
- [L4 GPU Optimizations](../configuration.md#l4-performance-optimizations)
