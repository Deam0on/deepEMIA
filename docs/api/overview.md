# API Reference Overview

This section provides technical documentation for the deepEMIA Python API.

## API Organization

The API is organized into three main categories:

### [Data Module](data.md)

Dataset management, model loading, and data preprocessing:

- Dataset registration and splitting
- COCO format conversion
- Model loading and configuration
- Custom data mappers

### [Functions Module](functions.md)

Core pipeline operations:

- Model training with hyperparameter optimization
- Model evaluation with COCO metrics
- Inference with multi-scale and class-specific processing
- Post-processing and measurement extraction

### [Utilities Module](utils.md)

Helper functions and utilities:

- Configuration management with dataset-specific overrides
- Logging utilities
- GCS integration
- GPU availability checking
- Mask processing
- Scale bar detection
- Measurement calculations
- Spatial constraints (containment/overlap rules)
- Safe file operations

## API Conventions

### Function Naming

- **Action verbs**: Functions start with action verbs (get, load, run, calculate, etc.)
- **Snake case**: All functions use snake_case naming
- **Descriptive**: Names clearly indicate purpose

### Parameters

- **Type hints**: All functions include type hints where possible
- **Default values**: Optional parameters have sensible defaults
- **Documentation**: All parameters documented in docstrings

### Return Values

- **Consistent types**: Functions return consistent types
- **None on failure**: Functions may return None for optional operations
- **Exceptions**: Raise custom exceptions for errors

### Error Handling

Custom exceptions defined in `src.utils.exceptions`:

- `PipelineError`: Base exception for pipeline operations
- `ConfigurationError`: Invalid or missing configuration
- `ModelLoadError`: Model loading failures
- `TrainingError`: Training operation failures

## Import Patterns

### Importing Functions

```python
# Import specific functions
from src.functions.train_model import train_on_dataset
from src.functions.evaluate_model import evaluate_model
from src.functions.inference import run_inference

# Import data utilities
from src.data.datasets import register_datasets, split_dataset
from src.data.models import load_model, get_trained_model_paths
```

### Importing Utilities

```python
# Configuration
from src.utils.config import get_config

# Logging
from src.utils.logger_utils import system_logger

# Measurements
from src.utils.measurements import calculate_measurements

# Mask processing
from src.utils.mask_utils import postprocess_masks, rle_encoding
```

## Configuration

All API functions read configuration from the global config:

```python
from src.utils.config import get_config

config = get_config()
bucket = config["bucket"]
paths = config["paths"]
```

See [Configuration Reference](../configuration.md) for details.

## Logging

All modules use the centralized logging system:

```python
from src.utils.logger_utils import system_logger

system_logger.info("Starting processing")
system_logger.error("An error occurred", exc_info=True)
```

Logs are written to `~/logs/system_YYYY-MM-DD_HH-MM-SS.log`.

## Common Workflows

### Training Pipeline

```python
from src.data.datasets import register_datasets, read_dataset_info
from src.functions.train_model import train_on_dataset

# Register dataset
dataset_info = read_dataset_info("dataset_info.json")
register_datasets(dataset_info, "polyhipes", test_size=0.2)

# Train model
train_on_dataset(
    dataset_name="polyhipes",
    rcnn=101,
    augment=True,
    optimize=True,
    n_trials=20
)
```

### Inference Pipeline

```python
from src.functions.inference import run_inference

# Run inference
run_inference(
    dataset_name="polyhipes",
    threshold=0.65,
    visualize=True,
    inference_mode="multi",
    max_iterations=10
)
```

### Evaluation Pipeline

```python
from src.functions.evaluate_model import evaluate_model

# Evaluate model
evaluate_model(
    dataset_name="polyhipes",
    output_dir="./output",
    visualize=True,
    rcnn=101
)
```

## API Stability

### Stable APIs

The following APIs are considered stable:

- Core pipeline functions (train, evaluate, inference)
- Configuration management
- Dataset registration
- Model loading

### Experimental APIs

The following APIs may change:

- Multi-scale inference algorithms
- Spatial constraint filtering
- Class-specific inference settings

## Next Steps

- [Data Module API](data.md) - Dataset and model management
- [Functions Module API](functions.md) - Core pipeline operations
- [Utilities Module API](utils.md) - Helper functions and utilities

## See Also

- [User Guide](../user-guide.md) - High-level usage guide
- [Architecture](../architecture/pipeline-overview.md) - System design
- [Examples](../examples/basic-workflow.md) - Working code examples
