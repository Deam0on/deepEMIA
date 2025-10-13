# Data Module API

API reference for dataset management and model loading.

## Module: src.data.datasets

Dataset preparation, splitting, and registration functions.

### split_dataset

```python
def split_dataset(
    img_dir: str,
    dataset_name: str,
    test_size: float = 0.2,
    seed: int = 42
) -> None
```

Split dataset into train and test sets.

**Parameters:**
- `img_dir`: Path to directory containing images
- `dataset_name`: Name of the dataset
- `test_size`: Fraction of data for testing (0.0-1.0)
- `seed`: Random seed for reproducibility

**Returns:** None

**Example:**

```python
split_dataset("/path/to/images", "polyhipes", test_size=0.2)
```

### register_datasets

```python
def register_datasets(
    dataset_info: dict,
    dataset_name: str,
    test_size: float = 0.2,
    dataset_format: str = "json"
) -> None
```

Register datasets with Detectron2.

**Parameters:**
- `dataset_info`: Dictionary containing dataset information
- `dataset_name`: Name of the dataset
- `test_size`: Test set fraction
- `dataset_format`: Format ("json" or "coco")

**Returns:** None

### read_dataset_info

```python
def read_dataset_info(file_path: str) -> dict
```

Read dataset information from JSON file.

**Parameters:**
- `file_path`: Path to dataset_info.json

**Returns:** Dictionary with dataset metadata

## Module: src.data.models

Model loading and configuration functions.

### get_trained_model_paths

```python
def get_trained_model_paths(
    base_dir: str,
    rcnn: int = 101
) -> dict
```

Get paths to trained model files.

**Parameters:**
- `base_dir`: Base directory containing models
- `rcnn`: RCNN backbone (50 or 101)

**Returns:** Dictionary with model paths:
```python
{
    "model_path": "/path/to/model_final.pth",
    "quantized_path": "/path/to/model_quantized.pth"
}
```

### load_model

```python
def load_model(
    cfg: CfgNode,
    model_path: str,
    dataset_name: str,
    is_quantized: bool = False
) -> DefaultPredictor
```

Load trained model for inference.

**Parameters:**
- `cfg`: Detectron2 configuration
- `model_path`: Path to model weights
- `dataset_name`: Dataset name
- `is_quantized`: Whether to load quantized model

**Returns:** Detectron2 predictor instance

**Raises:**
- `ModelLoadError`: If model fails to load

### choose_and_use_model

```python
def choose_and_use_model(
    model_paths: dict,
    dataset_name: str,
    threshold: float,
    metadata: MetadataCatalog,
    rcnn: int = 101
) -> DefaultPredictor
```

Choose and load the best available model.

**Parameters:**
- `model_paths`: Dictionary of available model paths
- `dataset_name`: Dataset name
- `threshold`: Confidence threshold
- `metadata`: Dataset metadata
- `rcnn`: RCNN backbone

**Returns:** Configured predictor

## Module: src.data.custom_mapper

Custom data augmentation and preprocessing.

### custom_mapper

```python
def custom_mapper(
    dataset_dict: dict,
    augment: bool = False
) -> dict
```

Custom data mapper for Detectron2 training.

**Parameters:**
- `dataset_dict`: Dictionary with image and annotation data
- `augment`: Whether to apply augmentation

**Returns:** Processed dataset dictionary with:
- `image`: Tensor in CHW format
- `instances`: Instances object with annotations

**Augmentations Applied:**
- Random horizontal/vertical flips (50% probability)
- Random rotation (-20 to 20 degrees)
- Random brightness (0.8 to 1.2x)

**Example:**

```python
from detectron2.data import build_detection_train_loader

# Use custom mapper with augmentation
train_loader = build_detection_train_loader(
    cfg,
    mapper=lambda x: custom_mapper(x, augment=True)
)
```

## Common Data Workflows

### Prepare Dataset for Training

```python
from src.data.datasets import register_datasets, read_dataset_info, split_dataset

# Read dataset information
dataset_info = read_dataset_info("dataset_info.json")

# Split dataset
split_dataset("/path/to/images", "mydata", test_size=0.2)

# Register with Detectron2
register_datasets(dataset_info, "mydata", dataset_format="json")
```

### Load Model for Inference

```python
from src.data.models import get_trained_model_paths, choose_and_use_model
from detectron2.data import MetadataCatalog

# Get model paths
model_paths = get_trained_model_paths("~/split_dir/mydata", rcnn=101)

# Load model
metadata = MetadataCatalog.get("mydata_train")
predictor = choose_and_use_model(
    model_paths=model_paths,
    dataset_name="mydata",
    threshold=0.65,
    metadata=metadata,
    rcnn=101
)
```

## See Also

- [Functions API](functions.md) - Training and inference functions
- [Architecture](../architecture/pipeline-overview.md) - Data pipeline design
- [Examples](../examples/basic-workflow.md) - Complete workflows
