# Functions Module API

API reference for core pipeline operations: training, evaluation, and inference.

## Module: src.functions.train_model

### train_on_dataset

```python
def train_on_dataset(
    dataset_name: str,
    rcnn: int = 101,
    augment: bool = False,
    optimize: bool = False,
    n_trials: int = 10,
    dataset_format: str = "json"
) -> None
```

Train instance segmentation model on dataset.

**Parameters:**
- `dataset_name`: Dataset name
- `rcnn`: Backbone (50, 101, or "combo")
- `augment`: Enable data augmentation
- `optimize`: Run hyperparameter optimization
- `n_trials`: Number of Optuna trials
- `dataset_format`: Annotation format

**Output:** Trained model saved to `~/split_dir/<dataset>/<rcnn>/`

**Example:**

```python
train_on_dataset("polyhipes", rcnn=101, augment=True, optimize=True, n_trials=20)
```

## Module: src.functions.evaluate_model

### evaluate_model

```python
def evaluate_model(
    dataset_name: str,
    output_dir: str,
    visualize: bool = False,
    dataset_format: str = "json",
    rcnn: int = 101
) -> None
```

Evaluate trained model on test set.

**Parameters:**
- `dataset_name`: Dataset name
- `output_dir`: Output directory for results
- `visualize`: Save prediction visualizations
- `dataset_format`: Annotation format
- `rcnn`: RCNN backbone

**Output:** COCO metrics and optional visualizations

## Module: src.functions.inference

### run_inference

```python
def run_inference(
    dataset_name: str,
    output_dir: str,
    visualize: bool = False,
    threshold: float = 0.65,
    draw_id: bool = False,
    dataset_format: str = "json",
    draw_scalebar: bool = False
) -> None
```

Run inference on images with automatic model detection and iterative processing.

**Parameters:**
- `dataset_name`: Dataset name (used to load dataset-specific config)
- `output_dir`: Output directory for results
- `threshold`: Base confidence threshold (may be adjusted by auto-detection)
- `visualize`: Save overlay visualizations
- `draw_id`: Draw instance IDs on visualizations
- `dataset_format`: Annotation format ("json" or "coco")
- `draw_scalebar`: Draw scale bar detection debug info on output

**Features:**
- Automatic model detection (uses available R50/R101 models)
- Iterative inference with configurable stopping criteria
- Multi-scale and tile-based inference
- Model ensemble for improved accuracy
- Spatial constraint filtering
- L4 GPU optimizations

**Output:** CSV files with measurements and optional visualizations

### Helper Functions

#### optimize_predictor_for_l4

```python
def optimize_predictor_for_l4(predictor: DefaultPredictor) -> DefaultPredictor
```

Optimize predictor for L4 GPU performance.

#### calculate_image_quality_score

```python
def calculate_image_quality_score(image: np.ndarray) -> float
```

Calculate image quality metric for adaptive thresholding.

#### cleanup_old_predictions

```python
def cleanup_old_predictions(split_dir: Path, output_dir: Path = None) -> None
```

Remove old prediction files before new inference run.

## See Also

- [Data API](data.md)
- [Utils API](utils.md)
- [User Guide](../user-guide.md)
