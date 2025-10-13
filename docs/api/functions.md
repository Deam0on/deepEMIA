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
    threshold: float = 0.65,
    visualize: bool = False,
    draw_instance_id: bool = False,
    inference_mode: str = "single",
    max_iterations: int = 10,
    rcnn: int = 101
) -> None
```

Run inference on images.

**Parameters:**
- `dataset_name`: Dataset name
- `threshold`: Confidence threshold
- `visualize`: Save visualizations
- `draw_instance_id`: Draw IDs on visualizations
- `inference_mode`: "single" or "multi"
- `max_iterations`: Max iterations for multi mode
- `rcnn`: RCNN backbone

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
