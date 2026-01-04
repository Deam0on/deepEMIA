# Utilities Module API

API reference for utility functions.

## Configuration (src.utils.config)

### get_config

```python
def get_config(dataset_name: str = None) -> dict
```

Load and return project configuration with optional dataset-specific overrides.

**Parameters:**
- `dataset_name`: Optional dataset name to merge dataset-specific settings

**Returns:** Configuration dictionary (merged with dataset-specific config if provided)

**Raises:** 
- `FileNotFoundError`: Config file not found
- `yaml.YAMLError`: Invalid YAML
- `ConfigValidationError`: Invalid configuration

### list_dataset_configs

```python
def list_dataset_configs() -> list
```

List all available dataset-specific configuration files.

**Returns:** List of dataset names with configs

### create_dataset_config

```python
def create_dataset_config(dataset_name: str, template: str = "template") -> Path
```

Create a new dataset-specific config from template.

**Parameters:**
- `dataset_name`: Name for the new dataset config
- `template`: Template to use ('template', 'polyhipes_tommy', or existing dataset name)

**Returns:** Path to created config file

## Logging (src.utils.logger_utils)

### system_logger

```python
from src.utils.logger_utils import system_logger

system_logger.info("Message")
system_logger.error("Error", exc_info=True)
```

Global logger instance. Writes to `~/logs/system_*.log`

## GCS Integration (src.utils.gcs_utils)

### download_data_from_bucket

```python
def download_data_from_bucket() -> float
```

Download dataset from GCS. Returns time taken.

### upload_data_to_bucket

```python
def upload_data_to_bucket() -> float
```

Upload results to GCS. Returns time taken.

## Mask Processing (src.utils.mask_utils)

### postprocess_masks

```python
def postprocess_masks(
    ori_mask: np.ndarray,
    ori_score: np.ndarray,
    image: np.ndarray,
    min_crys_size: int = None
) -> list
```

Post-process predicted masks: remove overlaps, fill holes, smooth boundaries.

### rle_encoding

```python
def rle_encoding(x: np.ndarray) -> list
```

Encode binary mask to run-length encoding.

## Measurements (src.utils.measurements)

### calculate_measurements

```python
def calculate_measurements(
    c: np.ndarray,
    single_im_mask: np.ndarray,
    um_pix: float = 1.0,
    pixelsPerMetric: float = 1.0,
    original_image: np.ndarray = None,
    measure_contrast_distribution: bool = False
) -> dict
```

Calculate geometric and physical measurements for detected particle.

**Returns:** Dictionary with area, perimeter, axes, circularity, wavelength, etc.

## Scale Bar Detection (src.utils.scalebar_ocr)

### detect_scale_bar

```python
def detect_scale_bar(
    image: np.ndarray,
    roi_config: dict = None,
    intensity_threshold: int = 200,
    proximity_threshold: int = 50,
    dataset_name: str = None,
    draw_debug: bool = False
) -> tuple
```

Detect scale bar and extract scale information using OCR and Hough line detection.

**Parameters:**
- `image`: Input image (will be modified in-place if draw_debug=True)
- `roi_config`: ROI configuration dict (auto-loaded if None)
- `intensity_threshold`: Minimum intensity for scale bar lines (0-255)
- `proximity_threshold`: Max distance between text and line (pixels)
- `dataset_name`: Dataset name for loading dataset-specific config
- `draw_debug`: If True, draws debug visualizations on the image

**Returns:** Tuple of (scale_bar_length_str, microns_per_pixel)

**Debug Visualization:**

When `draw_debug=True`, the function draws directly on the input image:
- **Green box**: Scale bar ROI region
- **Blue box**: Detected OCR text location
- **Cyan lines**: All detected horizontal line candidates with metadata
- **Gray lines**: Lines rejected for being too close to ROI edge
- **Red thick line**: Selected scale bar line (or failure message)

**Example:**

```python
from src.utils.scalebar_ocr import detect_scale_bar
import cv2

image = cv2.imread("image.jpg")
scale, um_pix = detect_scale_bar(
    image, 
    dataset_name="my_dataset",
    draw_debug=True  # Enable debug visualization
)
cv2.imwrite("output_with_debug.jpg", image)
```

### get_scalebar_roi_for_dataset

```python
def get_scalebar_roi_for_dataset(dataset_name: str = None) -> dict
```

Load scale bar ROI configuration for specific dataset.

**Returns:** Dict with x_start_factor, y_start_factor, width_factor, height_factor

## GPU Check (src.utils.gpu_check)

### check_gpu_availability

```python
def check_gpu_availability(require_gpu: bool = False, interactive: bool = True) -> bool
```

Check if GPU/CUDA is available and optionally prompt user if not.

**Parameters:**
- `require_gpu`: If True, will exit if no GPU is available and user doesn't confirm
- `interactive`: If True, prompts user for confirmation. If False, just logs warning

**Returns:** True if GPU available or user confirmed to continue, False otherwise

### log_device_info

```python
def log_device_info() -> None
```

Log comprehensive GPU/CPU device information including model, VRAM, CUDA version.

### get_optimal_device

```python
def get_optimal_device(prefer_gpu: bool = True) -> torch.device
```

Get the optimal torch device based on availability.

**Parameters:**
- `prefer_gpu`: Prefer GPU if available

**Returns:** torch.device (cuda or cpu)

## Spatial Constraints (src.utils.spatial_constraints)

### apply_spatial_constraints

```python
def apply_spatial_constraints(
    masks: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    dataset_name: str = None
) -> tuple
```

Apply spatial constraint filtering (containment and overlap rules).

**Returns:** Filtered (masks, scores, classes)

## Safe File Operations (src.utils.safe_file_ops)

### validate_path_safety

```python
def validate_path_safety(
    file_path: Union[str, Path],
    allowed_base_dirs: List[Union[str, Path]]
) -> Path
```

Validate path is safe and within allowed directories. Prevents directory traversal.

## See Also

- [Data API](data.md)
- [Functions API](functions.md)
- [Configuration Reference](../configuration.md)
