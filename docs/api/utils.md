# Utilities Module API

API reference for utility functions.

## Configuration (src.utils.config)

### get_config

```python
def get_config() -> dict
```

Load and return project configuration. Configuration is loaded once and cached.

**Returns:** Configuration dictionary

**Raises:** 
- `FileNotFoundError`: Config file not found
- `yaml.YAMLError`: Invalid YAML
- `ConfigValidationError`: Invalid configuration

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
