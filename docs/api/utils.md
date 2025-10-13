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
    dataset_name: str = None
) -> tuple
```

Detect scale bar and extract scale information.

**Returns:** Tuple of (micrometers_per_pixel, scale_value, unit, roi_coords)

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
