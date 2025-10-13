# Configuration Reference

This document provides a complete reference for the `config.yaml` configuration file.

## Configuration File Location

The configuration file is located at:

```text
~/deepEMIA/config/config.yaml
```

## Basic Structure

```yaml
bucket: your-gcs-bucket-name

paths:
  # Path definitions

scale_bar_rois:
  # Scale bar detection settings

scalebar_thresholds:
  # Detection thresholds

measure_contrast_distribution: false

rcnn_hyperparameters:
  # Model training parameters

inference_settings:
  # Inference configuration

l4_performance_optimizations:
  # GPU performance tuning
```

## Configuration Sections

### Bucket Configuration

```yaml
bucket: your-gcs-bucket-name
```

Specifies the Google Cloud Storage bucket for data storage and model sharing.

- **Type**: String
- **Required**: Yes
- **Example**: `nn-uct`

### Paths Configuration

```yaml
paths:
  main_script: "~/deepEMIA/main.py"
  split_dir: "~/split_dir"
  category_json: "~/deepEMIA/dataset_info.json"
  eta_file: "~/deepEMIA/config/eta_data.json"
  logs_dir: "~/logs"
  output_dir: "~/deepEMIA/output"
  local_dataset_root: "~"
```

Defines file system paths used by the pipeline.

- `main_script`: Path to main.py script
- `split_dir`: Directory for dataset splits
- `category_json`: Dataset category information
- `eta_file`: ETA tracking data file
- `logs_dir`: Log file directory
- `output_dir`: Output directory for results
- `local_dataset_root`: Root directory for local datasets

All paths support tilde (`~`) expansion for home directory.

### Scale Bar ROI Configuration

```yaml
scale_bar_rois:
  default:
    x_start_factor: 0.7
    y_start_factor: 0.05
    width_factor: 1.0
    height_factor: 0.05
  
  dataset_specific_name:
    x_start_factor: 0.5
    y_start_factor: 0.1
    width_factor: 0.9
    height_factor: 0.1
```

Defines regions of interest (ROI) for scale bar detection in images.

**Default Settings:**
- `x_start_factor`: Horizontal start position (0.0-1.0, fraction of image width)
- `y_start_factor`: Vertical start position (0.0-1.0, fraction of image height)
- `width_factor`: ROI width (0.0-1.0, fraction of image width)
- `height_factor`: ROI height (0.0-1.0, fraction of image height)

**Dataset-Specific Overrides:**

Add dataset-specific ROI configurations to override defaults for particular datasets. The dataset name must match the folder name in GCS.

**Example:**

```yaml
scale_bar_rois:
  default:
    x_start_factor: 0.7
    y_start_factor: 0.05
    width_factor: 1.0
    height_factor: 0.05
  
  polyhipes_tommy:  # Custom ROI for this dataset
    x_start_factor: 0.5
    y_start_factor: 0.1
    width_factor: 0.9
    height_factor: 0.1
```

### Scale Bar Thresholds

```yaml
scalebar_thresholds:
  intensity: 100
  proximity: 50
```

Thresholds for scale bar detection algorithm.

- `intensity`: Minimum pixel intensity for scale bar elements (0-255)
  - Lower values: Detect darker scale bars
  - Higher values: Only detect bright scale bars
  - Default: 100

- `proximity`: Maximum pixel distance for grouping scale bar elements
  - Lower values: Stricter grouping
  - Higher values: More lenient grouping
  - Default: 50

### Measurement Configuration

```yaml
measure_contrast_distribution: false
```

Enable or disable contrast distribution measurement for detected particles.

- **Type**: Boolean
- **Default**: false
- **When to enable**: Detailed particle analysis requiring contrast data
- **Performance impact**: Increases processing time

### RCNN Hyperparameters

```yaml
rcnn_hyperparameters:
  default:
    R50:
      base_lr: 0.00025
      ims_per_batch: 2
      warmup_iters: 1000
      warmup_factor: 0.001
      gamma: 0.1
      batch_size_per_image: 64
      max_iter: 10000
    R101:
      # Same structure as R50
  
  best:
    R50: {}
    R101: {}
  
  best_dataset_name:
    R50:
      # Optimized parameters for specific dataset
    R101:
      # Optimized parameters for specific dataset
```

Training hyperparameters for RCNN models.

**Parameter Hierarchy:**

1. Dataset-specific best (`best_<dataset_name>`) - highest priority
2. Global best (`best`) - fallback if no dataset-specific
3. Default (`default`) - ultimate fallback

**Parameters:**

- `base_lr`: Base learning rate (typical: 0.00025)
- `ims_per_batch`: Number of images per batch (typical: 2-4)
- `warmup_iters`: Number of warmup iterations (typical: 1000)
- `warmup_factor`: Learning rate warmup factor (typical: 0.001)
- `gamma`: Learning rate decay factor (typical: 0.1)
- `batch_size_per_image`: ROIs per image (typical: 64-256)
- `max_iter`: Maximum training iterations (calculated automatically if not specified)

**Automatic Optimization:**

When running with `--optimize`, the system automatically:
1. Searches for best parameters using Optuna
2. Saves results to `best_<dataset_name>`
3. Uses these parameters for future training on the same dataset

### Inference Settings

```yaml
inference_settings:
  use_class_specific_inference: true
  
  iterative_stopping:
    min_total_masks: 10
    min_relative_increase: 0.25
    max_consecutive_zero: 1
    min_iterations: 2
  
  class_specific_settings:
    class_0:
      confidence_threshold: 0.5
      iou_threshold: 0.7
      min_size: 25
    class_1:
      confidence_threshold: 0.3
      iou_threshold: 0.5
      min_size: 5
      use_multiscale: true
```

Configuration for inference behavior.

**Global Settings:**

- `use_class_specific_inference`: Enable class-specific processing
  - Type: Boolean
  - Default: true
  - Allows different thresholds per class

**Iterative Stopping Criteria:**

Controls when multi-pass inference terminates:

- `min_total_masks`: Minimum total masks before considering early stop
  - Type: Integer
  - Default: 10
  
- `min_relative_increase`: Minimum relative increase in masks (0.0-1.0)
  - Type: Float
  - Default: 0.25 (25% increase required)
  - Example: If 100 masks exist, need 25 new masks to continue
  
- `max_consecutive_zero`: Maximum iterations with zero new masks
  - Type: Integer
  - Default: 1
  - Stop after this many iterations without new detections
  
- `min_iterations`: Always run at least this many iterations
  - Type: Integer
  - Default: 2

**Class-Specific Settings:**

Define per-class inference parameters:

- `confidence_threshold`: Minimum confidence for this class (0.0-1.0)
- `iou_threshold`: IoU threshold for duplicate removal (0.0-1.0)
- `min_size`: Minimum mask size in pixels
- `use_multiscale`: Enable multi-scale inference for this class (Boolean)

**Example: Two-Class Configuration**

```yaml
class_specific_settings:
  class_0:  # Large particles
    confidence_threshold: 0.5
    iou_threshold: 0.7
    min_size: 25
  class_1:  # Small particles
    confidence_threshold: 0.3  # Lower threshold for small objects
    iou_threshold: 0.5
    min_size: 5
    use_multiscale: true  # Better detection for small objects
```

### Spatial Constraints

```yaml
spatial_constraints:
  dataset_name:
    containment_rules:
      - child_class: 1
        parent_class: 0
        containment_threshold: 0.95
    
    overlap_rules:
      - class_a: 0
        class_b: 1
        max_iou: 0.3
```

Define spatial relationships between classes (optional).

**Containment Rules:**

Enforce that one class must be contained within another:

- `child_class`: Class that must be inside
- `parent_class`: Class that must contain
- `containment_threshold`: Fraction of child that must be inside parent (0.0-1.0)

**Overlap Rules:**

Limit overlap between classes:

- `class_a`: First class
- `class_b`: Second class
- `max_iou`: Maximum allowed IoU between classes (0.0-1.0)

**Example Use Case:**

Cells (class 0) containing nuclei (class 1):

```yaml
spatial_constraints:
  cells_dataset:
    containment_rules:
      - child_class: 1  # nuclei
        parent_class: 0  # cells
        containment_threshold: 0.9  # 90% of nucleus must be inside cell
    
    overlap_rules:
      - class_a: 0  # cells
        class_b: 0  # cells
        max_iou: 0.2  # cells shouldn't overlap much
```

### L4 GPU Performance Optimizations

```yaml
l4_performance_optimizations:
  batch_processing:
    inference_batch_size: 4
    dataloader_workers: 2
    prefetch_factor: 2
  
  memory_management:
    max_image_size: 2048
    enable_memory_efficient_mode: true
    cleanup_individual_masks: true
```

Optimizations specifically tuned for NVIDIA L4 GPUs (g2-standard-4 instances).

**Batch Processing:**

- `inference_batch_size`: Number of images to process simultaneously
- `dataloader_workers`: Number of data loading threads
- `prefetch_factor`: Images to prefetch per worker

**Memory Management:**

- `max_image_size`: Maximum image dimension before resizing
- `enable_memory_efficient_mode`: Reduce memory usage at slight speed cost
- `cleanup_individual_masks`: Delete individual mask files after processing

## Configuration Best Practices

### 1. Start with Defaults

Don't modify settings unless needed. The defaults work well for most cases.

### 2. Dataset-Specific Overrides

Use dataset-specific configurations for datasets with unique characteristics:

```yaml
scale_bar_rois:
  default:
    # Standard settings
  
  high_magnification_dataset:
    # Different ROI for this dataset
```

### 3. Hyperparameter Management

Let the optimizer find best parameters:

```bash
python main.py --task train --dataset_name mydata --optimize --n-trials 20
```

The results are automatically saved to config.yaml.

### 4. Backup Configuration

Before making changes:

```bash
cp ~/deepEMIA/config/config.yaml ~/deepEMIA/config/config.yaml.backup
```

### 5. Validate Configuration

After editing:

```bash
python -c "from src.utils.config import get_config; get_config(); print('Valid')"
```

## Environment Variables

Some sensitive settings use environment variables instead of config.yaml:

### ADMIN_PASSWORD_HASH

SHA256 hash of admin password for web interface:

```bash
# Linux/Mac
export ADMIN_PASSWORD_HASH=$(echo -n 'your_password' | sha256sum | cut -d' ' -f1)

# Windows PowerShell
$env:ADMIN_PASSWORD_HASH = 'your_hash_here'
```

### GOOGLE_APPLICATION_CREDENTIALS

Path to GCS service account key:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

## Configuration Examples

### High-Throughput Inference

For fast batch processing:

```yaml
inference_settings:
  use_class_specific_inference: false
  iterative_stopping:
    min_iterations: 1
    max_consecutive_zero: 1

l4_performance_optimizations:
  batch_processing:
    inference_batch_size: 8
    dataloader_workers: 4
```

### High-Accuracy Research

For maximum accuracy:

```yaml
inference_settings:
  use_class_specific_inference: true
  iterative_stopping:
    min_iterations: 3
    min_relative_increase: 0.1
    max_consecutive_zero: 2
  
  class_specific_settings:
    class_0:
      confidence_threshold: 0.4
      use_multiscale: true
```

### Low-Memory Systems

For systems with limited RAM:

```yaml
l4_performance_optimizations:
  batch_processing:
    inference_batch_size: 1
    dataloader_workers: 1
  
  memory_management:
    max_image_size: 1024
    enable_memory_efficient_mode: true
    cleanup_individual_masks: true
```

## Troubleshooting Configuration Issues

### Configuration Not Loading

Check file location:

```bash
ls ~/deepEMIA/config/config.yaml
```

### Invalid YAML Syntax

Validate YAML:

```bash
python -c "import yaml; yaml.safe_load(open('~/deepEMIA/config/config.yaml'))"
```

### Settings Not Taking Effect

1. Restart the application after config changes
2. Check for dataset-specific overrides
3. Verify parameter names match documentation exactly

### Performance Issues

If experiencing performance problems:

1. Reduce `inference_batch_size`
2. Increase `max_consecutive_zero` for faster stopping
3. Disable `measure_contrast_distribution`
4. Lower `dataloader_workers`

## See Also

- [User Guide](user-guide.md) - How to use the configuration in practice
- [Examples](examples/basic-workflow.md) - Configuration examples for common scenarios
- [Architecture](architecture/pipeline-overview.md) - How configuration affects the pipeline
