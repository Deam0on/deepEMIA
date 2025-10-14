# User Guide

This guide covers day-to-day usage of deepEMIA for dataset management, model training, evaluation, and inference.

## Table of Contents

- [Interactive CLI Wizard](#interactive-cli-wizard)
- [Command Line Interface](#command-line-interface)
- [Web Interface](#web-interface)
- [Dataset Management](#dataset-management)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Running Inference](#running-inference)
- [Troubleshooting](#troubleshooting)

## Interactive CLI Wizard

The easiest way to use deepEMIA is through the interactive wizard:

```bash
python cli_main.py
```

The wizard provides:

- Guided task selection
- Dataset browsing from GCS
- Interactive parameter configuration
- Input validation and error recovery
- Progress tracking with ETA

### Wizard Features

- **Automatic dataset detection**: Lists available datasets from GCS
- **Smart defaults**: Suggests sensible parameter values
- **Error handling**: Retries and fallbacks for common issues
- **Progress tracking**: Real-time updates during long operations

## Command Line Interface

For automation and scripting, use the direct CLI:

```bash
python main.py --task <task> --dataset_name <name> [options]
```

### Core Arguments

- `--task` (required): Operation to perform
  - `setup`: Configure deepEMIA for first use
  - `prepare`: Split dataset into train/test sets
  - `train`: Train a model
  - `evaluate`: Evaluate model performance
  - `inference`: Run predictions on new images

- `--dataset_name` (required): Name of the dataset to use

### Model Configuration

- `--rcnn`: RCNN backbone architecture
  - `50`: ResNet-50 (faster, less accurate)
  - `101`: ResNet-101 (slower, more accurate)
  - `combo`: Both R50 and R101 (best accuracy)
  - Default: `101`

- `--threshold`: Confidence threshold for predictions (0.0-1.0)
  - Lower values: More detections, more false positives
  - Higher values: Fewer detections, fewer false positives
  - Default: `0.65`

- `--dataset_format`: Annotation format
  - `json`: Per-image JSON files (LabelMe style)
  - `coco`: COCO format JSON
  - Default: `json`

### Training Options

- `--augment`: Enable data augmentation
  - Adds flips, rotations, brightness adjustments
  - Improves model robustness
  
- `--optimize`: Run hyperparameter optimization with Optuna
  - Automatically finds best parameters
  - Results saved to config.yaml
  
- `--n-trials`: Number of optimization trials (default: 10)
  - More trials = better results but longer time
  - Recommended: 20-50 for production models

### Inference Options

- `--visualize`: Save visualization images with predictions
  - Creates overlay images with detected masks
  - Useful for quality checking

- `--id`: Draw instance IDs on visualizations
  - Numbers each detected instance
  - Helps with debugging and analysis

- `--pass`: Inference mode
  - `single`: One pass per image (fast)
  - `multi <N>`: Multiple passes with deduplication (accurate)
    - Default max iterations: 10
    - Early stopping based on config settings

### Data Transfer

- `--download`: Download data from GCS before task (default: true)
- `--upload`: Upload results to GCS after task (default: true)

## Web Interface

Launch the Streamlit web interface:

```bash
streamlit run gui/streamlit_gui.py
```

The web interface provides:

- Visual dataset management
- Point-and-click training configuration
- Real-time progress monitoring
- Result visualization and download
- Image gallery browsing

### Web Interface Features

- **Dataset Upload**: Drag-and-drop dataset upload to GCS
- **Training Dashboard**: Monitor training progress and metrics
- **Result Browser**: View and download predictions
- **CSV Export**: Download measurement data
- **Visualization Gallery**: Browse detection overlays

## Dataset Management

### Dataset Structure

Organize your data as follows:

```text
DATASET/
├── your_dataset_name/
│   ├── image1.tif
│   ├── image1.json
│   ├── image2.tif
│   ├── image2.json
│   └── ...
└── INFERENCE/
    ├── test_image1.tif
    ├── test_image2.tif
    └── ...
```

### Annotation Format

JSON annotation files (LabelMe style):

```json
{
  "imagePath": "image1.tif",
  "imageHeight": 1024,
  "imageWidth": 1024,
  "shapes": [
    {
      "label": "particle",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "polygon"
    }
  ]
}
```

### Preparing a Dataset

Split dataset into train/test sets:

```bash
python main.py --task prepare --dataset_name polyhipes
```

This creates:
- Train set (80% of images)
- Test set (20% of images)
- COCO-format annotations
- Category mapping JSON

## Model Training

### Basic Training

Train with default parameters:

```bash
python main.py --task train --dataset_name polyhipes --rcnn 101
```

### Training with Augmentation

Enable data augmentation for better generalization:

```bash
python main.py --task train --dataset_name polyhipes --rcnn 101 --augment
```

### Hyperparameter Optimization

Automatically find best parameters:

```bash
python main.py --task train --dataset_name polyhipes --rcnn combo --optimize --n-trials 20
```

Optimization searches for:
- Learning rate
- Batch size
- Warmup iterations
- Learning rate decay factor

Results are saved to `config.yaml` under `rcnn_hyperparameters.best_<dataset_name>`.

### Training Output

Training produces:
- `model_final.pth`: Final model weights
- `model_quantized.pth`: CPU-optimized quantized model
- Training logs in `~/logs/`
- Metrics and evaluation results

## Model Evaluation

### Basic Evaluation

Evaluate model on test set:

```bash
python main.py --task evaluate --dataset_name polyhipes --rcnn 101
```

### Evaluation with Visualization

Save prediction visualizations:

```bash
python main.py --task evaluate --dataset_name polyhipes --visualize
```

### Evaluation Metrics

The evaluation reports:
- Average Precision (AP) at different IoU thresholds
- AP for small, medium, and large objects
- Average Recall (AR)
- Per-class metrics

Results are saved to `output/` directory.

## Running Inference

### Single-Pass Inference

Fast inference for simple cases:

```bash
python main.py --task inference --dataset_name polyhipes --threshold 0.7
```

### Multi-Pass Inference

Iterative inference with deduplication for better accuracy:

```bash
python main.py --task inference --dataset_name polyhipes --pass multi 5 --threshold 0.65
```

Features:
- Multiple inference passes at different scales
- Automatic deduplication of overlapping masks
- Early stopping when no new detections found
- Configurable stopping criteria

### Inference with Visualization

Generate overlay images:

```bash
python main.py --task inference --dataset_name polyhipes --visualize --id
```

### Inference Output

Inference produces:
- `results.csv`: Aggregated measurements for all images
- `*_measurements.csv`: Per-image detailed measurements
- `*_overlay.png`: Visualization images (if --visualize)
  - Includes scale bar detection debug info (ROI, detected lines, selected scale bar)
- Individual mask files (optional)

### Measurement Columns

The CSV files contain:

- **Geometric Measurements**:
  - `Area_um2`: Particle area in square micrometers
  - `Perimeter_um`: Perimeter length
  - `Major_Axis_um`: Length of major axis
  - `Minor_Axis_um`: Length of minor axis
  - `Aspect_Ratio`: Major/minor axis ratio
  - `Circularity`: Shape circularity (0-1)

- **Physical Properties**:
  - `Wavelength_nm`: Estimated wavelength from color
  - `Confidence`: Detection confidence score

- **Metadata**:
  - `Image_Name`: Source image filename
  - `Instance_ID`: Unique instance identifier
  - `Class`: Predicted class label

## Troubleshooting

### Common Issues

**1. "Configuration not found" error**

Solution: Run setup first:

```bash
python cli_main.py  # Select "setup"
```

**2. "Dataset not found" error**

Solution: Ensure dataset exists in GCS:

```bash
gsutil ls gs://your-bucket/DATASET/
```

**3. CUDA out of memory during training**

Solutions:
- Reduce batch size in config.yaml
- Use R50 instead of R101
- Enable CPU training (automatic fallback)

**4. No detections in inference**

Solutions:
- Lower confidence threshold: `--threshold 0.5`
- Use multi-pass inference: `--pass multi 10`
- Check if model was trained on similar images

**5. Scale bar detection fails**

Solutions:

1. **Adjust ROI position** if scale bar is not in default location:
   ```yaml
   scale_bar_rois:
     your_dataset:
       x_start_factor: 0.7  # Adjust horizontal position
       y_start_factor: 0.05  # Adjust vertical position
       width_factor: 1.0     # Adjust width
       height_factor: 0.05   # Adjust height
   ```

2. **Adjust detection thresholds** in config.yaml:
   ```yaml
   scalebar_thresholds:
     intensity: 100      # Lower if scale bar is dark
     proximity: 100      # Increase if text is far from line
     merge_gap: 15       # Increase if scale bar is fragmented
     min_line_length: 30 # Adjust for your image resolution
   ```

3. **Check debug visualization** in output images:
   - Green box shows ROI region
   - Blue box shows detected text
   - Cyan/gray lines show line candidates
   - Red line shows selected scale bar
   - Use this to diagnose detection issues

4. **Common issues and fixes**:
   - **No lines detected**: Increase ROI `height_factor` or reduce `intensity` threshold
   - **Wrong line selected**: Adjust `proximity` or `edge_margin_factor`
   - **Fragmented scale bar**: Increase `merge_gap`
   - **Text not detected**: Adjust ROI position or check OCR requirements

### Log Files

Check logs for detailed error information:

```bash
# View recent system logs
cat ~/logs/system_*.log | tail -n 100

# View training logs
cat ~/logs/training_*.log | tail -n 100
```

### Performance Issues

**Slow training:**
- Use GPU if available
- Reduce image resolution
- Decrease number of iterations
- Use R50 instead of R101

**Slow inference:**
- Use single-pass mode
- Increase confidence threshold
- Use quantized model on CPU
- Process images in batches

### Getting More Help

1. Check the [Examples](examples/basic-workflow.md) for working workflows
2. Review the [Configuration Reference](configuration.md)
3. Search [existing GitHub issues](https://github.com/Deam0on/deepEMIA/issues)
4. Open a new issue with:
   - Error message from logs
   - Command that caused the issue
   - System information (OS, Python version, GPU)

## Advanced Usage

### Batch Processing

Process multiple datasets:

```bash
for dataset in dataset1 dataset2 dataset3; do
  python main.py --task inference --dataset_name $dataset --threshold 0.7
done
```

### Custom Configuration

Override config settings per dataset:

1. Add dataset-specific config in config.yaml
2. Use dataset-specific hyperparameters
3. Configure class-specific inference settings

See [Configuration Reference](configuration.md) for details.

### Integration with Other Tools

Export results for external analysis:

```python
import pandas as pd

# Load inference results
df = pd.read_csv('results.csv')

# Filter by confidence
high_conf = df[df['Confidence'] > 0.8]

# Calculate statistics
mean_area = df['Area_um2'].mean()
```

## Best Practices

1. **Always run prepare before training** to ensure proper dataset splits
2. **Use hyperparameter optimization** for best model performance
3. **Validate results** by checking visualizations
4. **Archive successful models** to GCS for reproducibility
5. **Monitor logs** during long operations
6. **Start with lower thresholds** and adjust based on results
7. **Use multi-pass inference** for critical analysis
8. **Back up configuration** before making changes
