# Custom Datasets

Guide to preparing and using custom datasets with deepEMIA.

## Dataset Requirements

### Image Format

Supported formats:
- TIFF (.tif, .tiff)
- PNG (.png)
- JPEG (.jpg, .jpeg)

### Annotation Format

Use LabelMe-style JSON annotations:

```json
{
  "imagePath": "image001.tif",
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

## Preparing Your Dataset

### 1. Organize Files

Structure your dataset:

```text
my_dataset/
├── image001.tif
├── image001.json
├── image002.tif
├── image002.json
└── ...
```

### 2. Upload to GCS

```bash
gsutil -m cp -r my_dataset/ gs://your-bucket/DATASET/
```

### 3. Update dataset_info.json

Create or update dataset info:

```json
{
  "my_dataset": {
    "dataset_dir": "my_dataset",
    "categories": ["particle"],
    "num_classes": 1
  }
}
```

Upload to GCS:

```bash
gsutil cp dataset_info.json gs://your-bucket/
```

### 4. Configure Scale Bar Detection

Add dataset-specific ROI to config.yaml:

```yaml
scale_bar_rois:
  my_dataset:
    x_start_factor: 0.7
    y_start_factor: 0.05
    width_factor: 1.0
    height_factor: 0.05
```

## Multi-Class Datasets

For multiple classes:

```json
{
  "shapes": [
    {"label": "large_particle", "points": [...], "shape_type": "polygon"},
    {"label": "small_particle", "points": [...], "shape_type": "polygon"}
  ]
}
```

Update dataset_info.json:

```json
{
  "my_dataset": {
    "categories": ["large_particle", "small_particle"],
    "num_classes": 2
  }
}
```

Configure class-specific inference:

```yaml
inference_settings:
  class_specific_settings:
    class_0:  # large_particle
      confidence_threshold: 0.5
      min_size: 50
    class_1:  # small_particle
      confidence_threshold: 0.3
      min_size: 10
      use_multiscale: true
```

## Testing Your Dataset

Prepare and verify:

```bash
python main.py --task prepare --dataset_name my_dataset
```

Train a quick test model:

```bash
python main.py --task train --dataset_name my_dataset --rcnn 50
```

## See Also

- [Basic Workflow](basic-workflow.md)
- [Configuration Reference](../configuration.md)
- [User Guide](../user-guide.md)
