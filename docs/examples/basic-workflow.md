# Basic Workflow

Complete walkthrough of a basic deepEMIA workflow.

## Scenario

Train a model on a new dataset and run inference on test images.

## Prerequisites

- deepEMIA installed and configured
- GCS bucket set up
- Dataset uploaded to GCS

## Step 1: Prepare Dataset

Split dataset into train/test sets:

```bash
python main.py --task prepare --dataset_name polyhipes
```

This creates:
- Training set (80% of images)
- Test set (20% of images)
- Category mapping

## Step 2: Train Model

Train with default parameters:

```bash
python main.py --task train --dataset_name polyhipes --rcnn 101
```

For better results, use augmentation:

```bash
python main.py --task train --dataset_name polyhipes --rcnn 101 --augment
```

Training takes 30-60 minutes depending on dataset size and hardware.

## Step 3: Evaluate Model

Check model performance on test set:

```bash
python main.py --task evaluate --dataset_name polyhipes --visualize
```

Review metrics in output directory and visualizations.

## Step 4: Run Inference

Process new images:

```bash
python main.py --task inference --dataset_name polyhipes --threshold 0.7 --visualize
```

Results are saved to:
- `results.csv`: All measurements
- `*_overlay.png`: Visualization images

## Step 5: Review Results

Open results.csv in Excel or analyze with Python:

```python
import pandas as pd

df = pd.read_csv('results.csv')
print(df.describe())
print(f"Total particles: {len(df)}")
print(f"Average area: {df['Area_um2'].mean():.2f} µm²")
```

## Troubleshooting

**No detections**: Lower threshold to 0.5 or 0.4

**Too many false positives**: Increase threshold to 0.8

**Missing small particles**: Iterative inference is now automatic. Configure stopping criteria in `config.yaml`:

```yaml
inference_settings:
  iterative_stopping:
    min_iterations: 3
    min_relative_increase: 0.1
```

## See Also

- [Hyperparameter Tuning](hyperparameter-tuning.md)
- [Custom Datasets](custom-datasets.md)
- [User Guide](../user-guide.md)
