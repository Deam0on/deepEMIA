# Hyperparameter Tuning

Guide to optimizing model performance with hyperparameter tuning.

## Overview

Hyperparameter optimization automatically finds the best training parameters for your dataset.

## Basic Optimization

Run optimization with default settings:

```bash
python main.py --task train --dataset_name polyhipes --optimize --n-trials 20
```

This searches for optimal:
- Learning rate
- Batch size
- Warmup iterations
- Learning rate decay factor

## Advanced Optimization

For best results, use more trials and augmentation:

```bash
python main.py --task train --dataset_name polyhipes --rcnn combo --optimize --n-trials 50 --augment
```

Using `--rcnn combo` optimizes both R50 and R101 backbones.

## Understanding Results

After optimization, check `config.yaml`:

```yaml
rcnn_hyperparameters:
  best_polyhipes:
    R50:
      base_lr: 0.0001  # Optimized value
      ims_per_batch: 4
      # ... other optimized parameters
```

These parameters are automatically used for future training on this dataset.

## Optimization Tips

- Start with 20 trials for quick results
- Use 50+ trials for production models
- Enable augmentation for better generalization
- Optimize on representative data

## Manual Tuning

Edit config.yaml directly for manual tuning:

```yaml
rcnn_hyperparameters:
  best_mydataset:
    R101:
      base_lr: 0.0002  # Manually set
      ims_per_batch: 2
```

## See Also

- [Basic Workflow](basic-workflow.md)
- [Configuration Reference](../configuration.md)
