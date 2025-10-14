# Training Pipeline

Detailed architecture of the model training process.

## Training Workflow

1. **Dataset Preparation**: Split and register dataset
2. **Configuration**: Load hyperparameters from config
3. **Augmentation**: Optional data augmentation
4. **Training**: Train with Detectron2
5. **Optimization**: Optional hyperparameter tuning
6. **Quantization**: Create CPU-optimized model
7. **Evaluation**: Assess model performance

## Hyperparameter Optimization

Uses Optuna to search for best parameters:

- Learning rate
- Batch sizes
- Warmup iterations
- Learning rate decay

Results saved to config for future use.

## See Also

- [Pipeline Overview](pipeline-overview.md)
- [API Reference](../api/functions.md)
