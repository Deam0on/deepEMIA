# Inference Pipeline

Architecture of the inference and measurement extraction process.

## Inference Modes

### Single-Pass Inference

Fast, single iteration per image.

### Multi-Pass Inference

Iterative processing with deduplication:
1. Initial prediction
2. Remove detected regions
3. Predict on remaining areas
4. Merge and deduplicate
5. Repeat until stopping criteria met

## Post-Processing Steps

1. **Mask Filtering**: Remove overlaps and small artifacts
2. **Spatial Constraints**: Apply containment/overlap rules
3. **Scale Detection**: Extract scale bar information
4. **Measurement**: Calculate geometric properties
5. **CSV Export**: Write results to file

## Class-Specific Inference

Different thresholds and processing per class:
- Confidence thresholds
- IoU thresholds
- Minimum sizes
- Multi-scale processing

## See Also

- [Pipeline Overview](pipeline-overview.md)
- [Configuration](../configuration.md)
