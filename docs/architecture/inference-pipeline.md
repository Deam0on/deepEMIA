# Inference Pipeline

Architecture of the inference and measurement extraction process.

## Inference Modes

### Automatic Iterative Inference

The default inference mode uses intelligent stopping criteria:
1. Initial prediction with base threshold
2. Apply spatial constraints and filtering
3. Check stopping criteria (min masks, relative increase, consecutive zeros)
4. Continue with next iteration or terminate
5. Merge and deduplicate across all iterations

Configuration via `config.yaml`:
```yaml
inference_settings:
  iterative_stopping:
    min_total_masks: 10
    min_relative_increase: 0.25
    max_consecutive_zero: 1
    min_iterations: 2
```

### Tile-Based Inference

For large images, tile-based processing:
1. Split image into overlapping tiles
2. Upscale tiles for better detection
3. Run inference on each tile
4. Merge predictions with edge filtering
5. Deduplicate overlapping detections

### Model Ensemble

Multi-model ensemble for improved accuracy:
- Combines predictions from R50 and R101 models
- Weighted averaging of confidence scores
- Optional: ensemble for small classes only

## Post-Processing Steps

1. **Mask Filtering**: Remove overlaps and small artifacts
2. **Spatial Constraints**: Apply containment/overlap rules
3. **Scale Detection**: Extract scale bar information with OCR
4. **Measurement**: Calculate geometric properties
5. **CSV Export**: Write results incrementally

## Class-Specific Inference

Different thresholds and processing per class:
- Confidence thresholds
- IoU thresholds
- Minimum sizes and size factors
- Multi-scale processing for small objects

## L4 GPU Optimizations

Performance tuning for NVIDIA L4:
- Mixed precision inference
- Parallel image loading
- Parallel mask processing
- Memory-efficient cache management
- Streaming CSV output

## See Also

- [Pipeline Overview](pipeline-overview.md)
- [Configuration](../configuration.md)
