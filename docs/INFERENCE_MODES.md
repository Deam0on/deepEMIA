# Inference Mode Selection Feature

## Overview

The inference pipeline now supports three distinct inference modes that give you flexibility in how you process and organize results:

1. **all_classes** (default) - Process all classes together in one output
2. **single_class** - Process only specific class(es)
3. **separate_classes** - Process each class separately with individual output directories

## Usage

### CLI Interface

When running inference via the CLI wizard (`python cli_main.py`):

1. Select "Inference" task
2. Choose your dataset
3. **NEW:** Select inference mode:
   - `all_classes (process all together)` - Default behavior
   - `single_class (specific classes only)` - Prompts for class selection
   - `separate_classes (individual outputs)` - Processes each class separately
4. If `single_class` is selected:
   - CLI displays available classes with indices
   - Enter comma-separated class indices (e.g., `0,2` or `1`)

### Command Line Interface

```bash
# Default: Process all classes together
python main.py --task inference --dataset_name my_dataset

# Process only specific classes (e.g., class 0 and class 2)
python main.py --task inference --dataset_name my_dataset \
    --inference_mode single_class --target_classes 0,2

# Process each class separately
python main.py --task inference --dataset_name my_dataset \
    --inference_mode separate_classes

# Process only class 1 separately (useful for large datasets)
python main.py --task inference --dataset_name my_dataset \
    --inference_mode separate_classes --target_classes 1
```

## Modes Explained

### All Classes Mode (default)

**When to use:** Standard workflow, when you want all detections in one place

**Behavior:**
- Processes all classes together
- Single CSV file: `measurements_results.csv`
- Single results file: `R50_flip_results.csv`
- Visualizations show all classes with different colors
- Standard output directory structure

**Example output:**
```
output/
├── measurements_results.csv (all classes)
├── R50_flip_results.csv
└── image001_predictions.png (all classes shown)
```

### Single Class Mode

**When to use:** 
- Focus analysis on specific particle types
- Reduce processing time by skipping unwanted classes
- Debug/optimize detection for particular classes

**Behavior:**
- Processes only specified classes (via `--target_classes`)
- Single CSV with only selected classes
- Faster processing (skips other classes)
- Visualizations show only selected classes
- Same output directory structure

**Example output:**
```
output/
├── measurements_results.csv (only class 0 and class 2)
├── R50_flip_results.csv
└── image001_predictions.png (only classes 0,2 shown)
```

### Separate Classes Mode

**When to use:**
- Need independent analysis per class
- Different post-processing pipelines per class
- Separate QA/review workflows for each class type
- Large datasets where you want to process classes in stages

**Behavior:**
- Creates subdirectory for each class: `class_{idx}_{name}/`
- Each subdirectory has complete, independent results
- Processes classes sequentially (one at a time)
- Each class gets its own CSV, results, and visualizations
- Can specify subset of classes with `--target_classes`

**Example output:**
```
output/
├── class_0_particles/
│   ├── measurements_results.csv (only class 0)
│   ├── R50_flip_results.csv
│   └── image001_predictions.png (only class 0)
└── class_1_aggregates/
    ├── measurements_results.csv (only class 1)
    ├── R50_flip_results.csv
    └── image001_predictions.png (only class 1)
```

## Implementation Details

### Function Signature Changes

**src/functions/inference.py:**
```python
def run_inference(
    dataset_name,
    output_dir,
    visualize=True,
    threshold=0.65,
    draw_id=False,
    dataset_format="json",
    draw_scalebar=False,
    inference_mode="all_classes",  # NEW
    target_classes=None,  # NEW (list of int)
):
```

### Key Changes

1. **inference.py:**
   - Added mode validation and class filtering logic
   - `separate_classes` mode recursively calls `run_inference` for each class
   - `classes_to_process` list determines which classes to run inference on
   - Loop changed from `range(num_classes)` to `classes_to_process`

2. **cli_main.py:**
   - Added inference mode selection menu
   - Auto-loads metadata to show available classes for `single_class` mode
   - Constructs `--inference_mode` and `--target_classes` arguments

3. **main.py:**
   - Added `--inference_mode` argument with choices validation
   - Added `--target_classes` argument for comma-separated indices
   - Parses target_classes string to list of integers before passing to `run_inference`

## Backward Compatibility

✅ **Fully backward compatible**

- Default `inference_mode="all_classes"` maintains existing behavior
- All existing scripts/workflows continue to work unchanged
- Optional parameters only activate when explicitly set

## Additional Bug Fix

### Scalebar Debug Image Cleanup

**Issue:** When `--draw-scalebar` is enabled, debug images (`*_scalebar_debug.png`) were not cleaned up between runs.

**Fix:** Updated `cleanup_old_predictions()` to include scalebar debug pattern:
```python
# Now removes both:
prediction_files = list(path.glob("*_predictions.png")) + list(path.glob("*_scalebar_debug.png"))
```

## Testing Recommendations

### Test Cases

1. **Default behavior (no changes):**
   ```bash
   python main.py --task inference --dataset_name test_dataset
   ```
   Expected: All classes processed together as before

2. **Single class selection:**
   ```bash
   python main.py --task inference --dataset_name test_dataset \
       --inference_mode single_class --target_classes 0
   ```
   Expected: Only class 0 in results

3. **Separate classes:**
   ```bash
   python main.py --task inference --dataset_name test_dataset \
       --inference_mode separate_classes
   ```
   Expected: Multiple class_{idx}_{name} subdirectories

4. **Separate classes with subset:**
   ```bash
   python main.py --task inference --dataset_name test_dataset \
       --inference_mode separate_classes --target_classes 0,1
   ```
   Expected: Only class_0 and class_1 subdirectories

5. **Scalebar debug cleanup:**
   ```bash
   # Run twice with --draw-scalebar to verify cleanup
   python main.py --task inference --dataset_name test_dataset --draw-scalebar
   python main.py --task inference --dataset_name test_dataset --draw-scalebar
   ```
   Expected: Old scalebar debug images removed on second run

## Error Handling

- Invalid class indices are logged and skipped
- Empty `target_classes` after validation raises ValueError
- Invalid `inference_mode` raises ValueError with helpful message
- Metadata loading failures in CLI fall back to manual entry

## Performance Considerations

- **single_class mode:** Faster than all_classes (skips unwanted classes)
- **separate_classes mode:** Slower than all_classes (sequential processing with overhead)
- **Memory:** separate_classes processes one class at a time, potentially better for large datasets

## Future Enhancements

Potential improvements for future versions:

1. Parallel processing for `separate_classes` mode
2. Per-class threshold configuration
3. Class-specific output format options
4. Merged summary CSV across all classes in `separate_classes` mode
5. Progress bar showing class X of N in `separate_classes` mode
