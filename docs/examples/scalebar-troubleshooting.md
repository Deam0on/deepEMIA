# Scale Bar Detection Troubleshooting Guide

This guide helps you diagnose and fix scale bar detection issues using the built-in debug visualization.

## Overview

Scale bar detection uses:
1. **OCR** to detect the scale bar text (e.g., "100 µm")
2. **Hough line detection** to find horizontal lines
3. **Proximity matching** to associate text with the correct line
4. **Edge filtering** to remove noise at ROI boundaries

## Debug Visualization

When inference runs, debug visualizations are automatically drawn on output images showing:

- **Green box**: Scale bar ROI region where detection occurs
- **Blue box**: Detected OCR text location
- **Cyan lines**: All detected horizontal line candidates with metadata
- **Gray lines**: Lines rejected for being too close to ROI edges
- **Red thick line**: The selected scale bar line
- **Red text**: "SCALE BAR DETECTION FAILED" if no valid line found

## Common Problems and Solutions

### Problem 1: "No text detected by OCR"

**Symptoms:**
- Log shows: "No text detected by OCR in scale bar ROI"
- No blue box in debug visualization

**Solutions:**

1. **Adjust ROI position** to include the text:
   ```yaml
   scale_bar_rois:
     your_dataset:
       x_start_factor: 0.6  # Move ROI left (lower value)
       y_start_factor: 0.9  # Move ROI down (higher value)
       width_factor: 0.4    # Make ROI narrower
       height_factor: 0.08  # Make ROI taller
   ```

2. **Check image quality**: OCR requires clear, high-resolution text

### Problem 2: "No scale bar line detected near OCR text"

**Symptoms:**
- Log shows: "No scale bar line detected near OCR text"
- Blue box visible but no red line selected
- May show cyan/gray lines but none are selected

**Debug Analysis:**

Check the log output for line candidates:
```
Top 5 line candidates:
  1. Length: 276.0px ✓, Intensity: 124.6 ✓, Distance: 80.3px ✗, Edge: ✓
```

Each line shows which checks passed (✓) or failed (✗):
- **Length**: Must be > `min_line_length` (default 30px)
- **Intensity**: Must be > `intensity` threshold (default 100)
- **Distance**: Must be < `proximity` threshold (default 100px)
- **Edge**: Must not be within 10% of ROI edges

**Solutions:**

1. **Text too far from line** (Distance: ✗):
   ```yaml
   scalebar_thresholds:
     proximity: 150  # Increase from default 100
   ```

2. **Scale bar too dark** (Intensity: ✗):
   ```yaml
   scalebar_thresholds:
     intensity: 80  # Decrease from default 100
   ```

3. **Line too short** (Length: ✗):
   ```yaml
   scalebar_thresholds:
     min_line_length: 20  # Decrease from default 30
   ```

4. **Line too close to edge** (Edge: ✗):
   ```yaml
   scalebar_thresholds:
     edge_margin_factor: 0.05  # Decrease from default 0.1
   ```

### Problem 3: Scale bar detected as multiple segments

**Symptoms:**
- Multiple short cyan lines instead of one long line
- Log shows: "Possible scale bar segmentation: Two similar lines detected"

**Solution:**

Increase the merge gap to join segments:
```yaml
scalebar_thresholds:
  merge_gap: 25  # Increase from default 15
```

### Problem 4: Wrong line selected

**Symptoms:**
- Red line drawn on wrong element (not the scale bar)
- Multiple cyan lines visible

**Solutions:**

1. **Tighten proximity** to associate text with correct line:
   ```yaml
   scalebar_thresholds:
     proximity: 60  # Reduce from default 100
   ```

2. **Increase minimum length** to filter out short noise lines:
   ```yaml
   scalebar_thresholds:
     min_line_length: 50  # Increase from default 30
   ```

3. **Adjust ROI** to exclude non-scale-bar elements:
   ```yaml
   scale_bar_rois:
     your_dataset:
       x_start_factor: 0.75  # Focus on bottom-right corner
       y_start_factor: 0.92  # Only bottom of image
       width_factor: 0.25    # Narrow width
       height_factor: 0.06   # Minimal height
   ```

### Problem 5: Scale bar at image edge gets filtered

**Symptoms:**
- Gray line drawn at correct scale bar location
- Line marked with [EDGE] in log
- Log shows: "Line too close to edge, skipping"

**Solution:**

Reduce edge margin to allow lines near edges:
```yaml
scalebar_thresholds:
  edge_margin_factor: 0.0  # Disable edge filtering (use carefully)
```

Or adjust ROI to better frame the scale bar:
```yaml
scale_bar_rois:
  your_dataset:
    y_start_factor: 0.88  # Adjust to center scale bar in ROI
    height_factor: 0.10   # Increase height for more margin
```

## Step-by-Step Troubleshooting Workflow

1. **Run inference** and check the output visualization images

2. **Identify the issue** using the color-coded debug info:
   - No green box → ROI not configured
   - Green box in wrong place → Adjust ROI position
   - No blue box → OCR failed, adjust ROI or check image quality
   - Blue box but no lines → Adjust intensity threshold or ROI height
   - Cyan lines but none selected → Check distance, intensity, length, edge criteria
   - Gray lines → Lines filtered by edge check
   - Wrong red line → Adjust proximity or minimum length

3. **Check the logs** for detailed numeric information:
   ```bash
   tail -100 ~/logs/system_*.log | grep -A 10 "scale bar"
   ```

4. **Adjust configuration** based on findings

5. **Test incrementally** - change one parameter at a time

6. **Iterate** until scale bar is correctly detected

## Dataset-Specific Configuration

Create dataset-specific configurations for consistent results:

```yaml
scale_bar_rois:
  default:
    x_start_factor: 0.7
    y_start_factor: 0.05
    width_factor: 1.0
    height_factor: 0.05
  
  sem_dataset_1:  # Custom for SEM images with bottom-left scale bars
    x_start_factor: 0.0
    y_start_factor: 0.92
    width_factor: 0.25
    height_factor: 0.08
  
  tem_dataset_2:  # Custom for TEM images with top-right scale bars
    x_start_factor: 0.75
    y_start_factor: 0.02
    width_factor: 0.25
    height_factor: 0.06

scalebar_thresholds:
  intensity: 100
  proximity: 100
  merge_gap: 15
  min_line_length: 30
  edge_margin_factor: 0.1
```

## Advanced Tips

### Understanding the Algorithm

1. **ROI Selection**: Green box defines search area
2. **OCR Text Detection**: Blue box shows text location
3. **Line Detection**: Hough transform finds all horizontal lines
4. **Filtering**: Lines checked for intensity, proximity, length, edge distance
5. **Merging**: Nearby collinear segments merged into single lines
6. **Selection**: Longest line meeting all criteria is selected

### Optimal Configuration Strategy

1. **Start with defaults** for most datasets
2. **Adjust ROI first** to properly frame the scale bar
3. **Then adjust thresholds** to handle specific detection issues
4. **Use debug visualization** to validate each change
5. **Document working configs** for each dataset type

### Performance Considerations

- Scale bar detection adds minimal overhead (~0.1-0.3s per image)
- Debug visualization has negligible performance impact
- Consider disabling if scale bars not needed for analysis

## See Also

- [Configuration Reference](../configuration.md#scale-bar-thresholds)
- [API Documentation](../api/utils.md#scale-bar-detection-srcutilsscalebar_ocrpy)
- [User Guide](../user-guide.md#troubleshooting)
