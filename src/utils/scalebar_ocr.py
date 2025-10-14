"""
Scale bar detection and optical character recognition utilities.

This module provides functions for:
- Detecting scale bars in SEM/microscopy images
- OCR text extraction from scale bar regions
- Converting pixel measurements to real-world units
- Hough line detection for scale bar boundaries

The module integrates EasyOCR for text recognition and OpenCV for image processing.
"""

from typing import List
import cv2
import easyocr
import numpy as np
import re
import yaml
from math import sqrt
from pathlib import Path

from src.utils.logger_utils import system_logger


class ScaleBarDetectionError(Exception):
    pass


def get_scalebar_roi_for_dataset(dataset_name: str = None) -> dict:
    """
    Get scale bar ROI configuration for a specific dataset.
    Falls back to default if dataset-specific config not found.
    
    Parameters:
    - dataset_name (str, optional): Name of the dataset
    
    Returns:
    - dict: ROI configuration with x_start_factor, y_start_factor, width_factor, height_factor
    """
    config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"
    
    default_roi = {
        'x_start_factor': 0.7,
        'y_start_factor': 0.05,
        'width_factor': 1,
        'height_factor': 0.05
    }
    
    if not config_path.exists():
        system_logger.warning(f"Config file not found: {config_path}, using default ROI")
        return default_roi
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        scale_bar_rois = config.get('scale_bar_rois', {})
        
        # Try to get dataset-specific ROI first
        if dataset_name and dataset_name in scale_bar_rois:
            roi_config = scale_bar_rois[dataset_name]
            system_logger.info(f"Using dataset-specific scale bar ROI for '{dataset_name}'")
            return roi_config
        
        # Fall back to default
        default_from_config = scale_bar_rois.get('default', default_roi)
        
        if dataset_name:
            system_logger.info(f"No dataset-specific ROI for '{dataset_name}', using default")
        
        return default_from_config
        
    except Exception as e:
        system_logger.error(f"Error loading scale bar ROI config: {e}")
        return default_roi


def detect_scale_bar(
    image, roi_config=None, intensity_threshold=200, proximity_threshold=50, 
    dataset_name=None, draw_debug=False
):
    """
    Detects scale bars in SEM images using OCR and Hough line detection.

    Parameters:
    - image (numpy.ndarray): Input image (will be modified in-place if draw_debug=True)
    - roi_config (dict, optional): ROI configuration. If None, will load from config based on dataset_name
    - intensity_threshold (int): Minimum intensity for scale bar line detection
    - proximity_threshold (int): Maximum distance between text and line
    - dataset_name (str, optional): Dataset name for loading dataset-specific ROI
    - draw_debug (bool): If True, draws debug visualizations directly on the input image

    Returns:
    - tuple: (scale_bar_length_str, microns_per_pixel)
        - scale_bar_length_str (str): The detected scale bar length as a string (e.g., "500")
        - microns_per_pixel (float): The conversion factor from pixels to microns
    """
    # Load ROI config if not provided
    if roi_config is None:
        roi_config = get_scalebar_roi_for_dataset(dataset_name)
    
    # --- Load thresholds from config if available ---
    config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"
    merge_gap = 15  # Default
    min_line_length = 30  # Default
    edge_margin_factor = 0.1  # Default 10% margin from ROI edges
    
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)
            scalebar_thresholds = full_config.get("scalebar_thresholds", {})
            # Only override if not explicitly passed in function call
            if "intensity" in scalebar_thresholds and intensity_threshold == 200:
                intensity_threshold = scalebar_thresholds["intensity"]
            if "proximity" in scalebar_thresholds and proximity_threshold == 50:
                proximity_threshold = scalebar_thresholds["proximity"]
            merge_gap = scalebar_thresholds.get("merge_gap", 15)
            min_line_length = scalebar_thresholds.get("min_line_length", 30)
            edge_margin_factor = scalebar_thresholds.get("edge_margin_factor", 0.1)
        except Exception as e:
            system_logger.warning(f"Could not load thresholds from config: {e}")

    if not isinstance(image, np.ndarray):
        system_logger.error("Input image is not a numpy array.")
        raise ScaleBarDetectionError("Input image is not a numpy array.")
    for key in ["x_start_factor", "y_start_factor", "width_factor", "height_factor"]:
        if key not in roi_config:
            system_logger.error(f"ROI config missing key: {key}")
            raise ScaleBarDetectionError(f"ROI config missing key: {key}")

    h, w = image.shape[:2]
    x_start = int(w * roi_config["x_start_factor"])
    y_start = int(h * roi_config["y_start_factor"])
    x_end = int(x_start + w * roi_config["width_factor"])
    y_end = int(y_start + h * roi_config["height_factor"])

    system_logger.info(
        f"ROI for scale bar OCR: x={x_start}:{x_end}, y={y_start}:{y_end}"
    )

    # Draw ROI rectangle if debug enabled
    if draw_debug:
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.putText(image, "ROI", (x_start, y_start - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    roi = image[y_start:y_end, x_start:x_end].copy()
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Calculate ROI dimensions and edge margins
    roi_height, roi_width = gray_roi.shape[:2]
    x_margin = int(roi_width * edge_margin_factor)
    y_margin = int(roi_height * edge_margin_factor)
    
    system_logger.debug(f"ROI dimensions: {roi_width}x{roi_height}, margins: x={x_margin}, y={y_margin}")

    try:
        reader = easyocr.Reader(["en"], verbose=False)
        result = reader.readtext(gray_roi, detail=1, paragraph=False)
    except Exception as e:
        system_logger.error(f"EasyOCR failed: {e}")
        result = []

    text_box_center = None
    psum = "0"

    if result:
        system_logger.info(f"OCR detected text: {[r[1] for r in result]}")
        for detection in result:
            bbox, text, _ = detection
            text_clean = re.sub("[^0-9]", "", text)
            if text_clean:
                pxum_r = text
                psum = text_clean
                x_min = int(min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]))
                y_min = int(min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]))
                x_max = int(max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]))
                y_max = int(max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]))
                text_box_center = (
                    (x_min + x_max) // 2,
                    (y_min + y_max) // 2,
                )
                
                # Draw text bounding box if debug enabled
                if draw_debug:
                    text_x_min = x_start + x_min
                    text_y_min = y_start + y_min
                    text_x_max = x_start + x_max
                    text_y_max = y_start + y_max
                    cv2.rectangle(image, (text_x_min, text_y_min), 
                                (text_x_max, text_y_max), (255, 0, 0), 2)
                    cv2.putText(image, f"Text: {text}", 
                              (text_x_min, text_y_min - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                break
        else:
            psum = "0"
            text_box_center = None
    else:
        psum = "0"
        text_box_center = None
        system_logger.warning("No text detected by OCR in scale bar ROI.")

    edges = cv2.Canny(gray_roi, 50, 150, apertureSize=3)
    
    longest_line = None
    max_length = 0
    horizontal_lines = []  # Track horizontal lines for debugging

    if text_box_center:
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=20,
            maxLineGap=10,
        )
        
        if lines is not None:
            system_logger.debug(f"Total lines detected by Hough transform: {len(lines)}")
            
            # Collect all horizontal line segments
            raw_segments = []
            for line_idx, points in enumerate(lines):
                x1, y1, x2, y2 = points[0]
                
                # Check if line is approximately horizontal (within 10 degrees)
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle > 10 and angle < 170:
                    continue
                
                # Check if line is too close to ROI edges
                line_min_x = min(x1, x2)
                line_max_x = max(x1, x2)
                line_min_y = min(y1, y2)
                line_max_y = max(y1, y2)
                
                if (line_min_x < x_margin or 
                    line_max_x > roi_width - x_margin or
                    line_min_y < y_margin or 
                    line_max_y > roi_height - y_margin):
                    system_logger.debug(f"Line {line_idx} too close to edge, skipping: ({x1},{y1})-({x2},{y2})")
                    continue
                
                length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                line_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                dist_to_text = sqrt(
                    (line_center[0] - text_box_center[0]) ** 2
                    + (line_center[1] - text_box_center[1]) ** 2
                )
                
                # Intensity check: mean intensity along the line should be high (white bar)
                line_mask = np.zeros_like(gray_roi, dtype=np.uint8)
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
                mean_intensity = cv2.mean(gray_roi, mask=line_mask)[0]
                
                raw_segments.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'length': length,
                    'intensity': mean_intensity,
                    'dist_to_text': dist_to_text,
                    'line_idx': line_idx
                })
            
            # Merge collinear line segments that are close together
            merged_segments = merge_collinear_segments(raw_segments, merge_gap)
            
            system_logger.debug(f"After merging: {len(merged_segments)} line segments")
            
            # Now evaluate merged segments
            for seg_idx, seg in enumerate(merged_segments):
                x1, y1 = seg['x1'], seg['y1']
                x2, y2 = seg['x2'], seg['y2']
                length = seg['length']
                mean_intensity = seg['intensity']
                dist_to_text = seg['dist_to_text']
                
                # Double-check edge constraint after merging
                line_min_x = min(x1, x2)
                line_max_x = max(x1, x2)
                line_min_y = min(y1, y2)
                line_max_y = max(y1, y2)
                
                near_edge = (line_min_x < x_margin or 
                           line_max_x > roi_width - x_margin or
                           line_min_y < y_margin or 
                           line_max_y > roi_height - y_margin)
                
                # Store info about this horizontal line
                horizontal_lines.append((x1, y1, x2, y2, length, mean_intensity, dist_to_text, near_edge))
                
                # Draw all horizontal lines if debug enabled
                if draw_debug:
                    line_x1 = x_start + x1
                    line_y1 = y_start + y1
                    line_x2 = x_start + x2
                    line_y2 = y_start + y2
                    # Use different color for lines near edge
                    color = (128, 128, 128) if near_edge else (255, 255, 0)  # Gray if near edge, cyan otherwise
                    cv2.line(image, (line_x1, line_y1), (line_x2, line_y2), color, 1)
                    # Add line info text
                    edge_marker = " [EDGE]" if near_edge else ""
                    cv2.putText(image, 
                              f"M{seg_idx}: {length:.0f}px, I:{mean_intensity:.0f}, D:{dist_to_text:.0f}{edge_marker}", 
                              (line_x1, line_y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
                # Check if this line meets all criteria (including not near edge)
                if (dist_to_text < proximity_threshold and 
                    mean_intensity > intensity_threshold and 
                    length > min_line_length and
                    not near_edge):
                    if length > max_length:
                        max_length = length
                        longest_line = (x1, y1, x2, y2)
            
            # Log horizontal lines found
            system_logger.debug(f"Horizontal lines found: {len(horizontal_lines)}")
            
            # Log top 5 candidates
            if horizontal_lines:
                sorted_lines = sorted(horizontal_lines, key=lambda x: x[4], reverse=True)
                system_logger.debug("Top 5 line candidates:")
                for i, (x1, y1, x2, y2, length, intensity, dist, near_edge) in enumerate(sorted_lines[:5]):
                    passes_intensity = intensity > intensity_threshold
                    passes_proximity = dist < proximity_threshold
                    passes_length = length > min_line_length
                    passes_edge = not near_edge
                    system_logger.debug(
                        f"  {i+1}. Length: {length:.1f}px {'✓' if passes_length else '✗'}, "
                        f"Intensity: {intensity:.1f} {'✓' if passes_intensity else '✗'}, "
                        f"Distance: {dist:.1f}px {'✓' if passes_proximity else '✗'}, "
                        f"Edge: {'✓' if passes_edge else '✗ (too close)'}, "
                        f"Position: ({x1},{y1})-({x2},{y2})"
                    )
                
                # Check for potential segmentation issues
                if len(sorted_lines) > 1:
                    top_two_lengths = [sorted_lines[0][4], sorted_lines[1][4]]
                    if abs(top_two_lengths[0] - top_two_lengths[1]) < 20:
                        system_logger.warning(
                            f"Possible scale bar segmentation: Two similar lines detected "
                            f"({top_two_lengths[0]:.1f}px and {top_two_lengths[1]:.1f}px)"
                        )

    scale_len = 0
    um_pix = 1

    if longest_line:
        x1, y1, x2, y2 = longest_line
        x1_full = x1 + x_start
        x2_full = x2 + x_start
        y1_full = y1 + y_start
        y2_full = y2 + y_start
        
        # Draw selected scale bar line if debug enabled
        if draw_debug:
            cv2.line(image, (x1_full, y1_full), (x2_full, y2_full), (0, 0, 255), 3)
            cv2.putText(image, f"SELECTED: {max_length:.0f}px", 
                       (x1_full, y1_full - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        scale_len = max_length
        um_pix = float(psum) / scale_len if scale_len > 0 else 1.0
        system_logger.info(
            f"Detected scale bar: {psum} units, {scale_len:.2f} pixels, {um_pix:.4f} units/pixel"
        )
    else:
        um_pix = 1
        psum = "0"
        system_logger.warning("No scale bar line detected near OCR text.")
        
        # Add failure message if debug enabled
        if draw_debug:
            cv2.putText(image, "SCALE BAR DETECTION FAILED", 
                       (x_start, y_start + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return psum, um_pix


def merge_collinear_segments(segments: List[dict], max_gap: int = 15, angle_tolerance: int = 5, y_tolerance: int = 5) -> List[dict]:
    """
    Merge line segments that are collinear and close together.
    
    Parameters:
    - segments: List of segment dictionaries with x1, y1, x2, y2, etc.
    - max_gap: Maximum horizontal gap to merge segments (pixels)
    - angle_tolerance: Maximum angle difference to consider segments collinear (degrees)
    - y_tolerance: Maximum vertical offset to consider segments on same line (pixels)
    
    Returns:
    - List of merged segments
    """
    if not segments:
        return []
    
    # Sort segments by leftmost x coordinate
    sorted_segments = sorted(segments, key=lambda s: min(s['x1'], s['x2']))
    
    merged = []
    current_group = [sorted_segments[0]]
    
    for seg in sorted_segments[1:]:
        last = current_group[-1]
        
        # Get rightmost point of last segment
        last_right_x = max(last['x1'], last['x2'])
        last_y = (last['y1'] + last['y2']) / 2
        
        # Get leftmost point of current segment
        curr_left_x = min(seg['x1'], seg['x2'])
        curr_y = (seg['y1'] + seg['y2']) / 2
        
        # Calculate horizontal gap
        gap = curr_left_x - last_right_x
        
        # Calculate vertical offset
        y_offset = abs(curr_y - last_y)
        
        # Check if segments should be merged
        if gap <= max_gap and y_offset <= y_tolerance:
            current_group.append(seg)
        else:
            # Finalize current group and start new one
            merged.append(merge_segment_group(current_group))
            current_group = [seg]
    
    # Don't forget the last group
    if current_group:
        merged.append(merge_segment_group(current_group))
    
    return merged


def merge_segment_group(group: List[dict]) -> dict:
    """
    Merge a group of segments into a single segment.
    Takes the leftmost and rightmost points, averages intensity and distance.
    """
    if len(group) == 1:
        return group[0]
    
    # Find leftmost and rightmost points
    all_x = [seg['x1'] for seg in group] + [seg['x2'] for seg in group]
    all_y = [seg['y1'] for seg in group] + [seg['y2'] for seg in group]
    
    x1 = min(all_x)
    x2 = max(all_x)
    
    # Average the y coordinates
    y_avg = sum(all_y) / len(all_y)
    y1 = y2 = int(y_avg)
    
    # Calculate new length
    length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    # Average intensity and distance (weighted by segment length)
    total_length = sum(seg['length'] for seg in group)
    avg_intensity = sum(seg['intensity'] * seg['length'] for seg in group) / total_length
    avg_dist = sum(seg['dist_to_text'] * seg['length'] for seg in group) / total_length
    
    return {
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'length': length,
        'intensity': avg_intensity,
        'dist_to_text': avg_dist,
        'line_idx': -1  # Merged segment
    }
