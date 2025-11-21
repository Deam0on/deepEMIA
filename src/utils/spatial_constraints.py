"""
Spatial constraint utilities for class-based instance filtering.

This module provides functions for:
- Enforcing containment rules (child classes inside parent classes)
- Enforcing overlap rules (preventing unwanted overlaps)
- Validating spatial relationships between detected instances

The module integrates with the inference pipeline to filter predictions
based on spatial constraints defined in the configuration.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from functools import lru_cache

from src.utils.logger_utils import system_logger
from src.utils.config import get_config


def load_spatial_constraints(dataset_name=None) -> dict:
    """Load spatial constraints with proper config path resolution."""
    default_config = {
        'enabled': False,
        'containment_rules': {},
        'overlap_rules': {},
        'containment_threshold': 0.95
    }
    
    try:
        config = get_config(dataset_name=dataset_name)
        
        # TRY MULTIPLE LOCATIONS (inference_overrides vs inference_settings)
        spatial_config = None
        
        # 1. Check inference_overrides (your current dataset config location)
        if 'inference_overrides' in config:
            spatial_config = config['inference_overrides'].get('spatial_constraints')
        
        # 2. Fallback to inference_settings (legacy location)
        if spatial_config is None:
            spatial_config = config.get('inference_settings', {}).get('spatial_constraints', {})
        
        # 3. Check top-level (just in case)
        if spatial_config is None:
            spatial_config = config.get('spatial_constraints', {})
        
        if spatial_config is None:
            system_logger.debug(f"No spatial constraints found for '{dataset_name}'")
            return default_config
        
        # Handle dataset-specific nested configs
        if dataset_name and dataset_name in spatial_config:
            spatial_config = spatial_config[dataset_name]
        
        result = {**default_config, **spatial_config}
        
        if result['enabled']:
            system_logger.info(f"✓ Spatial constraints ENABLED for '{dataset_name}'")
            system_logger.debug(f"  Containment rules: {result['containment_rules']}")
            system_logger.debug(f"  Overlap rules: {result['overlap_rules']}")
        
        return result
        
    except Exception as e:
        system_logger.error(f"Error loading spatial constraints: {e}")
        return default_config


def get_mask_bbox(mask):
    """
    Get bounding box of a binary mask.
    
    Parameters:
    - mask (numpy.ndarray): Binary mask
    
    Returns:
    - tuple: (y_min, x_min, y_max, x_max) or None if mask is empty
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return (y_min, x_min, y_max, x_max)


def bboxes_overlap(bbox1, bbox2):
    """
    Check if two bounding boxes overlap.
    Fast pre-filter before expensive mask operations.
    
    Parameters:
    - bbox1, bbox2: (y_min, x_min, y_max, x_max)
    
    Returns:
    - bool: True if bounding boxes overlap
    """
    if bbox1 is None or bbox2 is None:
        return False
    
    y1_min, x1_min, y1_max, x1_max = bbox1
    y2_min, x2_min, y2_max, x2_max = bbox2
    
    # Check if NOT overlapping (faster to check negative case)
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False
    
    return True


def calculate_iou(mask1, mask2, bbox1=None, bbox2=None):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    OPTIMIZED: Uses bounding boxes for fast pre-filtering.
    
    Parameters:
    - mask1 (numpy.ndarray): First binary mask
    - mask2 (numpy.ndarray): Second binary mask
    - bbox1 (tuple, optional): Pre-computed bbox for mask1
    - bbox2 (tuple, optional): Pre-computed bbox for mask2
    
    Returns:
    - float: IoU value between 0 and 1
    """
    # Fast pre-filter: check bounding box overlap first
    if bbox1 is None:
        bbox1 = get_mask_bbox(mask1)
    if bbox2 is None:
        bbox2 = get_mask_bbox(mask2)
    
    if not bboxes_overlap(bbox1, bbox2):
        return 0.0
    
    # Only compute expensive mask operations if bboxes overlap
    # Use bitwise operations which are faster than logical operations
    intersection = np.count_nonzero(mask1 & mask2)
    
    if intersection == 0:
        return 0.0
    
    union = np.count_nonzero(mask1 | mask2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_containment(child_mask, parent_mask, child_bbox=None, parent_bbox=None):
    """
    Calculate what percentage of child mask is contained within parent mask.
    OPTIMIZED: Uses bounding boxes for fast pre-filtering.
    
    Parameters:
    - child_mask (numpy.ndarray): Child binary mask
    - parent_mask (numpy.ndarray): Parent binary mask
    - child_bbox (tuple, optional): Pre-computed bbox for child
    - parent_bbox (tuple, optional): Pre-computed bbox for parent
    
    Returns:
    - float: Containment ratio (0 = not contained, 1 = fully contained)
    """
    # Fast pre-filter: if bboxes don't overlap, containment is 0
    if child_bbox is None:
        child_bbox = get_mask_bbox(child_mask)
    if parent_bbox is None:
        parent_bbox = get_mask_bbox(parent_mask)
    
    if not bboxes_overlap(child_bbox, parent_bbox):
        return 0.0
    
    # Use count_nonzero which is faster than sum
    child_area = np.count_nonzero(child_mask)
    
    if child_area == 0:
        return 0.0
    
    # Use bitwise AND which is faster than logical_and
    intersection = np.count_nonzero(child_mask & parent_mask)
    containment_ratio = intersection / child_area
    
    return containment_ratio


def filter_by_overlap_rules(masks, scores, classes, overlap_rules):
    """
    Filter instances based on overlap rules.
    Removes instances that violate overlap constraints.
    OPTIMIZED: Pre-computes bounding boxes and uses spatial filtering.
    
    Parameters:
    - masks (list): List of binary masks
    - scores (list): List of confidence scores
    - classes (list): List of class IDs
    - overlap_rules (dict): Overlap rules per class
    
    Returns:
    - tuple: (filtered_masks, filtered_scores, filtered_classes, removed_indices)
    """
    if not overlap_rules:
        return masks, scores, classes, set()
    
    n = len(masks)
    removed_indices = set()
    
    # OPTIMIZATION: Pre-compute all bounding boxes once
    bboxes = [get_mask_bbox(mask) for mask in masks]
    
    # Group masks by class
    class_groups = {}
    for idx, cls in enumerate(classes):
        if cls not in class_groups:
            class_groups[cls] = []
        class_groups[cls].append(idx)
    
    # Check each class group for violations
    for cls, indices in class_groups.items():
        if cls not in overlap_rules:
            continue
        
        rule = overlap_rules[cls]
        allow_overlap = rule.get('allow_overlap', True)
        max_iou = rule.get('max_iou_threshold', 0.5)
        
        if allow_overlap and max_iou >= 0.9:
            # No restrictions, skip this class
            continue
        
        # Sort by score (keep higher scores)
        sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)
        
        # OPTIMIZATION: Check only bbox-overlapping pairs
        for i, idx1 in enumerate(sorted_indices):
            if idx1 in removed_indices:
                continue
            
            bbox1 = bboxes[idx1]
            
            for idx2 in sorted_indices[i+1:]:
                if idx2 in removed_indices:
                    continue
                
                bbox2 = bboxes[idx2]
                
                # Fast pre-filter: skip if bboxes don't overlap
                if not bboxes_overlap(bbox1, bbox2):
                    continue
                
                # Only compute expensive IoU if bboxes overlap
                iou = calculate_iou(masks[idx1], masks[idx2], bbox1, bbox2)
                
                if iou > max_iou:
                    # Violation detected - remove lower-confidence instance
                    removed_indices.add(idx2)
                    system_logger.debug(
                        f"Removing instance {idx2} (class {cls}, score {scores[idx2]:.3f}) "
                        f"due to overlap with instance {idx1} (IoU={iou:.3f} > {max_iou})"
                    )
    
    # Filter out removed instances
    filtered_masks = [m for i, m in enumerate(masks) if i not in removed_indices]
    filtered_scores = [s for i, s in enumerate(scores) if i not in removed_indices]
    filtered_classes = [c for i, c in enumerate(classes) if i not in removed_indices]
    
    if removed_indices:
        system_logger.info(
            f"Removed {len(removed_indices)} instances due to overlap violations"
        )
    
    return filtered_masks, filtered_scores, filtered_classes, removed_indices


def filter_by_containment_rules(masks, scores, classes, containment_rules, 
                                 containment_threshold=0.95):
    """
    Filter instances based on containment rules.
    Removes child instances not sufficiently contained within parent instances.
    OPTIMIZED: Pre-computes bounding boxes and uses spatial filtering.
    
    Parameters:
    - masks (list): List of binary masks
    - scores (list): List of confidence scores
    - classes (list): List of class IDs
    - containment_rules (dict): Containment rules {child_class: parent_class}
    - containment_threshold (float): Minimum containment ratio required (default 0.95 = 95%)
    
    Returns:
    - tuple: (filtered_masks, filtered_scores, filtered_classes, removed_indices)
    """
    if not containment_rules:
        return masks, scores, classes, set()
    
    n = len(masks)
    removed_indices = set()
    
    # OPTIMIZATION: Pre-compute all bounding boxes once
    bboxes = [get_mask_bbox(mask) for mask in masks]
    
    # Group masks by class
    class_indices = {}
    for idx, cls in enumerate(classes):
        if cls not in class_indices:
            class_indices[cls] = []
        class_indices[cls].append(idx)
    
    # Check each child class
    for child_class, parent_class in containment_rules.items():
        if child_class not in class_indices:
            continue  # No instances of this child class
        
        if parent_class not in class_indices:
            # No parent instances - remove all child instances
            system_logger.warning(
                f"No instances of parent class {parent_class} found for child class {child_class}. "
                f"Removing all {len(class_indices[child_class])} child instances."
            )
            removed_indices.update(class_indices[child_class])
            continue
        
        # OPTIMIZATION: Build spatial index for parent masks using centroids
        parent_indices = class_indices[parent_class]
        parent_centroids = []
        
        for parent_idx in parent_indices:
            if parent_idx in removed_indices:
                continue
            bbox = bboxes[parent_idx]
            if bbox is None:
                continue
            # Use bbox center as approximation for centroid (faster than computing true centroid)
            y_min, x_min, y_max, x_max = bbox
            centroid = ((y_min + y_max) / 2, (x_min + x_max) / 2)
            parent_centroids.append((parent_idx, centroid, bbox))
        
        # Check each child instance
        for child_idx in class_indices[child_class]:
            if child_idx in removed_indices:
                continue
            
            child_mask = masks[child_idx]
            child_bbox = bboxes[child_idx]
            
            if child_bbox is None:
                removed_indices.add(child_idx)
                continue
            
            max_containment = 0.0
            best_parent_idx = None
            
            # OPTIMIZATION: Only check parents whose bboxes overlap with child
            for parent_idx, parent_centroid, parent_bbox in parent_centroids:
                if parent_idx in removed_indices:
                    continue
                
                # Fast pre-filter: skip if bboxes don't overlap
                if not bboxes_overlap(child_bbox, parent_bbox):
                    continue
                
                # Only compute expensive containment if bboxes overlap
                parent_mask = masks[parent_idx]
                containment = calculate_containment(child_mask, parent_mask, child_bbox, parent_bbox)
                
                if containment > max_containment:
                    max_containment = containment
                    best_parent_idx = parent_idx
            
            # Check if sufficiently contained
            if max_containment < containment_threshold:
                removed_indices.add(child_idx)
                system_logger.debug(
                    f"Removing instance {child_idx} (class {child_class}, score {scores[child_idx]:.3f}) "
                    f"- not contained within parent class {parent_class} "
                    f"(max containment={max_containment:.3f} < {containment_threshold})"
                )
            else:
                system_logger.debug(
                    f"Instance {child_idx} (class {child_class}) contained within "
                    f"instance {best_parent_idx} (class {parent_class}, containment={max_containment:.3f})"
                )
    
    # Filter out removed instances
    filtered_masks = [m for i, m in enumerate(masks) if i not in removed_indices]
    filtered_scores = [s for i, s in enumerate(scores) if i not in removed_indices]
    filtered_classes = [c for i, c in enumerate(classes) if i not in removed_indices]
    
    if removed_indices:
        system_logger.info(
            f"Removed {len(removed_indices)} instances due to containment violations"
        )
    
    return filtered_masks, filtered_scores, filtered_classes, removed_indices


def apply_spatial_constraints(masks, scores, classes, dataset_name=None):
    """
    Apply all spatial constraints to a set of detected instances.
    
    This function:
    1. Filters instances based on overlap rules (e.g., preventing Class 0 overlaps)
    2. Filters instances based on containment rules (e.g., Class 1 inside Class 0)
    
    Parameters:
    - masks (list): List of binary masks
    - scores (list): List of confidence scores
    - classes (list): List of class IDs
    - dataset_name (str, optional): Name of dataset for loading specific constraints
    
    Returns:
    - tuple: (filtered_masks, filtered_scores, filtered_classes)
    """
    if not masks:
        return masks, scores, classes
    
    # Load spatial constraints
    config = load_spatial_constraints(dataset_name)
    
    if not config.get('enabled', False):
        system_logger.debug("Spatial constraints disabled, returning all instances")
        return masks, scores, classes
    
    system_logger.info(
        f"Applying spatial constraints: {len(masks)} instances before filtering"
    )
    
    original_count = len(masks)
    
    # Get containment threshold from config
    containment_threshold = config.get('containment_threshold', 0.95)
    
    # Apply overlap rules first
    overlap_rules = config.get('overlap_rules', {})
    if overlap_rules:
        masks, scores, classes, removed_overlap = filter_by_overlap_rules(
            masks, scores, classes, overlap_rules
        )
    
    # Apply containment rules
    containment_rules = config.get('containment_rules', {})
    if containment_rules:
        masks, scores, classes, removed_containment = filter_by_containment_rules(
            masks, scores, classes, containment_rules, containment_threshold
        )
    
    final_count = len(masks)
    removed_count = original_count - final_count
    
    if removed_count > 0:
        system_logger.info(
            f"Spatial constraints removed {removed_count} instances "
            f"({original_count} → {final_count})"
        )
    
    return masks, scores, classes
