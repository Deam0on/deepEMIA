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
from pathlib import Path
import yaml

from src.utils.logger_utils import system_logger


def load_spatial_constraints(dataset_name=None):
    """
    Load spatial constraint configuration for a dataset.
    
    Parameters:
    - dataset_name (str, optional): Name of the dataset
    
    Returns:
    - dict: Spatial constraints configuration
    """
    config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"
    
    default_config = {
        'enabled': False,
        'containment_rules': {},
        'overlap_rules': {},
        'containment_threshold': 0.95
    }
    
    if not config_path.exists():
        system_logger.warning(f"Config file not found: {config_path}, spatial constraints disabled")
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        inference_settings = config.get('inference_settings', {})
        spatial_config = inference_settings.get('spatial_constraints', {})
        
        # Check for dataset-specific configuration first
        if dataset_name:
            dataset_config = spatial_config.get(dataset_name, None)
            if dataset_config:
                # Use dataset-specific configuration
                result = {
                    'enabled': dataset_config.get('enabled', False),
                    'containment_rules': dataset_config.get('containment_rules', {}),
                    'overlap_rules': dataset_config.get('overlap_rules', {}),
                    'containment_threshold': dataset_config.get('containment_threshold', 0.95)
                }
                system_logger.info(f"Using dataset-specific spatial constraints for '{dataset_name}'")
                return result
            else:
                system_logger.info(f"No dataset-specific spatial constraints for '{dataset_name}', constraints disabled")
                return default_config
        
        # If no dataset name provided, return default (disabled)
        return default_config
        
    except Exception as e:
        system_logger.error(f"Error loading spatial constraints config: {e}")
        return default_config


def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    Parameters:
    - mask1 (numpy.ndarray): First binary mask
    - mask2 (numpy.ndarray): Second binary mask
    
    Returns:
    - float: IoU value between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_containment(child_mask, parent_mask):
    """
    Calculate what percentage of child mask is contained within parent mask.
    
    Parameters:
    - child_mask (numpy.ndarray): Child binary mask
    - parent_mask (numpy.ndarray): Parent binary mask
    
    Returns:
    - float: Containment ratio (0 = not contained, 1 = fully contained)
    """
    child_area = child_mask.sum()
    
    if child_area == 0:
        return 0.0
    
    intersection = np.logical_and(child_mask, parent_mask).sum()
    containment_ratio = intersection / child_area
    
    return containment_ratio


def filter_by_overlap_rules(masks, scores, classes, overlap_rules):
    """
    Filter instances based on overlap rules.
    Removes instances that violate overlap constraints.
    
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
        
        # Check each pair
        for i, idx1 in enumerate(sorted_indices):
            if idx1 in removed_indices:
                continue
            
            for idx2 in sorted_indices[i+1:]:
                if idx2 in removed_indices:
                    continue
                
                iou = calculate_iou(masks[idx1], masks[idx2])
                
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
        
        # Check each child instance
        for child_idx in class_indices[child_class]:
            if child_idx in removed_indices:
                continue
            
            child_mask = masks[child_idx]
            max_containment = 0.0
            best_parent_idx = None
            
            # Find best parent container
            for parent_idx in class_indices[parent_class]:
                if parent_idx in removed_indices:
                    continue
                
                parent_mask = masks[parent_idx]
                containment = calculate_containment(child_mask, parent_mask)
                
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
