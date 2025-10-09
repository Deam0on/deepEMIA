"""
Inference tracking utilities to prevent duplicate processing.

This module provides functions to:
- Check if images have already been processed
- Track processed images in GCS bucket
- Skip already-processed images to save time
"""

import subprocess
from pathlib import Path
from typing import Set

from src.utils.logger_utils import system_logger
from src.utils.config import get_config

config = get_config()
bucket = config["bucket"]


def get_processed_images_from_bucket(dataset_name: str, model_info: str) -> Set[str]:
    """
    Get list of images that have already been processed by checking the bucket.
    
    Checks for the existence of mask PNG files in the most recent archive
    for the given dataset and model combination.
    
    Args:
        dataset_name: Name of the dataset
        model_info: Model configuration info (e.g., "auto_models_adaptive")
    
    Returns:
        Set of image filenames (without extension) that have been processed
    """
    processed_images = set()
    
    try:
        # Search for recent archives matching this dataset and model
        search_prefix = f"Archive/*_{dataset_name}_{model_info}/"
        
        system_logger.info(f"Checking bucket for previously processed images...")
        system_logger.debug(f"Search pattern: gs://{bucket}/{search_prefix}")
        
        # List all archives for this dataset+model combo
        cmd = ["gsutil", "ls", f"gs://{bucket}/{search_prefix}"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            system_logger.info("No previous archives found for this dataset/model combination")
            return processed_images
        
        archives = result.stdout.strip().split('\n')
        archives = [a for a in archives if a]  # Filter empty lines
        
        if not archives:
            system_logger.info("No previous archives found")
            return processed_images
        
        # Use the most recent archive (last in sorted order)
        archives.sort()
        most_recent_archive = archives[-1]
        
        system_logger.info(f"Found {len(archives)} archive(s), checking most recent: {most_recent_archive}")
        
        # Check for mask files in the archive
        # Masks are named like: "image_name_mask.png"
        mask_pattern = f"{most_recent_archive}*_mask.png"
        
        cmd = ["gsutil", "ls", mask_pattern]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            mask_files = result.stdout.strip().split('\n')
            mask_files = [m for m in mask_files if m]
            
            # Extract image names from mask file paths
            for mask_file in mask_files:
                # Extract filename from full path
                filename = Path(mask_file).name
                # Remove "_mask.png" suffix to get original image name
                if filename.endswith('_mask.png'):
                    image_name = filename[:-9]  # Remove "_mask.png"
                    processed_images.add(image_name)
            
            system_logger.info(f"Found {len(processed_images)} previously processed images")
            if processed_images:
                system_logger.debug(f"Processed images: {sorted(list(processed_images))[:10]}...")  # Show first 10
        else:
            system_logger.info("No mask files found in most recent archive")
    
    except Exception as e:
        system_logger.warning(f"Error checking bucket for processed images: {e}")
        system_logger.info("Will process all images")
    
    return processed_images


def is_image_processed_in_bucket(image_name: str, dataset_name: str, model_info: str) -> bool:
    """
    Check if a specific image has been processed.
    
    This is a convenience function that checks the most recent archive
    for the presence of this image's mask file.
    
    Args:
        image_name: Name of the image file (with or without extension)
        dataset_name: Name of the dataset
        model_info: Model configuration info
    
    Returns:
        True if image has been processed, False otherwise
    """
    # Remove extension if present
    if '.' in image_name:
        image_name = image_name.rsplit('.', 1)[0]
    
    processed_images = get_processed_images_from_bucket(dataset_name, model_info)
    return image_name in processed_images
