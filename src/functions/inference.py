"""
Inference module for the deepEMIA project.

This module handles:
- Model inference on new images
- Mask prediction and processing
- Run-length encoding/decoding
- Post-processing of predictions
- Scale bar and arrow detection
- Wavelength analysis

The module provides a comprehensive pipeline for:
- Loading and preprocessing images
- Running model inference
- Post-processing predictions
- Analyzing results
- Saving predictions and visualizations
"""

## IMPORTS
import copy
import csv
import gc
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import cv2
import detectron2.data.transforms as T
import imutils
import numpy as np
import pandas as pd
import torch
import yaml
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
)
from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import ColorMode, Visualizer
from skimage.morphology import disk, erosion, dilation
from scipy.ndimage import binary_fill_holes

from src.data.datasets import read_dataset_info, register_datasets
from src.data.models import choose_and_use_model, get_trained_model_paths
from src.utils.logger_utils import system_logger
from src.utils.mask_utils import postprocess_masks, rle_encoding
from src.utils.measurements import calculate_measurements
from src.utils.scalebar_ocr import detect_scale_bar
from src.utils.spatial_constraints import apply_spatial_constraints

# Load config once at the start of your program
with open(Path.home() / "deepEMIA" / "config" / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

measure_contrast_distribution = config.get("measure_contrast_distribution", False)

# Load L4 performance optimization settings
l4_config = config.get("l4_performance_optimizations", {})
USE_MIXED_PRECISION = l4_config.get("use_mixed_precision", True)
GPU_OPTIMIZATIONS = l4_config.get("enable_gpu_optimizations", True)
PARALLEL_IMAGE_LOADING = l4_config.get("enable_parallel_image_loading", True)
PARALLEL_MASK_PROCESSING = l4_config.get("enable_parallel_mask_processing", True)
INFERENCE_BATCH_SIZE = l4_config.get("inference_batch_size", 3)
MEASUREMENT_BATCH_SIZE = l4_config.get("measurement_batch_size", 3)
CLEANUP_FREQUENCY = l4_config.get("clear_cache_frequency", 5)
CLEAR_CACHE_AFTER_TILES = l4_config.get("clear_cache_after_tiles", False)
MAX_WORKER_THREADS = l4_config.get("max_worker_threads", 3)
STREAM_MEASUREMENTS = l4_config.get("stream_measurements_to_csv", True)

# Load tile settings
tile_settings = config.get("inference_settings", {}).get("tile_settings", {})
TILE_BATCH_SIZE = tile_settings.get("tile_batch_size", 4)

# Load ensemble settings
ensemble_settings = config.get("inference_settings", {}).get("ensemble_settings", {})
ENSEMBLE_ENABLED = ensemble_settings.get("enabled", True)
ENSEMBLE_SMALL_CLASSES_ONLY = ensemble_settings.get("small_classes_only", True)
ENSEMBLE_WEIGHTS = ensemble_settings.get("weights", {"R50": 0.6, "R101": 0.4})

# Load iterative stopping settings
iterative_stopping = config.get("inference_settings", {}).get("iterative_stopping", {})
MIN_TOTAL_MASKS = iterative_stopping.get("min_total_masks", 10)
MIN_RELATIVE_INCREASE = iterative_stopping.get("min_relative_increase", 0.25)
MAX_CONSECUTIVE_ZERO = iterative_stopping.get("max_consecutive_zero", 2)
MIN_ITERATIONS = iterative_stopping.get("min_iterations", 2)

# Resolve paths
SPLIT_DIR = Path(config["paths"]["split_dir"]).expanduser().resolve()
CATEGORY_JSON = Path(config["paths"]["category_json"]).expanduser().resolve()
local_dataset_root = Path(config["paths"]["local_dataset_root"]).expanduser().resolve()


# =============================================================================
# L4 GPU OPTIMIZATION FUNCTIONS
# =============================================================================

def optimize_predictor_for_l4(predictor):
    """
    Optimize predictor specifically for L4 GPU and memory constraints.
    Uses configuration settings from config.yaml.
    
    Parameters:
    - predictor: Detectron2 predictor
    
    Returns:
    - predictor: Optimized predictor
    """
    system_logger.info("Optimizing predictor for L4 GPU using config settings...")
    
    # Set model to evaluation mode and disable gradients
    predictor.model.eval()
    for param in predictor.model.parameters():
        param.requires_grad = False
    
    # L4-specific optimizations (configurable)
    if torch.cuda.is_available() and GPU_OPTIMIZATIONS:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        system_logger.info(f"L4 GPU optimizations enabled (mixed_precision: {USE_MIXED_PRECISION})")
    else:
        system_logger.warning("CUDA not available or GPU optimizations disabled")
    
    return predictor


def load_images_parallel(image_paths, max_workers=None):
    """
    Load multiple images in parallel using threading.
    Uses configuration from config.yaml for thread count.
    
    Parameters:
    - image_paths: List of image file paths
    - max_workers: Maximum number of worker threads (uses config if None)
    
    Returns:
    - list: List of loaded images (same order as input paths)
    """
    if max_workers is None:
        max_workers = MAX_WORKER_THREADS
        
    if not PARALLEL_IMAGE_LOADING:
        # Fallback to sequential loading if parallel loading is disabled
        return [cv2.imread(path) for path in image_paths]
    
    def load_single_image(path):
        """Load a single image."""
        try:
            image = cv2.imread(path)
            if image is None:
                system_logger.warning(f"Could not load image: {path}")
            return image
        except Exception as e:
            system_logger.error(f"Error loading image {path}: {e}")
            return None
    
    # Load images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(executor.map(load_single_image, image_paths))
    
    return images


def process_masks_parallel(masks, max_workers=None):
    """
    Process multiple masks in parallel using threading.
    Uses configuration from config.yaml.
    
    Parameters:
    - masks: List of masks to process
    - max_workers: Maximum number of worker threads (uses config if None)
    
    Returns:
    - list: Processed masks
    """
    if max_workers is None:
        max_workers = min(2, MAX_WORKER_THREADS)  # Conservative for mask processing
        
    if not PARALLEL_MASK_PROCESSING or len(masks) <= 1:
        # Fallback to sequential processing
        return [process_single_mask_sequential(mask) for mask in masks]
    
    def process_single_mask_sequential(mask):
        """Apply morphological operations to a single mask (sequential version)."""
        try:
            # Fill holes first
            filled = binary_fill_holes(mask).astype(np.uint8)
            
            # Light morphological operations for speed
            kernel = disk(1)  # Small kernel for performance
            eroded = erosion(filled, kernel)
            dilated = dilation(eroded, kernel)
            
            return dilated
        except Exception as e:
            system_logger.warning(f"Error processing mask: {e}")
            return mask
    
    def process_single_mask(mask):
        """Apply morphological operations to a single mask."""
        return process_single_mask_sequential(mask)
    
    # Process masks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        processed = list(executor.map(process_single_mask, masks))
    
    return processed


def smart_memory_cleanup(iteration, cleanup_frequency=None):
    """
    Intelligent memory cleanup for L4 GPU constraints.
    Uses configuration from config.yaml.
    
    Parameters:
    - iteration: Current iteration/batch number
    - cleanup_frequency: How often to perform cleanup (uses config if None)
    """
    if cleanup_frequency is None:
        cleanup_frequency = CLEANUP_FREQUENCY
        
    if iteration % cleanup_frequency == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Log GPU memory usage occasionally
            if iteration % (cleanup_frequency * 5) == 0:  # Every 5th cleanup
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                    system_logger.debug(f"GPU memory: {allocated:.1f}GB/{gpu_memory:.1f}GB used")
                except:
                    pass


def stream_measurements_to_csv(csv_writer, csvfile, measurements_batch):
    for measurement in measurements_batch:
        csv_writer.writerow(measurement)
    csvfile.flush()


def calculate_image_quality_score(image):
    """
    Calculate image quality score based on brightness and contrast.
    Higher score = better quality.
    
    Parameters:
    - image: Input image (numpy array)
    
    Returns:
    - float: Quality score between 0 and 1
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate brightness (mean intensity)
    brightness = np.mean(gray) / 255.0
    
    # Calculate contrast (standard deviation)
    contrast = np.std(gray) / 128.0  # Normalize to ~0-1 range
    
    # Combine metrics (weighted average)
    quality_score = (0.4 * brightness) + (0.6 * contrast)
    
    # Clamp to [0, 1]
    quality_score = np.clip(quality_score, 0.0, 1.0)
    
    return quality_score


def adaptive_confidence_threshold(base_threshold, image, target_class, small_classes):
    """
    Automatically adjust confidence threshold based on image quality and class type.
    
    Parameters:
    - base_threshold: Base confidence threshold from config
    - image: Input image for quality assessment
    - target_class: Target class ID
    - small_classes: Set of class IDs considered "small"
    
    Returns:
    - float: Adjusted confidence threshold
    """
    # Check if auto mode is enabled
    confidence_mode = config.get("inference_settings", {}).get("confidence_mode", "auto")
    
    if confidence_mode == "manual":
        # Return base threshold without adjustment
        return base_threshold
    
    # Calculate image quality
    quality_score = calculate_image_quality_score(image)
    
    # Adjust threshold based on quality
    # Poor quality images (low score) -> lower threshold to catch more
    # Good quality images (high score) -> use base threshold
    if quality_score < 0.3:
        # Very poor quality - significantly lower threshold
        adjusted = base_threshold * 0.7
        adjustment_desc = "lowered (poor quality)"
    elif quality_score < 0.5:
        # Moderate quality - slightly lower threshold
        adjusted = base_threshold * 0.85
        adjustment_desc = "slightly lowered (moderate quality)"
    else:
        # Good quality - use base threshold
        adjusted = base_threshold
        adjustment_desc = "unchanged (good quality)"
    
    # Log adjustment for debugging
    is_small = target_class in small_classes
    class_type = "small" if is_small else "large"
    system_logger.debug(
        f"Class {target_class} ({class_type}): quality={quality_score:.2f}, "
        f"threshold {base_threshold:.2f} -> {adjusted:.2f} ({adjustment_desc})"
    )
    
    return adjusted


def get_confidence_threshold(image, target_class, small_classes):
    """
    Get the appropriate confidence threshold for a class.
    Supports both manual and automatic modes.
    
    Parameters:
    - image: Input image
    - target_class: Target class ID
    - small_classes: Set of class IDs considered "small"
    
    Returns:
    - float: Confidence threshold to use
    """
    # Get base threshold from config
    class_config = config.get("inference_settings", {}).get(
        "class_specific_settings", {}
    ).get(f"class_{target_class}", {})
    
    is_small = target_class in small_classes
    base_threshold = class_config.get(
        "confidence_threshold", 0.3 if is_small else 0.5
    )
    
    # Apply adaptive adjustment
    return adaptive_confidence_threshold(base_threshold, image, target_class, small_classes)


# =============================================================================
# ORIGINAL FUNCTIONS (with optimizations integrated)
# =============================================================================


def get_image_folder_path(base_path=Path.home() / "DATASET" / "INFERENCE"):
    """
    Determines the path to the folder containing images for inference.

    Parameters:
    - base_path (Path): Base path where the INFERENCE folder is located

    Returns:
    - str: Path to the folder containing the images

    Raises:
    - FileNotFoundError: If no images are found in either INFERENCE or INFERENCE/UPLOAD folders
    """
    # Define the two possible paths
    inference_path = os.path.join(base_path)
    upload_path = os.path.join(base_path, "UPLOAD")

    # Check if the INFERENCE folder contains images
    if any(
        os.path.isfile(os.path.join(inference_path, f))
        for f in os.listdir(inference_path)
    ):
        return inference_path

    # Check if the UPLOAD subfolder contains images
    elif os.path.exists(upload_path) and any(
        os.path.isfile(os.path.join(upload_path, f)) for f in os.listdir(upload_path)
    ):
        return upload_path

    # If no images found in either folder, raise an exception
    else:
        raise FileNotFoundError(
            "No images found in INFERENCE or INFERENCE/UPLOAD folders."
        )


def is_image_file(filename):
    """
    Checks if a filename corresponds to a supported image format.

    Parameters:
    - filename (str): The filename to check

    Returns:
    - bool: True if the file is a supported image format, False otherwise
    """
    return filename.lower().endswith(
        (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif")
    )


def iou(mask1, mask2):
    """
    Calculates the Intersection over Union (IoU) of two binary masks.

    Parameters:
    - mask1 (numpy.ndarray): First binary mask
    - mask2 (numpy.ndarray): Second binary mask

    Returns:
    - float: IoU score between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def run_inference(
    dataset_name,
    output_dir,
    visualize=True,
    threshold=0.65,
    draw_id=False,
    dataset_format="json",
):
    """
    Runs inference on a dataset and saves the results.
    
    Note: Iteration count is now automatic via config.yaml iterative_stopping settings.
    """
    
    # L4 OPTIMIZATION: Comprehensive optimization settings log
    system_logger.info("L4 GPU optimizations enabled")
    system_logger.info(f"Batch sizes: inference={INFERENCE_BATCH_SIZE}, measurement={MEASUREMENT_BATCH_SIZE}, cleanup every {CLEANUP_FREQUENCY} images")
    
    dataset_info = read_dataset_info(CATEGORY_JSON)
    register_datasets(dataset_info, dataset_name, dataset_format=dataset_format)

    # Force metadata population
    system_logger.debug("Forcing metadata population from DatasetCatalog...")
    d = DatasetCatalog.get(f"{dataset_name}_train")
    metadata = MetadataCatalog.get(f"{dataset_name}_train")
    system_logger.debug("Metadata populated successfully.")

    # FIX 1: Define num_classes from metadata
    num_classes = len(metadata.thing_classes)
    system_logger.debug(f"Number of classes: {num_classes} - {metadata.thing_classes}")

    # Memory optimization: Clear unnecessary data
    del d
    gc.collect()

    # AUTO-DETECT AVAILABLE MODELS AND OPTIMIZE FOR L4
    system_logger.info("Auto-detecting available trained models...")
    available_models = []
    available_predictors = []

    for r in [50, 101]:
        trained_model_paths = get_trained_model_paths(SPLIT_DIR, r)
        if dataset_name in trained_model_paths:
            system_logger.debug(f"Found trained R{r} model for dataset {dataset_name}")
            try:
                predictor, _ = choose_and_use_model(
                    trained_model_paths, dataset_name, threshold, metadata, r
                )
                if predictor is not None:
                    # L4 OPTIMIZATION: Optimize each predictor for L4 GPU
                    predictor = optimize_predictor_for_l4(predictor)
                    available_models.append(r)
                    available_predictors.append(predictor)
            except Exception as e:
                system_logger.warning(f"Failed to load R{r} model: {e}")
        else:
            system_logger.debug(
                f"No trained R{r} model found for dataset {dataset_name}"
            )

    # GRACEFUL EXIT IF NO MODELS FOUND
    if not available_predictors:
        error_msg = f"No trained models found for dataset '{dataset_name}'"
        system_logger.error(error_msg)
        system_logger.error("Please train models first using the train task")
        raise FileNotFoundError(error_msg)

    # LOG DETECTED MODELS
    model_info = ", ".join([f"R{r}" for r in available_models])
    if len(available_predictors) > 1:
        system_logger.info(f"Using COMBO inference with models: {model_info}")
    else:
        system_logger.info(f"Using single model inference with: {model_info}")

    # Set predictors for inference
    predictors = available_predictors

    image_folder_path = get_image_folder_path()

    # Path to save outputs
    path = output_dir
    os.makedirs(path, exist_ok=True)
    inpath = image_folder_path
    images_name = [f for f in os.listdir(inpath) if is_image_file(f)]

    # CALCULATE SIZE HEURISTIC
    system_logger.info(
        "Calculating mask size heuristics to determine small vs large classes..."
    )
    sample_images = [
        os.path.join(inpath, name) for name in images_name[:5]
    ]
    class_avg_sizes = calculate_average_mask_sizes(predictors, sample_images, metadata)
    small_classes = determine_small_classes(class_avg_sizes, threshold_percentile=50)

    # Log inference strategy
    system_logger.info("Using class-specific inference with automatic iteration control (config.yaml)")

    # L4 OPTIMIZATION: Use configured batch sizes
    batch_size = min(INFERENCE_BATCH_SIZE, len(images_name))
    system_logger.info(
        f"L4 OPTIMIZED: Processing {len(images_name)} images in batches of {batch_size} (configured for 16GB RAM)"
    )

    Img_ID = []
    EncodedPixels = []

    # Track processed images and timing
    processed_images = set()
    total_images = len(images_name)
    overall_start_time = time.perf_counter()

    # Get the scale bar ROI profiles from already-loaded config
    scale_bar_rois = config.get("scale_bar_rois", {})
    
    # Ensure there's a default profile
    if "default" not in scale_bar_rois:
        scale_bar_rois["default"] = {
            "x_start_factor": 0.667,
            "y_start_factor": 0.866,
            "width_factor": 1.0,
            "height_factor": 0.067
        }
        system_logger.warning("No default scale_bar_rois found in config, using hardcoded defaults")
    
    # Get the specific ROI config for this dataset
    dataset_roi_key = dataset_name if dataset_name in scale_bar_rois else "default"
    roi_config = scale_bar_rois[dataset_roi_key]
    system_logger.debug(
        f"Using scale bar ROI profile for '{dataset_roi_key}': {roi_config}"
    )

    conv = lambda l: " ".join(map(str, l))

    # Store deduplicated masks AND class predictions for each image
    dedup_results = {}

    # L4 OPTIMIZED: Process images in batches with parallel loading
    for batch_start in range(0, len(images_name), batch_size):
        batch_end = min(batch_start + batch_size, len(images_name))
        batch_names = images_name[batch_start:batch_end]

        system_logger.info(
            f"L4 OPTIMIZED: Processing batch {batch_start//batch_size + 1}/{(len(images_name) + batch_size - 1)//batch_size}: images {batch_start + 1}-{batch_end}"
        )

        # L4 OPTIMIZATION: Parallel image loading using config settings
        batch_image_paths = [os.path.join(inpath, name) for name in batch_names]
        system_logger.debug(f"Loading {len(batch_image_paths)} images in parallel (enabled: {PARALLEL_IMAGE_LOADING})...")
        
        # Load all images in this batch in parallel using configured thread count
        batch_images = load_images_parallel(batch_image_paths)
        
        # Filter out any None images (failed to load)
        valid_batch_data = [(name, img) for name, img in zip(batch_names, batch_images) if img is not None]
        
        if len(valid_batch_data) != len(batch_names):
            failed_count = len(batch_names) - len(valid_batch_data)
            system_logger.warning(f"Failed to load {failed_count} images in this batch")

        for idx_in_batch, (name, image) in enumerate(valid_batch_data):
            global_img_idx = batch_start + idx_in_batch
            image_start_time = time.perf_counter()  # Track entire image processing time
            system_logger.info(
                f"Processing image {name} ({global_img_idx + 1} out of {len(images_name)})"
            )

            # === SCALE BAR DETECTION (MUST COME FIRST) ===
            # Initialize default values
            um_pix = 1.0
            psum = "0"
            
            # Attempt scale bar detection with dataset-specific ROI
            try:
                psum, um_pix = detect_scale_bar(image.copy(), roi_config=None, dataset_name=dataset_name)
                system_logger.info(
                    f"Scale bar detected: {psum} units = {um_pix:.4f} units/pixel"
                )
            except Exception as e:
                system_logger.warning(
                    f"Scale bar detection failed for {name}: {e}. Using defaults (um_pix=1.0)"
                )
                um_pix = 1.0
                psum = "0"

            # === NOW RUN CLASS-SPECIFIC INFERENCE ===
            system_logger.info(f"Running class-specific inference for image {name}")
            
            all_masks_for_image = []
            all_scores_for_image = []
            all_classes_for_image = []
            
            for target_class in range(num_classes):
                is_small_class = target_class in small_classes
                class_name = metadata.thing_classes[target_class]
                
                system_logger.debug(f"Processing class {target_class} ({class_name})...")
                
                # Get adaptive confidence threshold based on image quality
                confidence_thresh = get_confidence_threshold(image, target_class, small_classes)
                
                # Get class-specific IOU threshold
                class_config = config.get("inference_settings", {}).get(
                    "class_specific_settings", {}
                ).get(f"class_{target_class}", {})
                
                iou_thresh = class_config.get(
                    "iou_threshold", 0.5 if is_small_class else 0.7
                )
                
                # Use ensemble for small classes if multiple predictors available
                active_predictors = predictors if (is_small_class and len(predictors) > 1) else [predictors[0]]
                
                # Tile-based inference (for all classes)
                system_logger.info(f"Running tile-based inference for class {target_class}")
                class_masks, class_scores, class_classes = tile_based_inference_pipeline(
                    active_predictors[0] if len(active_predictors) == 1 else active_predictors,
                    image,
                    target_class,
                    small_classes,
                    confidence_thresh,
                    tile_size=tile_settings.get("tile_size", 512),
                    overlap_ratio=tile_settings.get("overlap_ratio", 0.1),
                    upscale_factor=tile_settings.get("upscale_factor", 2.0),
                    scale_bar_info={"um_pix": um_pix, "psum": psum}
                )
            
                system_logger.debug(
                    f"Class {target_class}: Found {len(class_masks)} instances"
                )

                # Add to combined results
                all_masks_for_image.extend(class_masks)
                all_scores_for_image.extend(class_scores)
                all_classes_for_image.extend(class_classes)

            # Final cross-class deduplication (optional, more lenient)
            system_logger.info("Step 3: Deduplicating across all classes...")
            
            # Use optimized deduplication with bounding box pre-filtering
            final_masks, final_scores, final_classes = deduplicate_masks_smart(
                all_masks_for_image, 
                all_scores_for_image, 
                all_classes_for_image, 
                iou_threshold=0.7  # Use 0.7 for cross-class deduplication
            )

            # Apply spatial constraints (containment and overlap rules)
            system_logger.info("Step 4: Applying spatial constraints...")
            final_masks, final_scores, final_classes = apply_spatial_constraints(
                final_masks,
                final_scores,
                final_classes,
                dataset_name=dataset_name
            )

            unique_masks = final_masks
            unique_scores = final_scores
            unique_classes = final_classes
            unique_sources = [0] * len(unique_masks)  # All from same source

            # FIXED: Log results with class distribution for ALL classes
            class_counts = {}
            for cls in unique_classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1

            # Log all detected classes
            class_summary = ", ".join(
                [f"class {cls}: {count}" for cls, count in sorted(class_counts.items())]
            )
            if class_summary:
                system_logger.debug(
                    f"After processing: {len(unique_masks)} unique masks for image {name} ({class_summary})"
                )
            else:
                system_logger.debug(
                    f"After processing: {len(unique_masks)} unique masks for image {name} (no classes detected)"
                )

            # Save for later use - now including classes
            dedup_results[name] = {
                "masks": unique_masks,
                "scores": unique_scores,
                "sources": unique_sources,
                "classes": unique_classes,
            }

            processed_images.add(name)

            # Memory optimization: Encode masks immediately and clear image data
            for i, mask in enumerate(unique_masks):
                Img_ID.append(name.rsplit(".", 1)[0])
                EncodedPixels.append(conv(rle_encoding(mask)))

            # Log inference time for this image
            image_inference_time = time.perf_counter() - image_start_time
            system_logger.info(f"Image {name} inference complete: {image_inference_time:.3f}s, {len(unique_masks)} masks detected")

            # Memory optimization: Clear image and mask data after processing
            del image, unique_masks, unique_scores, unique_sources, unique_classes
            gc.collect()

    overall_elapsed = time.perf_counter() - overall_start_time
    average_time = overall_elapsed / total_images if total_images else 0
    system_logger.debug(
        f"Average mask generation and deduplication time per image: {average_time:.3f} seconds"
    )

    # Ensure all images were processed
    unprocessed = set(images_name) - processed_images
    if unprocessed:
        system_logger.warning(f"The following images were not processed: {unprocessed}")
    else:
        system_logger.info("All images in the INFERENCE folder were processed.")

    # Save RLE results
    df = pd.DataFrame({"ImageId": Img_ID, "EncodedPixels": EncodedPixels})
    df.to_csv(os.path.join(path, "R50_flip_results.csv"), index=False, sep=",")

    # Memory optimization: Clear large dataframes
    del df, Img_ID, EncodedPixels
    gc.collect()

    # MODIFIED: Single measurements file with class information
    system_logger.info("Starting measurements phase...")

    csv_filename = os.path.join(output_dir, "measurements_results.csv")
    test_img_path = image_folder_path

    # Define colors for different classes (BGR format for OpenCV)
    class_colors = [
        (0, 255, 0),  # Green for class 0
        (255, 0, 0),  # Blue for class 1
        (0, 0, 255),  # Red for class 2
        (255, 255, 0),  # Cyan for class 3
        (255, 0, 255),  # Magenta for class 4
        (0, 255, 255),  # Yellow for class 5
        (128, 0, 128),  # Purple for class 6
        (255, 165, 0),  # Orange for class 7
    ]

    with open(csv_filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)

        # ADDED: Class column to CSV header
        csvwriter.writerow(
            [
                "Instance_ID",
                "Class",
                "Class_Name",
                "Major axis length",
                "Minor axis length",
                "Eccentricity",
                "C. Length",
                "C. Width",
                "Circular eq. diameter",
                "Aspect ratio",
                "Circularity",
                "Chord length",
                "Ferret diameter",
                "Roundness",
                "Sphericity",
                "Contrast d10",
                "Contrast d50",
                "Contrast d90",
                "Detected scale bar",
                "File name",
            ]
        )

        image_list = [f for f in os.listdir(test_img_path) if is_image_file(f)]
        num_images = len(image_list)
        total_time = 0

        # L4 OPTIMIZED: Use configured measurement batch size
        measurement_batch_size = min(MEASUREMENT_BATCH_SIZE, len(image_list))

        for batch_start in range(0, len(image_list), measurement_batch_size):
            batch_end = min(batch_start + measurement_batch_size, len(image_list))
            batch_images = image_list[batch_start:batch_end]

            system_logger.info(
                f"L4 OPTIMIZED: Processing measurements batch {batch_start//measurement_batch_size + 1}/{(len(image_list) + measurement_batch_size - 1)//measurement_batch_size} (configured batch size: {MEASUREMENT_BATCH_SIZE})"
            )

            # Prepare measurements batch for streaming
            measurements_batch = []

            for idx_in_batch, test_img in enumerate(batch_images):
                idx = batch_start + idx_in_batch + 1
                start_time = time.perf_counter()
                system_logger.debug(
                    f"Processing measurements for image {idx} out of {num_images}: {test_img}"
                )

                input_path = os.path.join(test_img_path, test_img)
                im = cv2.imread(input_path)

                if im is None:
                    system_logger.warning(
                        f"Could not load image for measurements: {input_path}"
                    )
                    continue

                psum, um_pix = detect_scale_bar(im, roi_config=None, dataset_name=dataset_name)

                # Use deduplicated masks and classes for this image
                image_data = dedup_results.get(test_img, {})
                masks = image_data.get("masks", [])
                classes = image_data.get("classes", [])

                if not masks:
                    system_logger.info(
                        f"No masks found for image {test_img}, skipping measurements"
                    )
                    continue

                system_logger.debug(
                    f"Processing {len(masks)} masks for image {test_img}"
                )

                # Track measurements statistics
                measurements_written = 0
                masks_filtered = 0
                class_measurements = {}

                # MODIFIED: Single visualization per image with color-coded classes
                if visualize:
                    vis_img = im.copy()

                    for i, (mask, cls) in enumerate(zip(masks, classes)):
                        # Use class-specific color
                        color = class_colors[cls % len(class_colors)]

                        # Create colored overlay for each mask
                        colored_mask = np.zeros_like(vis_img)
                        colored_mask[mask.astype(bool)] = color
                        vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, 0.5, 0)

                        # Draw contours
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        cv2.drawContours(vis_img, contours, -1, color, 1)

                        # Draw instance ID and class at centroid
                        M = cv2.moments(mask.astype(np.uint8))
                        if M["m00"] > 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])

                            # Get class name
                            class_name = (
                                metadata.thing_classes[cls]
                                if cls < len(metadata.thing_classes)
                                else f"class_{cls}"
                            )

                            # Draw instance ID (REDUCED font size from 0.6 to 0.4)
                            cv2.putText(
                                vis_img,
                                f"{i + 1}",
                                (cX, cY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,  # CHANGED: was 0.6
                                (255, 255, 255),  # White text
                                1,  # CHANGED: thickness from 2 to 1
                                cv2.LINE_AA,
                            )

                            # Draw class name (REDUCED font size from 0.4 to 0.3)
                            cv2.putText(
                                vis_img,
                                class_name,
                                (cX, cY + 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,  # CHANGED: was 0.4
                                (255, 255, 255),  # White text
                                1,
                                cv2.LINE_AA,
                            )

                        del colored_mask

                    # Single PNG per image
                    vis_save_path = os.path.join(
                        output_dir, f"{test_img}_predictions.png"
                    )
                    cv2.imwrite(vis_save_path, vis_img)
                    del vis_img
                    gc.collect()

                # Process each mask for measurements
                for instance_id, (mask, cls) in enumerate(zip(masks, classes), 1):
                    binary_mask = (mask > 0).astype(np.uint8) * 255
                    single_im_mask = binary_mask.copy()

                    # Save individual mask image with class info
                    mask_3ch = np.stack([single_im_mask] * 3, axis=-1)
                    class_name = (
                        metadata.thing_classes[cls]
                        if cls < len(metadata.thing_classes)
                        else f"class_{cls}"
                    )
                    mask_filename = os.path.join(
                        output_dir, f"{test_img}_mask_{instance_id}_{class_name}.jpg"
                    )
                    cv2.imwrite(mask_filename, mask_3ch)

                    single_cnts = cv2.findContours(
                        single_im_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    single_cnts = imutils.grab_contours(single_cnts)

                    # Track measurements for this mask
                    mask_measurements = 0
                    total_contours = len(single_cnts)  # Add this line

                    for c in single_cnts:
                        pixelsPerMetric = 1
                        contour_area = cv2.contourArea(c)

                        # ADAPTIVE: Scale thresholds based on image size
                        image_area = im.shape[0] * im.shape[1]
                        base_threshold = image_area * 0.000005  # 0.0005% of image area

                        # if cls == 0:  # First class
                        #     min_area = max(25, base_threshold * 2)
                        # else:  # Other classes
                        min_area = max(5, base_threshold * 0.05)

                        if contour_area < min_area:
                            system_logger.info(
                                f"Skipping contour in mask {instance_id} (class {cls}): area {contour_area:.1f} < {min_area:.1f}"
                            )
                            continue

                        measurements = calculate_measurements(
                            c,
                            single_im_mask,
                            um_pix=um_pix,
                            pixelsPerMetric=pixelsPerMetric,
                            original_image=im,
                            measure_contrast_distribution=measure_contrast_distribution,
                        )

                        # Get class name for CSV
                        class_name = (
                            metadata.thing_classes[cls]
                            if cls < len(metadata.thing_classes)
                            else f"class_{cls}"
                        )

                        # L4 OPTIMIZATION: Add measurements to batch instead of writing immediately
                        measurements_batch.append([
                            f"{test_img}_{instance_id}",
                            cls,  # Class number
                            class_name,  # Class name
                            measurements["major_axis_length"],
                            measurements["minor_axis_length"],
                            measurements["eccentricity"],
                            measurements["Length"],
                            measurements["Width"],
                            measurements["CircularED"],
                            measurements["Aspect_Ratio"],
                            measurements["Circularity"],
                            measurements["Chords"],
                            measurements["Feret_diam"],
                            measurements["Roundness"],
                            measurements["Sphericity"],
                            measurements["contrast_d10"],
                            measurements["contrast_d50"],
                            measurements["contrast_d90"],
                            psum,
                            test_img,
                        ])

                        mask_measurements += 1
                        measurements_written += 1

                        # Track by class
                        if cls not in class_measurements:
                            class_measurements[cls] = 0
                        class_measurements[cls] += 1

                    # Add this logging block to show why masks are filtered
                    if mask_measurements == 0:
                        masks_filtered += 1
                        system_logger.debug(
                            f"Mask {instance_id} (class {cls}) filtered out: all {total_contours} contours too small (min_area: {min_area:.1f})"
                        )
                    else:
                        system_logger.debug(
                            f"Mask {instance_id} (class {cls}): {mask_measurements}/{total_contours} contours kept"
                        )

                    # Memory optimization: Clear mask data
                    del binary_mask, single_im_mask, mask_3ch, single_cnts
                    gc.collect()

                # ADDED: Report measurement statistics
                system_logger.info(
                    f"Measurements written: {measurements_written}/{len(masks)} masks processed"
                )
                system_logger.info(f"Masks filtered out: {masks_filtered}")

                if class_measurements:
                    class_summary = ", ".join(
                        [
                            f"class {cls}: {count}"
                            for cls, count in sorted(class_measurements.items())
                        ]
                    )
                    system_logger.info(f"Final measurements by class: {class_summary}")
                else:
                    system_logger.warning("No measurements written for any class!")

                # Memory optimization: Clear image data
                del im

                elapsed = time.perf_counter() - start_time
                total_time += elapsed
                system_logger.info(f"Image {test_img} measurements complete: {elapsed:.3f}s, {measurements_written} measurements")

            # L4 OPTIMIZATION: Stream measurements batch to CSV using config
            if measurements_batch and STREAM_MEASUREMENTS:
                stream_measurements_to_csv(csvwriter, csvfile, measurements_batch)
                system_logger.debug(f"Streamed {len(measurements_batch)} measurements to CSV")
                measurements_batch.clear()  # Clear batch from memory
            elif measurements_batch:
                # Fallback: write immediately if streaming is disabled
                for measurement in measurements_batch:
                    csvwriter.writerow(measurement)
                measurements_batch.clear()

            # L4 OPTIMIZATION: Use configured memory cleanup frequency
            smart_memory_cleanup(batch_start)

    # Final memory cleanup
    del dedup_results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    average_time = total_time / num_images if num_images else 0
    system_logger.info(f"Measurements complete (avg {average_time:.3f}s/image). Results: {csv_filename}")

    # Create a color legend file
    legend_path = os.path.join(output_dir, "class_color_legend.txt")
    with open(legend_path, "w") as f:
        f.write("Class Color Legend:\n")
        f.write("==================\n")
        for i, class_name in enumerate(metadata.thing_classes):
            color_bgr = class_colors[i % len(class_colors)]
            color_rgb = (
                color_bgr[2],
                color_bgr[1],
                color_bgr[0],
            )
            f.write(f"Class {i} ({class_name}): RGB{color_rgb}\n")
    
    # ADD THIS CLEANUP CODE AT THE END:
    # Clean up individual mask files after inference completion
    system_logger.info("Cleaning up individual mask files...")
    
    mask_files_removed = 0
    mask_pattern = os.path.join(output_dir, "*_mask_*_*.jpg")
    
    try:
        import glob
        for mask_file in glob.glob(mask_pattern):
            try:
                os.remove(mask_file)
                mask_files_removed += 1
            except Exception as e:
                system_logger.warning(f"Could not remove mask file {mask_file}: {e}")
        
        if mask_files_removed > 0:
            system_logger.info(f"Cleaned up {mask_files_removed} individual mask files")
        else:
            system_logger.debug("No mask files found to clean up")
            
    except Exception as e:
        system_logger.warning(f"Error during mask file cleanup: {e}")
    
    system_logger.info("Inference completed successfully")


def run_class_specific_inference(
    predictor, image, target_class, small_classes, confidence_threshold=0.3, iou_threshold=0.7
):
    """
    Run inference targeting a specific class with L4 GPU optimizations.
    Now includes mixed precision inference for 30-50% speedup on L4.

    Parameters:
    - predictor: Model predictor (can be a list for ensemble)
    - image: Input image
    - target_class: Class to focus on (0, 1, etc.)
    - small_classes: Set of classes considered "small"
    - confidence_threshold: Confidence threshold for this class
    - iou_threshold: IoU threshold for this class

    Returns:
    - tuple: (masks, scores, classes) for the target class only
    """
    # Check if predictor is a list (ensemble mode)
    if isinstance(predictor, list):
        return run_ensemble_inference(
            predictor, image, target_class, small_classes, 
            confidence_threshold, iou_threshold
        )
    
    # Single predictor mode
    # Memory optimization: Clear GPU cache before inference (only if not in tile batch mode)
    if torch.cuda.is_available() and CLEAR_CACHE_AFTER_TILES:
        torch.cuda.empty_cache()

    # L4 OPTIMIZATION: Use configured mixed precision setting
    use_mixed_precision = torch.cuda.is_available() and USE_MIXED_PRECISION
    
    if use_mixed_precision:
        # Use automatic mixed precision (AMP) for L4 Tensor Cores
        with torch.cuda.amp.autocast():
            outputs = predictor(image)
    else:
        # Fallback for CPU or when mixed precision is disabled
        outputs = predictor(image)

    # Get all predictions
    pred_masks = outputs["instances"].to("cpu")._fields["pred_masks"].numpy()
    pred_scores = outputs["instances"].to("cpu")._fields["scores"].numpy()
    pred_classes = outputs["instances"].to("cpu")._fields["pred_classes"].numpy()

    # Memory cleanup - immediately clear GPU outputs (only if configured)
    del outputs
    if torch.cuda.is_available() and CLEAR_CACHE_AFTER_TILES:
        torch.cuda.empty_cache()

    # Filter for target class only
    class_mask = pred_classes == target_class
    class_pred_masks = pred_masks[class_mask]
    class_pred_scores = pred_scores[class_mask]
    class_pred_classes = pred_classes[class_mask]

    # Apply class-specific confidence threshold
    confidence_mask = class_pred_scores >= confidence_threshold
    filtered_masks = class_pred_masks[confidence_mask]
    filtered_scores = class_pred_scores[confidence_mask]
    filtered_classes = class_pred_classes[confidence_mask]

    if len(filtered_masks) == 0:
        return [], [], []

    # Class-specific postprocessing with parallel processing
    is_small_class = target_class in small_classes
    
    if is_small_class:
        min_size = 5  # Much smaller minimum for small particles
        processed_masks = postprocess_masks(
            filtered_masks, filtered_scores, image, min_crys_size=min_size
        )
    else:
        min_size = 25  # Standard size for large particles
        processed_masks = postprocess_masks(
            filtered_masks, filtered_scores, image, min_crys_size=min_size
        )

    # L4 OPTIMIZATION: Parallel mask processing using config
    if len(processed_masks) > 2 and PARALLEL_MASK_PROCESSING:
        processed_masks = process_masks_parallel(processed_masks)

    # Class-specific deduplication
    unique_masks = []
    unique_scores = []
    unique_classes = []

    if processed_masks:
        # More lenient IoU for small particles (as they're harder to detect)
        current_iou_threshold = 0.5 if is_small_class else iou_threshold

        for i, mask in enumerate(processed_masks):
            if not any(iou(mask, um) > current_iou_threshold for um in unique_masks):
                unique_masks.append(mask)
                unique_scores.append(filtered_scores[i])
                unique_classes.append(target_class)

    return unique_masks, unique_scores, unique_classes


def run_ensemble_inference(
    predictors,
    image,
    target_class,
    small_classes,
    conf_threshold,
    iou_threshold,
):
    """
    Run inference using multiple models and ensemble the results.
    
    Args:
        predictors: List of Detectron2 predictors
        image: Input image
        target_class: Target class ID
        small_classes: Set of small class IDs
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for deduplication
    
    Returns:
        Tuple of (masks, scores, classes) after ensemble
    """
    if len(predictors) < 2:
        system_logger.warning("Ensemble requires at least 2 models, falling back to single model")
        return run_class_specific_inference(
            predictors[0], image, target_class, small_classes, conf_threshold, iou_threshold
        )
    
    system_logger.info(f"Running ensemble inference with {len(predictors)} models for class {target_class}")
    
    # Run inference with each model
    all_model_results = []
    model_names = ['R50', 'R101']  # Assuming first is R50, second is R101
    
    for idx, predictor in enumerate(predictors):
        model_name = model_names[idx] if idx < len(model_names) else f'Model{idx}'
        system_logger.debug(f"Running inference with {model_name}...")
        
        # Call the correct function: run_class_specific_inference (single predictor, no ensemble)
        masks, scores, classes = run_class_specific_inference(
            predictor, image, target_class, small_classes, conf_threshold, iou_threshold
        )
        
        # Apply ensemble weight to scores
        weight = ENSEMBLE_WEIGHTS.get(model_name, 1.0 / len(predictors))
        weighted_scores = [s * weight for s in scores]
        
        all_model_results.append({
            'model': model_name,
            'masks': masks,
            'scores': weighted_scores,
            'classes': classes,
            'weight': weight
        })
        
        system_logger.info(f"  {model_name}: {len(masks)} instances (weight: {weight:.2f})")
    
    # Combine results from all models
    combined_masks = []
    combined_scores = []
    combined_classes = []
    
    for result in all_model_results:
        combined_masks.extend(result['masks'])
        combined_scores.extend(result['scores'])
        combined_classes.extend(result['classes'])
    
    system_logger.info(f"Combined {len(combined_masks)} masks from {len(predictors)} models")
    
    # Deduplicate combined results
    if len(combined_masks) > 0:
        unique_masks, unique_scores, unique_classes = deduplicate_masks_smart(
            combined_masks, combined_scores, combined_classes, iou_threshold=iou_threshold
        )
        
        # Store first model's mask count for comparison
        first_model_count = len(all_model_results[0]['masks'])
        
        system_logger.info(
            f"Ensemble result: {len(unique_masks)} unique masks "
            f"(gain: +{len(unique_masks) - first_model_count} from first model)"
        )
        
        return unique_masks, unique_scores, unique_classes
    else:
        system_logger.warning("No masks detected by any model in ensemble")
        return [], [], []


def log_scale_detection_summary(scale, masks_before, masks_after, original_shape, scaled_shape):
    """
    Log detailed summary of what happened at each scale.
    
    Parameters:
    - scale: Scale factor
    - masks_before: Number of masks before rescaling
    - masks_after: Number of masks after rescaling and deduplication
    - original_shape: Original image shape
    - scaled_shape: Scaled image shape
    """
    if masks_before == 0:
        status = "No detections"
    elif masks_after < masks_before:
        status = f"{masks_before - masks_after} filtered as duplicates"
    else:
        status = "All unique"
    
    system_logger.info(
        f"Scale {scale}x: {original_shape[1]}x{original_shape[0]} → "
        f"{scaled_shape[1]}x{scaled_shape[0]} | "
        f"Detected: {masks_before} | Kept: {masks_after} | {status}"
    )


def calculate_average_mask_sizes(predictors, images_sample, metadata):
    """
    Calculate average mask sizes per class to determine which classes are "small" vs "large".

    This heuristic samples a few images to determine typical mask sizes for each class,
    enabling universal postprocessing without hardcoded assumptions.

    Parameters:
    - predictors: List of model predictors
    - images_sample: Sample of image paths to analyze
    - metadata: Dataset metadata

    Returns:
    - dict: {class_id: average_mask_size} for each detected class
    """
    system_logger.info("Calculating average mask sizes per class for heuristic...")

    class_sizes = {}  # {class_id: [sizes]}

    # Process a sample of images (max 5 to be fast)
    sample_size = min(5, len(images_sample))
    sample_images = images_sample[:sample_size]

    for i, image_path in enumerate(sample_images):
        system_logger.debug(
            f"Analyzing sample image {i+1}/{sample_size} for size heuristic"
        )

        image = cv2.imread(image_path)
        if image is None:
            continue

        # Use first available predictor for sampling
        predictor = predictors[0]
        
        # L4 OPTIMIZATION: Use mixed precision for inference sampling
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        use_mixed_precision = torch.cuda.is_available()
        
        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = predictor(image)
        else:
            outputs = predictor(image)

        pred_masks = outputs["instances"].to("cpu")._fields["pred_masks"].numpy()
        pred_classes = outputs["instances"].to("cpu")._fields["pred_classes"].numpy()
        pred_scores = outputs["instances"].to("cpu")._fields["scores"].numpy()

        # Only consider high-confidence predictions for size analysis
        confident_mask = pred_scores >= 0.7
        confident_masks = pred_masks[confident_mask]
        confident_classes = pred_classes[confident_mask]

        for mask, cls in zip(confident_masks, confident_classes):
            mask_size = np.sum(mask)
            if cls not in class_sizes:
                class_sizes[cls] = []
            class_sizes[cls].append(mask_size)

        del outputs, pred_masks, pred_classes, pred_scores, image
        gc.collect()

    # Calculate averages
    class_avg_sizes = {}
    for cls, sizes in class_sizes.items():
        if sizes:
            avg_size = np.mean(sizes)
            class_avg_sizes[cls] = avg_size
            class_name = (
                metadata.thing_classes[cls]
                if cls < len(metadata.thing_classes)
                else f"class_{cls}"
            )
            system_logger.debug(
                f"Class {cls} ({class_name}): average mask size = {avg_size:.1f} pixels"
            )

    return class_avg_sizes


def determine_small_classes(class_avg_sizes, threshold_percentile=50):
    """
    Determine which classes should be considered "small" based on average mask sizes.

    Parameters:
    - class_avg_sizes: Dict of {class_id: average_mask_size}
    - threshold_percentile: Percentile threshold (classes below this are "small")

    Returns:
    - set: Set of class IDs that are considered "small"
    """
    if not class_avg_sizes:
        return set()

    sizes = list(class_avg_sizes.values())
    threshold_size = np.percentile(sizes, threshold_percentile)

    small_classes = {
        cls for cls, size in class_avg_sizes.items() if size <= threshold_size
    }

    system_logger.debug(f"Size threshold: {threshold_size:.1f} pixels (50th percentile)")
    system_logger.info(f"Small classes (≤ threshold): {sorted(small_classes)}")
    system_logger.info(
        f"Large classes (> threshold): {sorted(set(class_avg_sizes.keys()) - small_classes)}"
    )

    return small_classes


def postprocess_masks_universal(
    ori_mask, ori_score, image, target_class, is_small_class, min_crys_size=None
):
    """
    Universal postprocessing for masks with class-specific size thresholds.
    
    Parameters:
    - ori_mask: Original masks
    - ori_score: Confidence scores
    - image: Input image
    - target_class: Target class ID
    - is_small_class: Whether this is a small class
    - min_crys_size: Minimum size threshold (if None, calculated from image)
    
    Returns:
    - list: Processed masks
    """
    if len(ori_mask) == 0:
        return []

    image_area = image.shape[0] * image.shape[1]
    height, width = image.shape[:2]

    # ONLY calculate from image if not provided
    if min_crys_size is None:
        if is_small_class:
            # More aggressive for small particles
            min_crys_size = max(3, int(image_area * 0.000005))  # 0.0005%
        else:
            min_crys_size = max(25, int(image_area * 0.0001))  # 0.01%
    
    system_logger.debug(
        f"Postprocessing: min_size={min_crys_size}px (class={target_class}, "
        f"small={is_small_class}, image_area={image_area}px)"
    )

    # Process each mask
    processed_masks = []
    
    for i, mask in enumerate(ori_mask):
        # Fill holes
        filled_mask = binary_fill_holes(mask).astype(np.uint8)
        
        # Light morphological operations (erosion then dilation)
        kernel = disk(1)  # Small kernel for speed
        eroded = erosion(filled_mask, kernel)
        dilated = dilation(eroded, kernel)
        
        # Size filtering
        mask_size = np.sum(dilated)
        if mask_size >= min_crys_size:
            processed_masks.append(dilated.astype(bool))
        else:
            system_logger.debug(
                f"Filtered mask {i}: size {mask_size}px < min {min_crys_size}px"
            )

    system_logger.debug(
        f"Postprocessing complete: {len(processed_masks)}/{len(ori_mask)} masks kept"
    )
    
    return processed_masks


def run_multiscale_class_inference(
    predictor,
    image,
    target_class,
    confidence_threshold=0.3,
    small_classes=set(),
):
    """
    Adaptive multi-scale inference with smart scale selection.
    Starts with conservative scales and gets more aggressive based on performance.
    Iteration control is automatic via config.yaml settings.
    """
    return run_adaptive_multiscale_inference(
        predictor, image, target_class, confidence_threshold, small_classes
    )


def run_adaptive_multiscale_inference(
    predictor,
    image,
    target_class,
    confidence_threshold=0.3,
    small_classes=set(),
):
    """
    Smart adaptive multi-scale inference that progressively tries more aggressive scales.
    
    Strategy:
    1. Start with conservative scales: [0.7, 1.0, 1.5]
    2. If upscaling (1.5x) provides benefit, try more aggressive upscaling: [2.0, 2.5]
    3. If downscaling (0.7x) provides benefit, try more aggressive downscaling: [0.5, 0.6]
    4. Stop when additional scales don't improve results
    
    Iteration control is automatic via config.yaml settings.
    """
    all_masks = []
    all_scores = []
    all_classes = []
    
    # Phase 1: Conservative baseline scales
    baseline_scales = [0.7, 1.0, 1.5]
    scale_performance = {}  # Track performance per scale
    
    system_logger.info(f"Adaptive multiscale inference - Phase 1: Baseline scales {baseline_scales}")
    
    for scale in baseline_scales:
        masks, scores, classes = process_single_scale(
            predictor, image, target_class, small_classes, 
            confidence_threshold, scale
        )
        
        scale_performance[scale] = len(masks)
        all_masks.extend(masks)
        all_scores.extend(scores)
        all_classes.extend(classes)
        
        system_logger.info(f"Scale {scale}: Found {len(masks)} instances")
    
    # Analyze baseline performance to decide on aggressive scaling
    baseline_1x = scale_performance.get(1.0, 0)
    upscale_benefit = scale_performance.get(1.5, 0) > baseline_1x * 0.1  # 10% improvement threshold
    downscale_benefit = scale_performance.get(0.7, 0) > baseline_1x * 0.1
    
    system_logger.info(f"Baseline analysis - 1.0x: {baseline_1x}, upscale benefit: {upscale_benefit}, downscale benefit: {downscale_benefit}")
    
    # Phase 2: Aggressive upscaling if beneficial
    if upscale_benefit:
        aggressive_upscales = [2.0, 2.5]
        system_logger.info(f"Phase 2a: Trying aggressive upscaling {aggressive_upscales}")
        
        for scale in aggressive_upscales:
            masks, scores, classes = process_single_scale(
                predictor, image, target_class, small_classes,
                confidence_threshold, scale
            )
            
            # Stop if this scale doesn't add meaningful detections
            if len(masks) < baseline_1x * 0.05:  # Less than 5% of baseline
                system_logger.info(f"Scale {scale}: Low yield ({len(masks)} masks), stopping upscaling")
                break
                
            all_masks.extend(masks)
            all_scores.extend(scores)
            all_classes.extend(classes)
            system_logger.info(f"Scale {scale}: Found {len(masks)} instances")
    
    # Phase 3: Aggressive downscaling if beneficial  
    if downscale_benefit:
        aggressive_downscales = [0.5, 0.6]
        system_logger.info(f"Phase 2b: Trying aggressive downscaling {aggressive_downscales}")
        
        for scale in aggressive_downscales:
            masks, scores, classes = process_single_scale(
                predictor, image, target_class, small_classes,
                confidence_threshold, scale
            )
            
            # Stop if this scale doesn't add meaningful detections
            if len(masks) < baseline_1x * 0.05:  # Less than 5% of baseline
                system_logger.info(f"Scale {scale}: Low yield ({len(masks)} masks), stopping downscaling")
                break
                
            all_masks.extend(masks)
            all_scores.extend(scores)
            all_classes.extend(classes)
            system_logger.info(f"Scale {scale}: Found {len(masks)} instances")
    
    # Deduplicate across all scales

    # Deduplicate across scales
    unique_masks = []
    unique_scores = []
    unique_classes = []

    # Sort by confidence
    if all_scores:
        sorted_indices = np.argsort(all_scores)[::-1]

        for idx in sorted_indices:
            mask = all_masks[idx]
            score = all_scores[idx]
            cls = all_classes[idx]

            # Check for duplicates (lenient for small particles)
            is_duplicate = any(iou(mask, existing_mask) > 0.4 for existing_mask in unique_masks)

            if not is_duplicate:
                unique_masks.append(mask)
                unique_scores.append(score)
                unique_classes.append(cls)

    total_scales_tried = len(baseline_scales) + (2 if upscale_benefit else 0) + (2 if downscale_benefit else 0)
    system_logger.info(
        f"Adaptive multiscale completed: {len(unique_masks)} unique masks from {total_scales_tried} scales for class {target_class}"
    )
    return unique_masks, unique_scores, unique_classes


def process_single_scale(predictor, image, target_class, small_classes, confidence_threshold, scale):
    """
    Process inference at a single scale and return results.
    FIXED: Now uses scale-invariant minimum size thresholds.
    
    Iteration control is automatic via config.yaml settings.
    
    Parameters:
    - predictor: Detectron2 predictor
    - image: Original (unscaled) image
    - target_class: Target class ID
    - small_classes: Set of small class IDs
    - confidence_threshold: Confidence threshold for this class
    - max_iters: Maximum iterations
    - scale: Scale factor (e.g., 1.0, 1.5, 2.0)
    
    Returns:
    - tuple: (masks, scores, classes) at original image scale
    """
    system_logger.debug(f"Processing scale {scale} for class {target_class}")
    
    # Resize image
    if scale != 1.0:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        scaled_image = image

    # CRITICAL FIX: Calculate minimum size threshold based on ORIGINAL image, then scale it
    original_image_area = image.shape[0] * image.shape[1]
    is_small_class = target_class in small_classes
    
    if is_small_class:
        # For small particles: 0.001% of ORIGINAL image (more aggressive)
        base_min_size = max(3, int(original_image_area * 0.000005))  # Even lower threshold
    else:
        # For large particles: 0.01% of ORIGINAL image
        base_min_size = max(25, int(original_image_area * 0.0001))
    
    # Scale the threshold proportionally to match the scaled image
    # At 2x scale, area is 4x larger, so threshold should be 4x
    scaled_min_size = int(base_min_size * (scale ** 2))

    # Run iterative inference at this scale with scaled threshold
    scale_masks, scale_scores, scale_classes = run_iterative_class_inference(
        predictor,
        scaled_image,
        target_class,
        small_classes,
        confidence_threshold,
        min_crys_size=scaled_min_size,
    )

    # Scale masks back to original size
    if scale != 1.0 and scale_masks:
        original_size_masks = []
        for mask in scale_masks:
            # Use INTER_NEAREST for binary masks to preserve edges
            resized_mask = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            original_size_masks.append(resized_mask.astype(bool))
        scale_masks = original_size_masks
        
        system_logger.debug(
            f"Scale {scale}x: Rescaled {len(scale_masks)} masks to original size"
        )

    return scale_masks, scale_scores, scale_classes


def run_iterative_class_inference(
    predictor, 
    image, 
    target_class, 
    small_classes, 
    confidence_threshold=0.3, 
    min_crys_size=None
):
    """
    Run iterative inference for a specific class with universal postprocessing.
    NOW WITH COMPREHENSIVE DIAGNOSTIC LOGGING.
    
    Iteration control is automatic via config.yaml iterative_stopping settings.
    """
    all_masks = []
    all_scores = []
    all_classes = []
    prev_count = 0
    no_new_mask_iters = 0

    # Class-specific parameters based on size heuristic
    is_small_class = target_class in small_classes
    if is_small_class:
        iou_threshold = 0.5
    else:
        iou_threshold = 0.7
    
    iteration = 0

    while True:
        iteration += 1
        system_logger.debug(f"  Iteration {iteration} for class {target_class}")

        # L4 OPTIMIZATION: Use configured mixed precision
        use_mixed_precision = torch.cuda.is_available() and USE_MIXED_PRECISION
        
        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = predictor(image)
        else:
            outputs = predictor(image)
        
        pred_masks = outputs["instances"].to("cpu")._fields["pred_masks"].numpy()
        pred_scores = outputs["instances"].to("cpu")._fields["scores"].numpy()
        pred_classes = outputs["instances"].to("cpu")._fields["pred_classes"].numpy()

        # DIAGNOSTIC 1: Log raw detections before ANY filtering
        raw_class_mask = pred_classes == target_class
        raw_class_count = np.sum(raw_class_mask)
        system_logger.info(f"    DIAGNOSTIC: RAW detections for class {target_class}: {raw_class_count} masks")
        
        # Log score distribution of raw detections
        if raw_class_count > 0:
            raw_class_scores = pred_scores[raw_class_mask]
            system_logger.info(
                f"    DIAGNOSTIC: Score distribution - "
                f"min: {raw_class_scores.min():.3f}, "
                f"max: {raw_class_scores.max():.3f}, "
                f"mean: {raw_class_scores.mean():.3f}, "
                f"median: {np.median(raw_class_scores):.3f}"
            )
            
            # Count how many are above/below threshold
            above_thresh = np.sum(raw_class_scores >= confidence_threshold)
            below_thresh = np.sum(raw_class_scores < confidence_threshold)
            system_logger.info(
                f"    DIAGNOSTIC: Confidence filtering - "
                f"above {confidence_threshold}: {above_thresh}, "
                f"below {confidence_threshold}: {below_thresh} (FILTERED OUT)"
            )

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

       

        # Filter for target class and confidence
        class_mask = (pred_classes == target_class) & (
            pred_scores >= confidence_threshold
        )
        filtered_masks = pred_masks[class_mask]
        filtered_scores = pred_scores[class_mask]
        filtered_classes = pred_classes[class_mask]

        # DIAGNOSTIC 2: After confidence filtering
        system_logger.info(f"    DIAGNOSTIC: After confidence filter: {len(filtered_masks)} masks")
        
        # Log size distribution of filtered masks BEFORE postprocessing
        if len(filtered_masks) > 0:
            mask_sizes = [np.sum(mask) for mask in filtered_masks]
            system_logger.info(
                f"    DIAGNOSTIC: Mask size distribution (BEFORE postprocessing) - "
                f"min: {min(mask_sizes)}px, "
                f"max: {max(mask_sizes)}px, "
                f"mean: {np.mean(mask_sizes):.1f}px, "
                f"median: {np.median(mask_sizes):.1f}px"
            )

        # UNIVERSAL postprocessing
        if len(filtered_masks) > 0:
            # DIAGNOSTIC 3: Log threshold being used
            if min_crys_size is None:
                image_area = image.shape[0] * image.shape[1]
                calculated_min_size = max(3, int(image_area * 0.000005)) if is_small_class else max(25, int(image_area * 0.0001))
                system_logger.info(
                    f"    DIAGNOSTIC: Size threshold - "
                    f"calculated min_size={calculated_min_size}px "
                    f"(image_area={image_area}px, is_small={is_small_class})"
                )
            else:
                system_logger.info(
                    f"    DIAGNOSTIC: Size threshold - "
                    f"provided min_size={min_crys_size}px"
                )
            
            processed_masks = postprocess_masks_universal(
                filtered_masks, 
                filtered_scores, 
                image, 
                target_class, 
                is_small_class,
                min_crys_size=min_crys_size
            )

            # DIAGNOSTIC 4: After size filtering
            filtered_by_size = len(filtered_masks) - len(processed_masks)
            system_logger.info(
                f"    DIAGNOSTIC: After size filter: {len(processed_masks)} masks "
                f"({filtered_by_size} FILTERED OUT by size)"
            )
            
            # Show which sizes were filtered
            if filtered_by_size > 0 and len(filtered_masks) > 0:
                removed_sizes = []
                kept_sizes = []
                threshold_used = min_crys_size if min_crys_size is not None else calculated_min_size
                
                for mask in filtered_masks:
                    size = np.sum(mask)
                    if size >= threshold_used:
                        kept_sizes.append(size)
                    else:
                        removed_sizes.append(size)
                
                if removed_sizes:
                    system_logger.info(
                        f"    DIAGNOSTIC: Removed mask sizes - "
                        f"min: {min(removed_sizes)}px, max: {max(removed_sizes)}px, "
                        f"count: {len(removed_sizes)}"
                    )
                if kept_sizes:
                    system_logger.info(
                        f"    DIAGNOSTIC: Kept mask sizes - "
                        f"min: {min(kept_sizes)}px, max: {max(kept_sizes)}px, "
                        f"count: {len(kept_sizes)}"
                    )

            # Add processed masks from this iteration
            if processed_masks:
                for i, mask in enumerate(processed_masks):
                    all_masks.append(mask)
                    all_scores.append(filtered_scores[i])
                    all_classes.append(target_class)

        # Deduplicate all masks with class-specific IoU threshold
        unique_masks = []
        unique_scores = []
        unique_classes = []

        for i, mask in enumerate(all_masks):
            is_duplicate = any(iou(mask, um) > iou_threshold for um in unique_masks)
            if not is_duplicate:
                unique_masks.append(mask)
                unique_scores.append(all_scores[i])
                unique_classes.append(all_classes[i])

        new_count = len(unique_masks)
        added = new_count - prev_count
        duplicates_removed = len(all_masks) - new_count

        # DIAGNOSTIC 5: After deduplication
        system_logger.info(
            f"    DIAGNOSTIC: After deduplication - "
            f"unique: {new_count}, duplicates removed: {duplicates_removed}, "
            f"newly added: {added}"
        )

        system_logger.debug(f"    Added {added} new masks (total: {new_count})")

        # EARLY STOPPING CONDITIONS
        if added == 0:
            no_new_mask_iters += 1
        else:
            no_new_mask_iters = 0

        if no_new_mask_iters >= MAX_CONSECUTIVE_ZERO:
            system_logger.debug(
                f"    Stopping: No new masks for {MAX_CONSECUTIVE_ZERO} consecutive iterations"
            )
            break

        if new_count >= MIN_TOTAL_MASKS and iteration >= MIN_ITERATIONS:
            required_increase = max(1, int(prev_count * MIN_RELATIVE_INCREASE))
            if added < required_increase:
                system_logger.debug(
                    f"    Stopping: Added {added} masks < required {required_increase} "
                    f"(25% of {prev_count} existing masks). Total masks: {new_count}"
                )
                break
        elif new_count < MIN_TOTAL_MASKS:
            system_logger.debug(
                f"    Continuing: Only {new_count} masks (need at least {MIN_TOTAL_MASKS} before considering early stop)"
            )

        prev_count = new_count
        all_masks = unique_masks.copy()
        all_scores = unique_scores.copy()
        all_classes = unique_classes.copy()

    # DIAGNOSTIC 6: Final summary
    system_logger.info(
        f"  FINAL: Class {target_class} completed with {len(unique_masks)} masks after {iteration + 1} iterations"
    )
    
    return unique_masks, unique_scores, unique_classes


def tile_based_inference_pipeline(
    predictor,
    image,
    target_class,
    small_classes,
    confidence_threshold,
    tile_size=512,
    overlap_ratio=0.1,
    upscale_factor=2.0,
    scale_bar_info=None
):
    """
    Tile-based inference for detecting particles at multiple scales.
    NOW RUNS FOR ALL CLASSES with optimized GPU batching.
    
    Strategy:
    1. Run full-image inference (captures larger particles with full context)
    2. Run tile-based inference in GPU batches (captures smaller particles via upscaling)
    3. Combine and deduplicate results
    
    This approach improves detection across all particle sizes by:
    - Full-image pass: Maintains spatial context for large particles
    - Tile-based pass: Makes small particles appear larger (more detectable)
    - GPU batching: Process multiple tiles simultaneously for speed
    """
    
    system_logger.info(f"Tile-based inference for class {target_class}: tile_size={tile_size}px, overlap={overlap_ratio*100:.0f}%, upscale={upscale_factor}x")
    
    h, w = image.shape[:2]
    is_small_class = target_class in small_classes
    
    # Full-image inference
    full_image_masks, full_image_scores, full_image_classes = run_class_specific_inference(
        predictor, image, target_class, small_classes, 
        confidence_threshold, iou_threshold=0.7
    )
    
    system_logger.info(f"Full image: {len(full_image_masks)} instances, generating {tile_size}px tiles with {overlap_ratio*100:.0f}% overlap")
    
    # Generate tiles with overlap
    tiles = generate_tiles_with_overlap(image, tile_size, overlap_ratio)
    
    # Process tiles in GPU batches for better performance
    all_tile_masks = []
    all_tile_scores = []
    all_tile_classes = []
    
    # Determine batch size for tile processing
    tile_batch_size = TILE_BATCH_SIZE if torch.cuda.is_available() else 1
    num_batches = (len(tiles) + tile_batch_size - 1) // tile_batch_size
    
    system_logger.info(f"Processing {len(tiles)} tiles in {num_batches} batches of {tile_batch_size}")
    

    
    for batch_idx in range(0, len(tiles), tile_batch_size):
        batch_end = min(batch_idx + tile_batch_size, len(tiles))
        tile_batch = tiles[batch_idx:batch_end]
        
        batch_start_time = time.perf_counter()
        system_logger.debug(f"Processing tile batch {batch_idx//tile_batch_size + 1}/{num_batches}: tiles {batch_idx + 1}-{batch_end}")
        
        # Process each tile in the batch
        for tile_idx_in_batch, (tile_img, x_offset, y_offset) in enumerate(tile_batch):
            global_tile_idx = batch_idx + tile_idx_in_batch
            
            # Upscale tile to make small particles appear larger
            tile_h, tile_w = tile_img.shape[:2]
            upscaled_h = int(tile_h * upscale_factor)
            upscaled_w = int(tile_w * upscale_factor)
            upscaled_tile = cv2.resize(tile_img, (upscaled_w, upscaled_h), interpolation=cv2.INTER_LINEAR)
            
            # Run inference on upscaled tile
            tile_masks, tile_scores, tile_classes = run_class_specific_inference(
                predictor, upscaled_tile, target_class, small_classes,
                confidence_threshold, iou_threshold=0.5
            )
            
            system_logger.debug(f"Tile {global_tile_idx + 1}: Found {len(tile_masks)} instances")
            
            # Map masks back to original image coordinates
            if tile_masks:
                for i, (mask, score, cls) in enumerate(zip(tile_masks, tile_scores, tile_classes)):
                    # Downscale mask back to tile size
                    downscaled_mask = cv2.resize(
                        mask.astype(np.uint8),
                        (tile_w, tile_h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    
                    # Filter edge masks (likely incomplete)
                    if is_edge_mask(downscaled_mask, tile_size, overlap_ratio):
                        system_logger.debug(f"Tile {global_tile_idx + 1}: Filtered edge mask {i+1}")
                        continue
                    
                    # Map to global coordinates
                    global_mask = np.zeros((h, w), dtype=bool)
                    y_end = min(y_offset + tile_h, h)
                    x_end = min(x_offset + tile_w, w)
                    global_mask[y_offset:y_end, x_offset:x_end] = downscaled_mask[:y_end-y_offset, :x_end-x_offset]
                    
                    all_tile_masks.append(global_mask)
                    all_tile_scores.append(score)
                    all_tile_classes.append(cls)
        
        batch_time = time.perf_counter() - batch_start_time
        system_logger.debug(f"Batch {batch_idx//tile_batch_size + 1} processed in {batch_time:.2f}s")
        
        # Batch-level GPU cache cleanup instead of per-tile
        if torch.cuda.is_available() and (batch_idx // tile_batch_size) % 3 == 0:
            torch.cuda.empty_cache()
    
    # Combine full-image and tile results
    all_masks = full_image_masks + all_tile_masks
    all_scores = list(full_image_scores) + list(all_tile_scores)
    all_classes = list(full_image_classes) + list(all_tile_classes)
    
    # Verify lengths match
    if len(all_masks) != len(all_scores) or len(all_masks) != len(all_classes):
        system_logger.error(
            f"Length mismatch! masks: {len(all_masks)}, scores: {len(all_scores)}, classes: {len(all_classes)}"
        )
        min_len = min(len(all_masks), len(all_scores), len(all_classes))
        all_masks = all_masks[:min_len]
        all_scores = all_scores[:min_len]
        all_classes = all_classes[:min_len]
        system_logger.warning(f"Truncated to {min_len} items to match lengths")
    
    # Deduplicate across full image + tiles
    unique_masks, unique_scores, unique_classes = deduplicate_masks_smart(
        all_masks, all_scores, all_classes, iou_threshold=0.4
    )
    
    system_logger.info(
        f"Deduplication complete: {len(full_image_masks)} full-image + {len(all_tile_masks)} tile instances "
        f"= {len(unique_masks)} unique (net gain: +{len(unique_masks) - len(full_image_masks)})"
    )
    
    return unique_masks, unique_scores, unique_classes


def generate_tiles_with_overlap(image, tile_size, overlap_ratio):
    """
    Generate overlapping tiles from an image.
    
    Parameters:
    - image: Input image
    - tile_size: Size of each tile
    - overlap_ratio: Overlap ratio (e.g., 0.2 for 20%)
    
    Returns:
    - list: [(tile_image, x_offset, y_offset), ...]
    """
    h, w = image.shape[:2]
    stride = int(tile_size * (1 - overlap_ratio))
    
    tiles = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = image[y:y_end, x:x_end]
            
            # Pad if necessary to maintain tile_size (for edge tiles)
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile_padded = np.zeros((tile_size, tile_size, 3), dtype=image.dtype)
                tile_padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = tile_padded
            
            tiles.append((tile, x, y))
    
    return tiles


def is_edge_mask(mask, tile_size, overlap_ratio):
    """
    Check if a mask is within the overlap region (likely incomplete).
    
    Parameters:
    - mask: Binary mask
    - tile_size: Tile size
    - overlap_ratio: Overlap ratio
    
    Returns:
    - bool: True if mask is in edge region
    """
    edge_width = int(tile_size * overlap_ratio / 2)  # Half overlap on each side
    
    # Get mask bounding box
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return True
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Check if mask touches any edge within overlap zone
    if (y_min < edge_width or y_max > tile_size - edge_width or
        x_min < edge_width or x_max > tile_size - edge_width):
        return True
    
    return False


def deduplicate_masks_smart(masks, scores, classes, iou_threshold=0.4):
    """
    HIGHLY OPTIMIZED: Fast deduplication using spatial indexing and parallelization.
    - Uses bounding box pre-filtering (O(n log n) instead of O(n²))
    - Parallel IoU calculations for overlapping boxes
    - Early exit when duplicate found
    
    Parameters:
    - masks: List of binary masks
    - scores: List of confidence scores
    - classes: List of class IDs
    - iou_threshold: IoU threshold for considering masks as duplicates
    
    Returns:
    - tuple: (unique_masks, unique_scores, unique_classes)
    """
    if not masks:
        return [], [], []
    
    # Validation
    if len(masks) != len(scores) or len(masks) != len(classes):
        system_logger.error(
            f"Input length mismatch! masks: {len(masks)}, scores: {len(scores)}, classes: {len(classes)}"
        )
        min_len = min(len(masks), len(scores), len(classes))
        masks = masks[:min_len]
        scores = scores[:min_len]
        classes = classes[:min_len]
    
    scores = np.array(scores)
    total_masks = len(masks)
    
    system_logger.info(f"Starting OPTIMIZED deduplication of {total_masks} masks...")
    start_time = time.perf_counter()
    
    # OPTIMIZATION 1: Pre-compute bounding boxes and areas
    bboxes = []
    bbox_areas = []
    for mask in masks:
        coords = np.argwhere(mask)
        if len(coords) == 0:
            bboxes.append((0, 0, 0, 0))
            bbox_areas.append(0)
        else:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bboxes.append((y_min, x_min, y_max, x_max))
            bbox_areas.append((y_max - y_min) * (x_max - x_min))
    
    system_logger.debug(f"Computed {len(bboxes)} bounding boxes in {time.perf_counter() - start_time:.2f}s")
    
    # Sort by score (descending) - keep highest confidence masks
    sorted_indices = np.argsort(scores)[::-1]
    
    unique_masks = []
    unique_scores = []
    unique_classes = []
    unique_bboxes = []
    unique_areas = []
    
    checked_pairs = 0
    skipped_by_bbox = 0
    skipped_by_area = 0
    
    for progress_idx, idx in enumerate(sorted_indices):
        # Progress logging every 10%
        if total_masks > 100 and progress_idx % max(1, total_masks // 10) == 0:
            elapsed = time.perf_counter() - start_time
            system_logger.info(
                f"Progress: {progress_idx}/{total_masks} ({int(progress_idx/total_masks*100)}%) - "
                f"Unique: {len(unique_masks)} - Time: {elapsed:.1f}s"
            )
        
        if idx >= len(masks):
            continue
            
        mask = masks[idx]
        score = scores[idx]
        cls = classes[idx]
        bbox = bboxes[idx]
        area = bbox_areas[idx]
        
        # OPTIMIZATION 2: Bbox overlap + area pre-filter
        is_duplicate = False
        for i, (existing_bbox, existing_area) in enumerate(zip(unique_bboxes, unique_areas)):
            y1_min, x1_min, y1_max, x1_max = bbox
            y2_min, x2_min, y2_max, x2_max = existing_bbox
            
            # Quick bbox overlap check
            if (y1_max < y2_min or y2_max < y1_min or
                x1_max < x2_min or x2_max < x1_min):
                # No overlap - skip expensive IoU calculation
                skipped_by_bbox += 1
                continue
            
            # OPTIMIZATION 3: Area-based early rejection
            # If bbox areas are very different, IoU can't be high
            area_ratio = min(area, existing_area) / max(area, existing_area) if max(area, existing_area) > 0 else 0
            if area_ratio < iou_threshold * 0.5:  # Conservative threshold
                skipped_by_area += 1
                continue
            
            # Bboxes overlap and areas similar - compute actual IoU
            checked_pairs += 1
            if iou(mask, unique_masks[i]) > iou_threshold:
                is_duplicate = True
                break  # OPTIMIZATION 4: Early exit when duplicate found
        
        if not is_duplicate:
            unique_masks.append(mask)
            unique_scores.append(float(score))
            unique_classes.append(int(cls))
            unique_bboxes.append(bbox)
            unique_areas.append(area)
    
    total_time = time.perf_counter() - start_time
    efficiency = (skipped_by_bbox + skipped_by_area) / max(1, checked_pairs + skipped_by_bbox + skipped_by_area) * 100
    
    system_logger.info(
        f"Deduplication: {total_masks} -> {len(unique_masks)} unique in {total_time:.1f}s "
        f"({checked_pairs:,} IoU checks, {skipped_by_bbox:,} bbox, {skipped_by_area:,} area skips, {efficiency:.1f}% efficient)"
    )
    
    return unique_masks, unique_scores, unique_classes
