"""
Constants and default values for the deepEMIA project.

This module centralizes all magic numbers, thresholds, and default values
used throughout the project for better maintainability.
"""

from typing import Dict, Any


# Default detection and processing thresholds
class DefaultThresholds:
    """Default threshold values for various operations."""

    # Mask processing
    SCORE_THRESHOLD = 0.5
    IOA_THRESHOLD = 0.7  # Intersection over Area
    IOU_THRESHOLD = 0.7  # Intersection over Union
    MIN_CRYSTAL_SIZE = 2  # Minimum crystal size for postprocessing

    # Scale bar detection
    SCALEBAR_INTENSITY = 200
    SCALEBAR_PROXIMITY = 50

    # Image processing
    CANNY_LOWER = 50
    CANNY_UPPER = 150
    CANNY_APERTURE = 3

    # OCR confidence
    OCR_MIN_CONFIDENCE = 0.5

    # Contour filtering
    MIN_CONTOUR_AREA = 100
    MAX_CONTOUR_AREA = 50000


# Default hyperparameters for model training
class DefaultHyperparameters:
    """Default hyperparameters for model training."""

    BASE_LR = 0.00025
    IMS_PER_BATCH = 2
    WARMUP_ITERS = 1000
    WARMUP_FACTOR = 1e-3
    GAMMA = 0.1
    BATCH_SIZE_PER_IMAGE = 64
    MAX_ITER = None  # Will be calculated based on dataset size


# File and directory patterns
class FilePatterns:
    """File patterns and extensions."""

    IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif")
    LABEL_EXTENSIONS = (".json",)
    MODEL_EXTENSIONS = (".pth", ".pkl")

    # File naming patterns
    LOG_FILE_PATTERN = "system_{timestamp}.log"
    MODEL_FILE_PATTERN = "{dataset_name}_{backbone}_model.pth"


# Processing limits and constraints
class ProcessingLimits:
    """Limits for processing operations."""

    MAX_IMAGE_SIZE = (4096, 4096)  # Maximum image dimensions
    MAX_BATCH_SIZE = 16
    MAX_ITERATIONS = 100

    # Memory and performance
    MAX_WORKERS = 8
    CHUNK_SIZE = 1000
    TIMEOUT_SECONDS = 3600  # 1 hour

    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # seconds
    EXPONENTIAL_BACKOFF = True


# Scale bar ROI defaults
class ScaleBarDefaults:
    """Default scale bar region of interest settings."""

    X_START_FACTOR = 0.667
    Y_START_FACTOR = 0.866
    WIDTH_FACTOR = 1.0
    HEIGHT_FACTOR = 0.067


# Color and visualization settings
class VisualizationDefaults:
    """Default settings for visualization."""

    # Colors (BGR format for OpenCV)
    BBOX_COLOR = (0, 255, 0)  # Green
    MASK_COLOR = (255, 0, 0)  # Blue
    TEXT_COLOR = (255, 255, 255)  # White
    SCALE_BAR_COLOR = (0, 0, 255)  # Red

    # Font settings
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1

    # Transparency
    MASK_ALPHA = 0.3
    OVERLAY_ALPHA = 0.7


# Dataset splitting defaults
class DatasetDefaults:
    """Default settings for dataset operations."""

    TEST_SIZE = 0.2  # 20% for testing
    VALIDATION_SIZE = 0.1  # 10% for validation
    RANDOM_SEED = 42

    # Augmentation probabilities
    FLIP_PROB = 0.5
    ROTATION_PROB = 0.3
    BRIGHTNESS_PROB = 0.2


# Network and communication settings
class NetworkDefaults:
    """Default network and communication settings."""

    CONNECTION_TIMEOUT = 30  # seconds
    READ_TIMEOUT = 300  # seconds
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2.0


# Measurement defaults
class MeasurementDefaults:
    """Default settings for measurements and analysis."""

    PIXELS_PER_METRIC = 1.0
    UM_PER_PIXEL = 1.0

    # Contrast distribution bins
    HISTOGRAM_BINS = 256
    HISTOGRAM_RANGE = (0, 255)

    # Statistical percentiles
    D10_PERCENTILE = 0.10
    D50_PERCENTILE = 0.50
    D90_PERCENTILE = 0.90


# Configuration for various tasks
TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "prepare": {
        "default_test_size": DatasetDefaults.TEST_SIZE,
        "random_seed": DatasetDefaults.RANDOM_SEED,
    },
    "train": {
        "default_lr": DefaultHyperparameters.BASE_LR,
        "default_batch_size": DefaultHyperparameters.IMS_PER_BATCH,
        "max_iterations": ProcessingLimits.MAX_ITERATIONS,
    },
    "inference": {
        "default_threshold": DefaultThresholds.SCORE_THRESHOLD,
        "iou_threshold": DefaultThresholds.IOU_THRESHOLD,
        "max_iterations": ProcessingLimits.MAX_ITERATIONS,
    },
    "evaluate": {
        "confidence_threshold": DefaultThresholds.SCORE_THRESHOLD,
        "iou_threshold": DefaultThresholds.IOU_THRESHOLD,
    },
}


# Error messages
class ErrorMessages:
    """Standard error messages."""

    CONFIG_NOT_FOUND = "Configuration file not found: {path}"
    INVALID_CONFIG = "Invalid configuration: {details}"
    MODEL_LOAD_FAILED = "Failed to load model: {path}"
    DATASET_NOT_FOUND = "Dataset not found: {name}"
    INSUFFICIENT_SPACE = (
        "Insufficient disk space: {required} GB required, {available} GB available"
    )
    NETWORK_ERROR = "Network operation failed: {details}"
    FILE_OPERATION_FAILED = "File operation failed: {operation} on {path}"


# Success messages
class SuccessMessages:
    """Standard success messages."""

    CONFIG_LOADED = "Configuration loaded successfully from {path}"
    MODEL_SAVED = "Model saved successfully to {path}"
    DATASET_PREPARED = (
        "Dataset prepared successfully: {train_size} train, {test_size} test samples"
    )
    TRAINING_COMPLETE = "Training completed successfully in {duration:.2f} seconds"
    INFERENCE_COMPLETE = (
        "Inference completed on {num_images} images in {duration:.2f} seconds"
    )
    EVALUATION_COMPLETE = "Evaluation completed: AP = {ap:.3f}"
