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
