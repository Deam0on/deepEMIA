"""
Configuration schema validation utilities.

This module provides validation for the project configuration to ensure
all required fields are present and have valid values.
"""

from typing import Dict, Any, List
from pathlib import Path

from src.utils.logger_utils import system_logger


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


# Configuration schema definition
REQUIRED_SECTIONS = {
    'bucket': str,
    'paths': dict,
    'scale_bar_rois': dict,
    'scalebar_thresholds': dict,
    'rcnn_hyperparameters': dict,
    'inference_settings': dict,
    'l4_performance_optimizations': dict
}

REQUIRED_PATHS = [
    'main_script',
    'split_dir',
    'category_json',
    'eta_file',
    'logs_dir',
    'output_dir',
    'local_dataset_root'
]

# Optional paths that won't trigger warnings
OPTIONAL_PATHS = ['dataset_configs_dir']

REQUIRED_SCALEBAR_KEYS = ['intensity', 'proximity']

# Optional scalebar threshold keys that won't trigger warnings
OPTIONAL_SCALEBAR_KEYS = ['merge_gap', 'min_line_length', 'edge_margin_factor']

REQUIRED_INFERENCE_KEYS = [
    'use_class_specific_inference',
    'confidence_mode',
    'class_specific_settings',
    'ensemble_settings'
]

REQUIRED_L4_KEYS = [
    'inference_batch_size',
    'measurement_batch_size',
    'clear_cache_frequency',
    'max_memory_usage'
]


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates the configuration dictionary.

    Args:
        config (dict): Raw configuration dictionary

    Returns:
        dict: Validated configuration dictionary

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    # Check required top-level sections
    for section, expected_type in REQUIRED_SECTIONS.items():
        if section not in config:
            raise ConfigValidationError(f"Missing required configuration section: {section}")
        if not isinstance(config[section], expected_type):
            raise ConfigValidationError(
                f"Configuration section '{section}' must be of type {expected_type.__name__}"
            )

    # Validate paths section
    for path_key in REQUIRED_PATHS:
        if path_key not in config['paths']:
            raise ConfigValidationError(f"Missing required path configuration: {path_key}")

    # Validate scale_bar_rois has default
    if 'default' not in config['scale_bar_rois']:
        raise ConfigValidationError("scale_bar_rois must contain a 'default' configuration")

    # Validate scalebar_thresholds - only check required keys
    for key in REQUIRED_SCALEBAR_KEYS:
        if key not in config['scalebar_thresholds']:
            raise ConfigValidationError(f"Missing required scalebar_thresholds key: {key}")

    # Validate rcnn_hyperparameters structure
    if 'default' not in config['rcnn_hyperparameters']:
        raise ConfigValidationError("rcnn_hyperparameters must contain a 'default' configuration")

    # Validate inference_settings
    for key in REQUIRED_INFERENCE_KEYS:
        if key not in config['inference_settings']:
            raise ConfigValidationError(f"Missing required inference_settings key: {key}")

    # Validate L4 optimizations
    for key in REQUIRED_L4_KEYS:
        if key not in config['l4_performance_optimizations']:
            raise ConfigValidationError(f"Missing required l4_performance_optimizations key: {key}")

    system_logger.debug("Configuration validation passed")
    return config
