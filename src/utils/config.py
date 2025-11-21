"""
Configuration Loader Utility

Provides a singleton-style loader for the main YAML configuration file
with support for dataset-specific overrides.
Ensures the config is loaded only once and reused throughout the project.
Includes configuration validation for security and reliability.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from src.utils.logger_utils import system_logger

_config = None
_dataset_configs = {}  # Cache for dataset-specific configs


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries, with override values taking precedence.
    
    Parameters:
    - base: Base dictionary
    - override: Override dictionary
    
    Returns:
    - Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def load_dataset_config(dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    Load dataset-specific configuration if it exists.
    
    Parameters:
    - dataset_name: Name of the dataset
    
    Returns:
    - Dataset configuration dictionary or None if not found
    """
    if dataset_name in _dataset_configs:
        return _dataset_configs[dataset_name]
    
    # Try to find dataset config file
    config_dir = Path.home() / "deepEMIA" / "config" / "datasets"
    config_file = config_dir / f"{dataset_name}.yaml"
    
    if not config_file.exists():
        system_logger.debug(f"No dataset-specific config found for '{dataset_name}'")
        return None
    
    try:
        with open(config_file, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        _dataset_configs[dataset_name] = dataset_config
        system_logger.info(f"Loaded dataset-specific config for '{dataset_name}'")
        return dataset_config
        
    except Exception as e:
        system_logger.error(f"Error loading dataset config for '{dataset_name}': {e}")
        return None


def get_config(dataset_name: str = None) -> Dict[str, Any]:
    """Load config with consistent dataset override handling."""
    global _config
    
    if _config is None:
        config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"
        try:
            with open(config_path, "r") as f:
                raw_config = yaml.safe_load(f)

            # Validate configuration
            try:
                from src.utils.config_validator import validate_config
                _config = validate_config(raw_config)
                system_logger.info(f"Loaded and validated configuration from {config_path}")
            except ImportError:
                _config = raw_config
                system_logger.warning("Configuration validator not available, using unvalidated config")
            except Exception as e:
                system_logger.error(f"Configuration validation failed: {e}")
                _config = raw_config
                system_logger.warning("Using unvalidated configuration due to validation error")

        except FileNotFoundError:
            system_logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            system_logger.error(f"Error parsing configuration file: {e}")
            raise
    
    if dataset_name is None:
        return _config
    
    dataset_config = load_dataset_config(dataset_name)
    if dataset_config is None:
        return _config
    
    merged_config = deep_merge(_config, {})
    
    # === CONSISTENT NAMING FIX ===
    # Map inference_overrides -> inference_settings for consistency
    if 'inference_overrides' in dataset_config:
        if 'inference_settings' not in merged_config:
            merged_config['inference_settings'] = {}
        
        # Deep merge, ensuring spatial_constraints are properly nested
        override_config = dataset_config['inference_overrides']
        merged_config['inference_settings'] = deep_merge(
            merged_config['inference_settings'],
            override_config
        )
        
        system_logger.debug(f"Applied inference_overrides for '{dataset_name}'")
    
    # Handle scale_bar_roi
    if 'scale_bar_roi' in dataset_config:
        if 'scale_bar_rois' not in merged_config:
            merged_config['scale_bar_rois'] = {}
        merged_config['scale_bar_rois'][dataset_name] = dataset_config['scale_bar_roi']
    
    # Handle scalebar_thresholds override
    if 'scalebar_thresholds' in dataset_config:
        merged_config['scalebar_thresholds'] = deep_merge(
            merged_config.get('scalebar_thresholds', {}),
            dataset_config['scalebar_thresholds']
        )
    
    # Handle spatial constraints
    if 'spatial_constraints' in dataset_config:
        if 'spatial_constraints' not in merged_config.get('inference_settings', {}):
            if 'inference_settings' not in merged_config:
                merged_config['inference_settings'] = {}
            merged_config['inference_settings']['spatial_constraints'] = {}
        merged_config['inference_settings']['spatial_constraints'][dataset_name] = \
            dataset_config['spatial_constraints']
    
    # Handle rcnn_hyperparameters
    if 'rcnn_hyperparameters' in dataset_config:
        if 'best' not in merged_config['rcnn_hyperparameters']:
            merged_config['rcnn_hyperparameters']['best'] = {}
        
        for key in ['best_R50', 'best_R101']:
            if key in dataset_config['rcnn_hyperparameters']:
                backbone = key.replace('best_', '')
                merged_config['rcnn_hyperparameters']['best'][backbone] = \
                    dataset_config['rcnn_hyperparameters'][key]
    
    system_logger.debug(f"Merged config for dataset '{dataset_name}'")
    return merged_config


def list_dataset_configs() -> list:
    """
    List all available dataset-specific configuration files.
    
    Returns:
    - List of dataset names with configs
    """
    config_dir = Path.home() / "deepEMIA" / "config" / "datasets"
    
    if not config_dir.exists():
        return []
    
    configs = []
    for config_file in config_dir.glob("*.yaml"):
        configs.append(config_file.stem)
    
    return sorted(configs)


def create_dataset_config(dataset_name: str, template: str = "template") -> Path:
    """
    Create a new dataset-specific config from template.
    
    Parameters:
    - dataset_name: Name for the new dataset config
    - template: Template to use ('template' or existing dataset name)
    
    Returns:
    - Path to created config file
    """
    config_dir = Path.home() / "deepEMIA" / "config" / "datasets"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    target_file = config_dir / f"{dataset_name}.yaml"
    
    if target_file.exists():
        system_logger.warning(f"Dataset config already exists: {target_file}")
        return target_file
    
    # Load template - check example directory first, then datasets directory
    example_dir = Path.home() / "deepEMIA" / "config" / "datasets.example"
    
    if template in ["template", "polyhipes_tommy", "update_test"]:
        # Look in examples directory
        template_file = example_dir / f"{template}.yaml"
    else:
        # Look for existing dataset config as template
        template_file = config_dir / f"{template}.yaml"
    
    if not template_file.exists():
        system_logger.error(f"Template not found: {template_file}")
        raise FileNotFoundError(f"Template not found: {template_file}")
    
    # Copy template
    with open(template_file, 'r') as f:
        template_content = f.read()
    
    # Replace template name with actual dataset name in metadata
    template_content = template_content.replace(f'name: "{template}"', f'name: "{dataset_name}"')
    template_content = template_content.replace(f"name: '{template}'", f"name: '{dataset_name}'")
    
    with open(target_file, 'w') as f:
        f.write(template_content)
    
    system_logger.info(f"Created dataset config: {target_file}")
    return target_file

