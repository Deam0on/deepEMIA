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


def load_dataset_config(dataset_name: str, variant: str = "default") -> Optional[Dict[str, Any]]:
    """
    Load dataset-specific configuration with variant support.
    
    Parameters:
    - dataset_name: Name of the dataset
    - variant: Configuration variant (default: "default")
    
    Returns:
    - Dataset configuration dictionary or None if not found
    """
    cache_key = f"{dataset_name}:{variant}"
    if cache_key in _dataset_configs:
        return _dataset_configs[cache_key]
    
    config_dir = Path.home() / "deepEMIA" / "config" / "datasets"
    
    # Try directory-based variant first (preferred)
    variant_file = config_dir / dataset_name / f"{variant}.yaml"
    
    # Fallback to flat structure with naming convention
    if not variant_file.exists():
        variant_file = config_dir / f"{dataset_name}_{variant}.yaml"
    
    if not variant_file.exists():
        system_logger.debug(f"No config found for '{dataset_name}' variant '{variant}'")
        return None
    
    try:
        with open(variant_file, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        _dataset_configs[cache_key] = dataset_config
        system_logger.info(f"Loaded config: {dataset_name}/{variant}")
        return dataset_config
        
    except Exception as e:
        system_logger.error(f"Error loading config for '{dataset_name}:{variant}': {e}")
        return None


def get_config(dataset_name: str = None, variant: str = "default") -> Dict[str, Any]:
    """
    Loads configuration with dataset and variant support.

    Parameters:
    - dataset_name: Optional dataset name
    - variant: Configuration variant (default: "default")

    Returns:
        dict: The loaded and validated configuration dictionary.
    """
    global _config
    
    # Load base config once
    if _config is None:
        config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"
        try:
            with open(config_path, "r") as f:
                raw_config = yaml.safe_load(f)

            try:
                from src.utils.config_validator import validate_config
                _config = validate_config(raw_config)
                system_logger.info(f"Loaded and validated configuration from {config_path}")
            except ImportError:
                _config = raw_config
                system_logger.warning("Configuration validator not available")
            except Exception as e:
                system_logger.error(f"Configuration validation failed: {e}")
                _config = raw_config

        except FileNotFoundError:
            system_logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            system_logger.error(f"Error parsing configuration file: {e}")
            raise
    
    # If no dataset specified, return base config
    if dataset_name is None:
        return _config
    
    # Load and merge dataset-specific config with variant
    dataset_config = load_dataset_config(dataset_name, variant)
    
    if dataset_config is None:
        return _config
    
    # Deep merge configs
    merged_config = deep_merge(_config, {})
    merged_config = deep_merge(merged_config, dataset_config)
    
    system_logger.debug(f"Merged config for dataset '{dataset_name}' variant '{variant}'")
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


def list_dataset_variants(dataset_name: str) -> list:
    """
    List all available configuration variants for a dataset.
    
    Parameters:
    - dataset_name: Name of the dataset
    
    Returns:
    - List of available variant names
    """
    config_dir = Path.home() / "deepEMIA" / "config" / "datasets"
    variants = []
    
    # Check directory-based structure
    dataset_dir = config_dir / dataset_name
    if dataset_dir.exists() and dataset_dir.is_dir():
        for config_file in dataset_dir.glob("*.yaml"):
            variants.append(config_file.stem)
    
    # Check flat structure
    for config_file in config_dir.glob(f"{dataset_name}_*.yaml"):
        variant = config_file.stem.replace(f"{dataset_name}_", "")
        if variant not in variants:
            variants.append(variant)
    
    return sorted(variants) if variants else ["default"]


def get_all_datasets_with_variants() -> Dict[str, List[str]]:
    """
    Get all datasets and their available variants.
    
    Returns:
    - Dictionary mapping dataset names to list of variants
    """
    config_dir = Path.home() / "deepEMIA" / "config" / "datasets"
    datasets = {}
    
    if not config_dir.exists():
        return datasets
    
    # Directory-based variants
    for dataset_dir in config_dir.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
            variants = [f.stem for f in dataset_dir.glob("*.yaml")]
            if variants:
                datasets[dataset_dir.name] = sorted(variants)
    
    # Flat structure variants
    for config_file in config_dir.glob("*.yaml"):
        if '_' in config_file.stem:
            parts = config_file.stem.split('_', 1)
            if len(parts) == 2:
                dataset_name, variant = parts
                if dataset_name not in datasets:
                    datasets[dataset_name] = []
                if variant not in datasets[dataset_name]:
                    datasets[dataset_name].append(variant)
    
    return datasets


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

