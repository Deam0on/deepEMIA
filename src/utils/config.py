"""
Configuration Loader Utility

Provides a singleton-style loader for the main YAML configuration file
with support for dataset-specific overrides.
Ensures the config is loaded only once and reused throughout the project.
Includes configuration validation for security and reliability.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

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


def get_all_datasets_with_variants() -> Dict[str, List[str]]:
    """
    Get all datasets and their available configuration variants.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping dataset names to list of variants
    """
    config_dir = Path.home() / "deepEMIA" / "config" / "datasets"
    datasets = {}
    
    if not config_dir.exists():
        return datasets
    
    # Iterate through all subdirectories
    for dataset_dir in config_dir.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
            # Look for YAML files in each dataset directory
            variants = [f.stem for f in dataset_dir.glob("*.yaml")]
            if variants:
                datasets[dataset_dir.name] = sorted(variants)
    
    return datasets


def list_dataset_configs() -> List[str]:
    """
    List all available dataset configuration directories.
    
    Returns:
        List[str]: List of dataset names with configurations
    """
    config_dir = Path.home() / "deepEMIA" / "config" / "datasets"
    
    if not config_dir.exists():
        return []
    
    datasets = []
    for dataset_dir in config_dir.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
            # Check if it has YAML files
            if list(dataset_dir.glob("*.yaml")):
                datasets.append(dataset_dir.name)
    
    return sorted(datasets)


def create_dataset_config(dataset_name: str, template: str = "template") -> Path:
    """
    Create a new dataset configuration from a template.
    
    Parameters:
        dataset_name: Name of the dataset
        template: Template to use (template, polyhipes_tommy, update_test)
    
    Returns:
        Path: Path to the created config file
    """
    config_dir = Path.home() / "deepEMIA" / "config" / "datasets"
    dataset_dir = config_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Load template
    template_dir = Path(__file__).parent.parent.parent / "config" / "datasets.example"
    template_file = template_dir / f"{template}.yaml"
    
    if not template_file.exists():
        # Fallback to template.yaml
        template_file = template_dir / "template.yaml"
    
    # Copy template to dataset directory with "default" variant name
    output_file = dataset_dir / "default.yaml"
    
    if template_file.exists():
        with open(template_file, 'r') as f:
            content = f.read()
        
        with open(output_file, 'w') as f:
            f.write(content)
    else:
        # Create minimal config if template not found
        minimal_config = f"""# Dataset-specific configuration for {dataset_name}

metadata:
  name: "{dataset_name}"
  description: "Configuration for {dataset_name}"
  created: "{datetime.now().isoformat()}"
"""
        with open(output_file, 'w') as f:
            f.write(minimal_config)
    
    return output_file

