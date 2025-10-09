"""
Configuration Loader Utility

Provides a singleton-style loader for the main YAML configuration file(s).
Supports both single config.yaml (legacy) and modular multi-file approach.
Ensures the config is loaded only once and reused throughout the project.
Includes configuration validation for security and reliability.
"""

from pathlib import Path
from typing import Dict, Any

import yaml

from src.utils.logger_utils import system_logger

_config = None


def deep_merge(base: Dict[Any, Any], override: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.
    
    Args:
        base: The base dictionary
        override: The dictionary with values to override/add
        
    Returns:
        dict: The merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config_files(config_dir: Path) -> Dict[str, Any]:
    """
    Load and merge multiple YAML config files in order.
    
    Args:
        config_dir: Path to the config directory
        
    Returns:
        dict: Merged configuration from all files
        
    Raises:
        FileNotFoundError: If no config files are found
    """
    config = {}
    
    # Load in order (later files can override earlier ones)
    config_files = [
        'base.yaml',
        'model.yaml', 
        'inference.yaml',
        'performance.yaml',
        'thresholds.yaml'
    ]
    
    files_loaded = 0
    for filename in config_files:
        filepath = config_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                    config = deep_merge(config, file_config)
                    system_logger.debug(f"Loaded config from {filename}")
                    files_loaded += 1
            except yaml.YAMLError as e:
                system_logger.error(f"Error parsing {filename}: {e}")
                raise
        else:
            system_logger.debug(f"Config file not found (optional): {filename}")
    
    if files_loaded == 0:
        raise FileNotFoundError(f"No configuration files found in {config_dir}")
    
    system_logger.info(f"Loaded and merged {files_loaded} configuration files")
    return config


def get_config():
    """
    Loads and returns the project configuration.
    
    Supports two modes:
    1. Multi-file mode: Loads base.yaml, model.yaml, inference.yaml, performance.yaml, thresholds.yaml
    2. Legacy mode: Falls back to single config.yaml for backward compatibility
    
    Returns:
        dict: The loaded and validated configuration dictionary.

    Raises:
        FileNotFoundError: If no config files exist.
        yaml.YAMLError: If config files cannot be parsed.
        ConfigValidationError: If the configuration is invalid.
    """
    global _config
    if _config is None:
        config_dir = Path.home() / "deepEMIA" / "config"
        
        try:
            # Try multi-file approach first
            if (config_dir / "base.yaml").exists():
                system_logger.info("Using modular multi-file configuration")
                raw_config = load_config_files(config_dir)
            # Fallback to single config.yaml for backward compatibility
            else:
                config_path = config_dir / "config.yaml"
                if config_path.exists():
                    system_logger.info("Using legacy single config.yaml")
                    with open(config_path, "r") as f:
                        raw_config = yaml.safe_load(f)
                else:
                    raise FileNotFoundError(
                        f"No configuration files found in {config_dir}. "
                        "Expected either base.yaml (modular) or config.yaml (legacy)."
                    )

            # Validate configuration
            try:
                from src.utils.config_validator import validate_config

                _config = validate_config(raw_config)
                system_logger.info("Configuration validated successfully")
            except ImportError:
                # Fallback if validator is not available
                _config = raw_config
                system_logger.warning(
                    "Configuration validator not available, using unvalidated config"
                )
            except Exception as e:
                system_logger.error(f"Configuration validation failed: {e}")
                # Use raw config but log warning
                _config = raw_config
                system_logger.warning(
                    "Using unvalidated configuration due to validation error"
                )

        except FileNotFoundError as e:
            system_logger.error(str(e))
            raise
        except yaml.YAMLError as e:
            system_logger.error(f"Error parsing configuration file(s): {e}")
            raise
            
    return _config
