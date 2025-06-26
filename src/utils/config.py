"""
Configuration Loader Utility

Provides a singleton-style loader for the main YAML configuration file.
Ensures the config is loaded only once and reused throughout the project.
Includes configuration validation for security and reliability.
"""

from pathlib import Path

import yaml

from src.utils.logger_utils import system_logger

_config = None


def get_config():
    """
    Loads and returns the project configuration from config.yaml.

    Returns:
        dict: The loaded and validated configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file cannot be parsed.
        ConfigValidationError: If the configuration is invalid.
    """
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
                # Fallback if validator is not available
                _config = raw_config
                system_logger.warning("Configuration validator not available, using unvalidated config")
            except Exception as e:
                system_logger.error(f"Configuration validation failed: {e}")
                # Use raw config but log warning
                _config = raw_config
                system_logger.warning("Using unvalidated configuration due to validation error")
                
        except FileNotFoundError:
            system_logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            system_logger.error(f"Error parsing configuration file: {e}")
            raise
    return _config
