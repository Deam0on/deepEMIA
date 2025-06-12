"""
Configuration Loader Utility

Provides a singleton-style loader for the main YAML configuration file.
Ensures the config is loaded only once and reused throughout the project.
"""

import yaml
from pathlib import Path
import logging

_config = None

def get_config():
    """
    Loads and returns the project configuration from config.yaml.

    Returns:
        dict: The loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file cannot be parsed.
    """
    global _config
    if _config is None:
        config_path = Path.home() / "uw-com-vision" / "config" / "config.yaml"
        try:
            with open(config_path, "r") as f:
                _config = yaml.safe_load(f)
            logging.info(f"Loaded configuration from {config_path}")
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
            raise
    return _config