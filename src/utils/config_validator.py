"""
Configuration schema validation utilities.

This module provides validation for the project configuration to ensure
all required fields are present and have valid values.
"""

from typing import Dict, Any, List
from pathlib import Path
import os

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass

# Configuration schema definition
CONFIG_SCHEMA = {
    'bucket': {
        'type': str,
        'required': True,
        'description': 'Google Cloud Storage bucket name'
    },
    'paths': {
        'type': dict,
        'required': True,
        'fields': {
            'main_script': {'type': str, 'required': True},
            'split_dir': {'type': str, 'required': True},
            'category_json': {'type': str, 'required': True},
            'eta_file': {'type': str, 'required': True},
            'logs_dir': {'type': str, 'required': True},
            'output_dir': {'type': str, 'required': True},
            'local_dataset_root': {'type': str, 'required': True}
        }
    },
    'scale_bar_rois': {
        'type': dict,
        'required': True,
        'fields': {
            'default': {
                'type': dict,
                'required': True,
                'fields': {
                    'x_start_factor': {'type': (int, float), 'required': True},
                    'y_start_factor': {'type': (int, float), 'required': True},
                    'width_factor': {'type': (int, float), 'required': True},
                    'height_factor': {'type': (int, float), 'required': True}
                }
            }
        }
    },
    'scalebar_thresholds': {
        'type': dict,
        'required': True,
        'fields': {
            'intensity': {'type': int, 'required': True},
            'proximity': {'type': int, 'required': True}
        }
    },
    'measure_contrast_distribution': {
        'type': bool,
        'required': False,
        'default': False
    },
    'rcnn_hyperparameters': {
        'type': dict,
        'required': False,
        'fields': {
            'default': {'type': dict, 'required': False},
            'best': {'type': dict, 'required': False}
        }
    }
}

def validate_field(value: Any, field_schema: Dict[str, Any], field_name: str) -> Any:
    """Validate a single field against its schema."""
    field_type = field_schema.get('type')
    required = field_schema.get('required', False)
    default = field_schema.get('default')
    
    # Handle missing required fields
    if value is None:
        if required:
            raise ConfigValidationError(f"Required field '{field_name}' is missing")
        return default
    
    # Type validation
    if field_type and not isinstance(value, field_type):
        raise ConfigValidationError(f"Field '{field_name}' must be of type {field_type.__name__ if hasattr(field_type, '__name__') else field_type}, got {type(value).__name__}")
    
    # Nested object validation
    if field_type == dict and 'fields' in field_schema:
        return validate_config_dict(value, field_schema['fields'], field_name)
    
    # Path validation for path fields
    if field_name.endswith('_dir') or field_name.endswith('_file') or 'path' in field_name.lower():
        expanded_path = Path(value).expanduser()
        # Create parent directories if they don't exist
        if field_name.endswith('_dir') or field_name.endswith('_file'):
            try:
                expanded_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory for {field_name}: {e}")
    
    return value

def validate_config_dict(config_dict: Dict[str, Any], schema: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """Validate a configuration dictionary against a schema."""
    validated_config = {}
    
    for field_name, field_schema in schema.items():
        full_field_name = f"{prefix}.{field_name}" if prefix else field_name
        field_value = config_dict.get(field_name)
        
        validated_config[field_name] = validate_field(field_value, field_schema, full_field_name)
    
    # Check for unexpected fields
    unexpected_fields = set(config_dict.keys()) - set(schema.keys())
    if unexpected_fields:
        print(f"Warning: Unexpected configuration fields found: {unexpected_fields}")
        # Include unexpected fields in validated config
        for field in unexpected_fields:
            validated_config[field] = config_dict[field]
    
    return validated_config

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the complete configuration against the schema.
    
    Args:
        config: The configuration dictionary to validate
        
    Returns:
        dict: The validated and normalized configuration
        
    Raises:
        ConfigValidationError: If validation fails
    """
    try:
        return validate_config_dict(config, CONFIG_SCHEMA)
    except ConfigValidationError:
        raise
    except Exception as e:
        raise ConfigValidationError(f"Configuration validation failed: {str(e)}")

def get_schema_description() -> str:
    """Get a human-readable description of the configuration schema."""
    description = "Configuration Schema:\n"
    
    def describe_fields(schema: Dict, indent: int = 0) -> str:
        result = ""
        for field_name, field_schema in schema.items():
            prefix = "  " * indent
            field_type = field_schema.get('type')
            required = field_schema.get('required', False)
            default = field_schema.get('default')
            desc = field_schema.get('description', '')
            
            type_name = field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)
            req_str = " (required)" if required else f" (optional, default: {default})" if default is not None else " (optional)"
            
            result += f"{prefix}- {field_name}: {type_name}{req_str}"
            if desc:
                result += f" - {desc}"
            result += "\n"
            
            if field_type == dict and 'fields' in field_schema:
                result += describe_fields(field_schema['fields'], indent + 1)
        
        return result
    
    return description + describe_fields(CONFIG_SCHEMA)
