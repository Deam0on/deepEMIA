# Code Style Guide

Detailed code style guidelines for deepEMIA.

## Python Guidelines

### Import Organization

```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
import torch
from detectron2.config import get_cfg

# Local
from src.utils.config import get_config
from src.utils.logger_utils import system_logger
```

### Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore`

### Type Hints

Use type hints for all function signatures:

```python
from typing import List, Dict, Optional, Union
from pathlib import Path

def process_images(
    image_paths: List[Path],
    threshold: float = 0.5,
    output_dir: Optional[Path] = None
) -> Dict[str, any]:
    pass
```

## Documentation Style

### Module Docstrings

```python
"""
Module description.

This module provides...
"""
```

### Function Docstrings

Use Google style:

```python
def function(arg1: str, arg2: int) -> bool:
    """
    Brief description.

    Detailed description.

    Args:
        arg1: Description
        arg2: Description

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when raised
    """
    pass
```

## Best Practices

- Keep functions focused and small
- Use descriptive variable names
- Comment complex logic
- Avoid global variables
- Handle errors explicitly
- Log important operations

## See Also

- [Contributing Guide](contributing.md)
- [Testing Guide](testing.md)
