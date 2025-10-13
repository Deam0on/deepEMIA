# Testing Guide

Guidelines for testing deepEMIA.

## Test Structure

```text
tests/
├── test_data/
│   ├── test_datasets.py
│   └── test_models.py
├── test_functions/
│   ├── test_train.py
│   ├── test_evaluate.py
│   └── test_inference.py
└── test_utils/
    ├── test_config.py
    ├── test_measurements.py
    └── test_mask_utils.py
```

## Writing Tests

### Unit Tests

Test individual functions:

```python
import pytest
from src.utils.measurements import midpoint

def test_midpoint():
    result = midpoint((0, 0), (10, 10))
    assert result == (5.0, 5.0)
```

### Integration Tests

Test component interactions:

```python
def test_training_pipeline():
    # Setup
    dataset_name = "test_dataset"
    
    # Execute
    train_on_dataset(dataset_name, rcnn=50)
    
    # Verify
    assert model_file_exists(dataset_name)
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_utils/test_measurements.py

# Run with coverage
pytest --cov=src tests/
```

## See Also

- [Contributing Guide](contributing.md)
- [Code Style Guide](code-style.md)
