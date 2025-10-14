# Contributing to deepEMIA

Guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch
4. Make your changes
5. Run tests
6. Submit a pull request

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/deepEMIA.git
cd deepEMIA
pip install -r requirements.txt
```

## Code Style

### Python Style

Follow PEP 8 guidelines:
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use snake_case for functions and variables
- Use PascalCase for classes

### Docstrings

All functions must have docstrings:

```python
def function_name(param1: str, param2: int = 10) -> dict:
    """
    Brief description of function.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ExceptionType: When this exception is raised
    """
    pass
```

## Testing

Run tests before submitting:

```bash
python -m pytest tests/
```

## Pull Request Process

1. Update documentation
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit PR with clear description

## Code Review

All submissions require review. We look for:
- Code quality and style
- Test coverage
- Documentation updates
- Performance considerations

## See Also

- [Code Style Guide](code-style.md)
- [Testing Guide](testing.md)
