"""
Custom exception classes for the UW Computer Vision project.

This module defines project-specific exceptions to improve error handling
and debugging throughout the application.
"""

class PipelineError(Exception):
    """Base exception for pipeline operations."""
    
    def __init__(self, message: str, stage: str = None, details: dict = None):
        super().__init__(message)
        self.stage = stage
        self.details = details or {}
    
    def __str__(self):
        base_msg = super().__str__()
        if self.stage:
            base_msg = f"[{self.stage}] {base_msg}"
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base_msg = f"{base_msg} ({details_str})"
        return base_msg

class ConfigurationError(PipelineError):
    """Raised when configuration is invalid or missing."""
    pass

class ModelLoadError(PipelineError):
    """Raised when model loading fails."""
    pass

class DatasetError(PipelineError):
    """Raised when dataset operations fail."""
    pass

class InferenceError(PipelineError):
    """Raised when inference operations fail."""
    pass

class TrainingError(PipelineError):
    """Raised when training operations fail."""
    pass

class FileOperationError(PipelineError):
    """Raised when file operations fail."""
    pass

class NetworkError(PipelineError):
    """Raised when network operations fail."""
    pass

class ValidationError(PipelineError):
    """Raised when validation fails."""
    pass

class ResourceError(PipelineError):
    """Raised when resource allocation or management fails."""
    pass

class ProcessingError(PipelineError):
    """Raised when image or data processing fails."""
    pass

# Error handling utilities
def handle_exception(func):
    """Decorator to handle exceptions consistently across the pipeline."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PipelineError:
            # Re-raise pipeline errors as-is
            raise
        except Exception as e:
            # Wrap other exceptions in a generic pipeline error
            stage = getattr(func, '__module__', 'unknown')
            raise PipelineError(f"Unexpected error in {func.__name__}: {str(e)}", stage=stage) from e
    return wrapper

def create_error_context(stage: str, operation: str = None, **kwargs):
    """Create standardized error context for exceptions."""
    context = {'stage': stage}
    if operation:
        context['operation'] = operation
    context.update(kwargs)
    return context
