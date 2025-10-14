"""
Custom exception classes for the deepEMIA project.

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


class TrainingError(PipelineError):
    """Raised when training operations fail."""

    pass
