"""
Gradio helper functions for the deepEMIA project.

This module provides utilities specifically for Gradio interface components,
including state management, authentication, and UI helpers.
"""
import os
import hashlib
import threading
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import get_config
from src.utils.logger_utils import system_logger

config = get_config()


class SessionState:
    """Simple session state management for Gradio."""
    
    def __init__(self):
        self._state = {}
        self._lock = threading.Lock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from session state."""
        with self._lock:
            return self._state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in session state."""
        with self._lock:
            self._state[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple values in session state."""
        with self._lock:
            self._state.update(updates)
    
    def clear(self) -> None:
        """Clear all session state."""
        with self._lock:
            self._state.clear()


# Global session state instance
session_state = SessionState()


def verify_admin_password(input_password: str) -> bool:
    """
    Verify admin password using environment variable hash.
    
    Args:
        input_password: Password to verify
        
    Returns:
        bool: True if password is correct, False otherwise
    """
    stored_hash = os.environ.get('ADMIN_PASSWORD_HASH')
    if not stored_hash:
        system_logger.warning("ADMIN_PASSWORD_HASH environment variable not set")
        return False
    
    input_hash = hashlib.sha256(input_password.encode()).hexdigest()
    return stored_hash == input_hash


def format_time_remaining(seconds: int) -> str:
    """
    Format seconds into human-readable time format.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        str: Formatted time string (e.g., "2h 30m 15s")
    """
    if seconds <= 0:
        return "0s"
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def create_status_message(
    status: str, 
    message: str, 
    progress: Optional[float] = None,
    eta: Optional[int] = None
) -> str:
    """
    Create a formatted status message for display.
    
    Args:
        status: Status type ("success", "error", "info", "warning")
        message: Main message text
        progress: Optional progress percentage (0-100)
        eta: Optional ETA in seconds
        
    Returns:
        str: Formatted HTML status message
    """
    status_colors = {
        "success": "#28a745",
        "error": "#dc3545", 
        "info": "#17a2b8",
        "warning": "#ffc107"
    }
    
    color = status_colors.get(status, "#6c757d")
    
    html = f'<div style="padding: 10px; margin: 10px 0; border-left: 4px solid {color}; background-color: #f8f9fa;">'
    html += f'<strong style="color: {color};">{status.upper()}:</strong> {message}'
    
    if progress is not None:
        html += f'<br><div style="background-color: #e9ecef; border-radius: 4px; margin-top: 8px;">'
        html += f'<div style="width: {progress}%; height: 20px; background-color: {color}; border-radius: 4px; transition: width 0.3s;"></div>'
        html += f'</div><small>{progress:.1f}% complete</small>'
    
    if eta is not None:
        eta_str = format_time_remaining(eta)
        html += f'<br><small>⏱️ ETA: {eta_str}</small>'
    
    html += '</div>'
    return html


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing unsafe characters.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Safe filename
    """
    import re
    # Remove unsafe characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing dots and spaces
    safe = safe.strip('. ')
    # Limit length
    if len(safe) > 255:
        name, ext = os.path.splitext(safe)
        safe = name[:255-len(ext)] + ext
    
    return safe or "file"


def validate_dataset_name(name: str) -> Tuple[bool, str]:
    """
    Validate a dataset name.
    
    Args:
        name: Dataset name to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not name:
        return False, "Dataset name cannot be empty"
    
    if len(name) < 2:
        return False, "Dataset name must be at least 2 characters long"
    
    if len(name) > 50:
        return False, "Dataset name must be less than 50 characters"
    
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        return False, "Dataset name can only contain letters, numbers, hyphens, and underscores"
    
    return True, ""


def parse_class_list(classes_str: str) -> List[str]:
    """
    Parse comma-separated class list and return cleaned list.
    
    Args:
        classes_str: Comma-separated string of class names
        
    Returns:
        List[str]: List of cleaned class names
    """
    if not classes_str:
        return []
    
    classes = [cls.strip() for cls in classes_str.split(",")]
    return [cls for cls in classes if cls]  # Remove empty strings


class ProgressTracker:
    """Track progress for long-running operations."""
    
    def __init__(self):
        self.current_progress = 0.0
        self.status_message = ""
        self.eta_seconds = None
        self.is_running = False
        self._lock = threading.Lock()
    
    def update(self, progress: float, message: str = "", eta: Optional[int] = None):
        """Update progress information."""
        with self._lock:
            self.current_progress = max(0, min(100, progress))
            if message:
                self.status_message = message
            if eta is not None:
                self.eta_seconds = eta
    
    def start(self, message: str = "Starting..."):
        """Start tracking progress."""
        with self._lock:
            self.is_running = True
            self.current_progress = 0.0
            self.status_message = message
            self.eta_seconds = None
    
    def finish(self, message: str = "Complete"):
        """Finish tracking progress."""
        with self._lock:
            self.is_running = False
            self.current_progress = 100.0
            self.status_message = message
            self.eta_seconds = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        with self._lock:
            return {
                "progress": self.current_progress,
                "message": self.status_message,
                "eta": self.eta_seconds,
                "is_running": self.is_running
            }


# Global progress tracker
progress_tracker = ProgressTracker()
