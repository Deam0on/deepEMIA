"""
Logging utilities for the deepEMIA project.

This module provides centralized logging configuration with file and console
output for both development and production environments.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def get_log_dir():
    """
    Get the appropriate log directory based on the environment.

    Returns:
        Path: Path to the log directory.
    """
    if getattr(sys, "frozen", False):
        # Running from PyInstaller, if ever used
        base_dir = Path.home() / "AppData" / "Local" / "AD_APP" / "logs"
    else:
        # Local dev
        base_dir = Path.home() / "logs"
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def set_console_log_level(level=logging.INFO):
    """
    Set the console log level for the system logger.
    
    Args:
        level: logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    """
    for handler in system_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(level)


LOG_DIR = get_log_dir()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- SYSTEM LOGGER ---
system_log_path = LOG_DIR / f"system_{timestamp}.log"
system_logger = logging.getLogger("system")
system_logger.setLevel(logging.DEBUG)  # File always gets DEBUG
system_handler = logging.FileHandler(system_log_path, encoding="utf-8")
system_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
system_logger.addHandler(system_handler)

if not getattr(sys, "frozen", False):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    console_handler.setLevel(logging.INFO)  # Console defaults to INFO
    system_logger.addHandler(console_handler)


def log_memory_usage(stage: str = "") -> None:
    """
    Log current memory usage for debugging OOM issues.
    
    Args:
        stage: Description of the current processing stage
    """
    try:
        import psutil
        import gc
        import torch
        
        # Python memory
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024**2
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            system_logger.info(
                f"[{stage}] Memory - RAM: {mem_mb:.1f}MB, "
                f"GPU: {gpu_allocated:.2f}GB allocated / {gpu_reserved:.2f}GB reserved"
            )
        else:
            system_logger.info(f"[{stage}] Memory - RAM: {mem_mb:.1f}MB")
        
    except Exception as e:
        system_logger.debug(f"[{stage}] Could not log memory usage: {e}")

