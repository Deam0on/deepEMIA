"""
Safe file operations utilities.

This module provides utilities for safe file operations with path validation
to prevent path traversal and other security issues.
"""

import os
import glob
import shutil
from pathlib import Path
from typing import List, Union
from src.utils.logger_utils import system_logger

class SafeFileOperationError(Exception):
    """Raised when a file operation is deemed unsafe."""
    pass

def validate_path_safety(file_path: Union[str, Path], allowed_base_dirs: List[Union[str, Path]]) -> Path:
    """
    Validate that a file path is safe and within allowed directories.
    
    Args:
        file_path: The file path to validate
        allowed_base_dirs: List of allowed base directories
        
    Returns:
        Path: The validated absolute path
        
    Raises:
        SafeFileOperationError: If the path is not safe
    """
    file_path = Path(file_path).resolve()
    
    # Convert allowed base dirs to absolute paths
    allowed_abs_dirs = [Path(d).resolve() for d in allowed_base_dirs]
    
    # Check if the file path is within any allowed directory
    for allowed_dir in allowed_abs_dirs:
        try:
            file_path.relative_to(allowed_dir)
            return file_path
        except ValueError:
            continue
    
    raise SafeFileOperationError(
        f"Path {file_path} is not within any allowed directory: {allowed_abs_dirs}"
    )

def safe_remove_files(pattern: str, allowed_dirs: List[Union[str, Path]]) -> List[str]:
    """
    Safely remove files matching a pattern within allowed directories.
    
    Args:
        pattern: Glob pattern for files to remove
        allowed_dirs: List of allowed base directories
        
    Returns:
        List of removed file paths
        
    Raises:
        SafeFileOperationError: If any file is outside allowed directories
    """
    removed_files = []
    
    for file_path in glob.glob(pattern):
        try:
            validated_path = validate_path_safety(file_path, allowed_dirs)
            
            if validated_path.exists():
                if validated_path.is_file():
                    validated_path.unlink()
                    removed_files.append(str(validated_path))
                    system_logger.info(f"Safely removed file: {validated_path}")
                elif validated_path.is_dir():
                    shutil.rmtree(validated_path)
                    removed_files.append(str(validated_path))
                    system_logger.info(f"Safely removed directory: {validated_path}")
        except SafeFileOperationError as e:
            system_logger.error(f"Unsafe file operation blocked: {e}")
            raise
        except Exception as e:
            system_logger.error(f"Error removing {file_path}: {e}")
            
    return removed_files

def safe_copy_file(src: Union[str, Path], dst: Union[str, Path], allowed_dirs: List[Union[str, Path]]) -> None:
    """
    Safely copy a file within allowed directories.
    
    Args:
        src: Source file path
        dst: Destination file path
        allowed_dirs: List of allowed base directories
        
    Raises:
        SafeFileOperationError: If paths are outside allowed directories
    """
    src_path = validate_path_safety(src, allowed_dirs)
    dst_path = validate_path_safety(dst, allowed_dirs)
    
    # Ensure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(src_path, dst_path)
    system_logger.info(f"Safely copied {src_path} to {dst_path}")

def safe_move_file(src: Union[str, Path], dst: Union[str, Path], allowed_dirs: List[Union[str, Path]]) -> None:
    """
    Safely move a file within allowed directories.
    
    Args:
        src: Source file path
        dst: Destination file path
        allowed_dirs: List of allowed base directories
        
    Raises:
        SafeFileOperationError: If paths are outside allowed directories
    """
    src_path = validate_path_safety(src, allowed_dirs)
    dst_path = validate_path_safety(dst, allowed_dirs)
    
    # Ensure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.move(src_path, dst_path)
    system_logger.info(f"Safely moved {src_path} to {dst_path}")

def get_safe_temp_dir(base_dir: Union[str, Path] = None) -> Path:
    """
    Get a safe temporary directory within the project structure.
    
    Args:
        base_dir: Base directory for temp files (defaults to ~/deepEMIA/temp)
        
    Returns:
        Path to the temporary directory
    """
    if base_dir is None:
        base_dir = Path.home() / "deepEMIA" / "temp"
    else:
        base_dir = Path(base_dir)
    
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def cleanup_temp_files(temp_dir: Union[str, Path], max_age_hours: int = 24) -> None:
    """
    Clean up old temporary files.
    
    Args:
        temp_dir: Directory containing temporary files
        max_age_hours: Maximum age of files to keep in hours
    """
    import time
    
    temp_dir = Path(temp_dir)
    if not temp_dir.exists():
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    removed_count = 0
    for file_path in temp_dir.rglob("*"):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    system_logger.error(f"Error removing old temp file {file_path}: {e}")
    
    if removed_count > 0:
        system_logger.info(f"Cleaned up {removed_count} old temporary files from {temp_dir}")
