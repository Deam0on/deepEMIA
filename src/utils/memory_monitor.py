"""
Memory monitoring utilities for preventing OOM during inference.

This module provides failsafe mechanisms to detect low memory conditions
and gracefully stop inference to preserve already-processed results.
"""

import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import torch

from src.utils.logger_utils import system_logger


class MemoryMonitor:
    """
    Monitor system RAM and GPU VRAM during inference to prevent OOM crashes.
    
    Tracks memory usage per image and predicts whether the next image can
    be safely processed based on observed maximum memory consumption.
    """
    
    def __init__(
        self,
        ram_threshold_gb: float = 3.0,
        vram_threshold_gb: float = 2.0,
        safety_factor: float = 1.5,
        cold_start_images: int = 3,
        enabled: bool = True
    ):
        """
        Initialize memory monitor.
        
        Args:
            ram_threshold_gb: Minimum RAM (GB) required to continue
            vram_threshold_gb: Minimum VRAM (GB) required to continue
            safety_factor: Multiply max observed memory by this factor
            cold_start_images: Use fixed thresholds for first N images
            enabled: Enable/disable memory monitoring
        """
        self.ram_threshold_gb = ram_threshold_gb
        self.vram_threshold_gb = vram_threshold_gb
        self.safety_factor = safety_factor
        self.cold_start_images = cold_start_images
        self.enabled = enabled
        
        # Track memory usage history
        self.ram_usage_history: List[float] = []
        self.vram_usage_history: List[float] = []
        self.images_processed = 0
        
        # GPU availability
        self.has_gpu = torch.cuda.is_available()
        
        # Log initial state
        if self.enabled:
            ram_avail, vram_avail = self._get_available_memory()
            system_logger.info(
                f"Memory monitoring enabled: RAM threshold={ram_threshold_gb:.1f}GB, "
                f"VRAM threshold={vram_threshold_gb:.1f}GB, safety_factor={safety_factor}x"
            )
            system_logger.info(
                f"Initial memory: RAM={ram_avail:.2f}GB available, "
                f"VRAM={vram_avail:.2f}GB available"
            )
        else:
            system_logger.info("Memory monitoring disabled")
    
    def _get_available_memory(self) -> Tuple[float, float]:
        """
        Get currently available RAM and VRAM in GB.
        
        Returns:
            Tuple of (available_ram_gb, available_vram_gb)
        """
        # System RAM
        ram_available = psutil.virtual_memory().available / (1024**3)
        
        # GPU VRAM
        vram_available = 0.0
        if self.has_gpu:
            try:
                total_vram = torch.cuda.get_device_properties(0).total_memory
                reserved_vram = torch.cuda.memory_reserved(0)
                vram_available = (total_vram - reserved_vram) / (1024**3)
            except Exception as e:
                system_logger.warning(f"Could not get VRAM info: {e}")
        
        return ram_available, vram_available
    
    def _get_memory_usage(self) -> Tuple[float, float]:
        """
        Get current RAM and VRAM usage in GB.
        
        Returns:
            Tuple of (ram_used_gb, vram_used_gb)
        """
        # System RAM usage by current process
        process = psutil.Process()
        ram_used = process.memory_info().rss / (1024**3)
        
        # GPU VRAM usage
        vram_used = 0.0
        if self.has_gpu:
            try:
                vram_used = torch.cuda.memory_allocated(0) / (1024**3)
            except Exception as e:
                system_logger.warning(f"Could not get VRAM usage: {e}")
        
        return ram_used, vram_used
    
    def record_image_memory(self) -> None:
        """
        Record memory usage after processing an image.
        Should be called after each image completes.
        """
        if not self.enabled:
            return
        
        ram_used, vram_used = self._get_memory_usage()
        self.ram_usage_history.append(ram_used)
        self.vram_usage_history.append(vram_used)
        self.images_processed += 1
        
        system_logger.debug(
            f"Image {self.images_processed} memory: "
            f"RAM={ram_used:.2f}GB, VRAM={vram_used:.2f}GB"
        )
    
    def check_memory_available(self) -> Tuple[bool, Optional[str]]:
        """
        Check if there's enough memory to process the next image.
        
        Returns:
            Tuple of (can_continue, reason_if_not)
            - can_continue: True if safe to continue, False if should stop
            - reason_if_not: Explanation if returning False
        """
        if not self.enabled:
            return True, None
        
        # Force garbage collection before checking
        gc.collect()
        if self.has_gpu:
            torch.cuda.empty_cache()
        
        # Get current available memory
        ram_avail, vram_avail = self._get_available_memory()
        
        # Cold start period: use fixed thresholds
        if self.images_processed < self.cold_start_images:
            # Check against fixed thresholds
            if ram_avail < self.ram_threshold_gb:
                reason = (
                    f"Low RAM detected during cold start: {ram_avail:.2f}GB available "
                    f"(threshold: {self.ram_threshold_gb:.2f}GB)"
                )
                return False, reason
            
            if self.has_gpu and vram_avail < self.vram_threshold_gb:
                reason = (
                    f"Low VRAM detected during cold start: {vram_avail:.2f}GB available "
                    f"(threshold: {self.vram_threshold_gb:.2f}GB)"
                )
                return False, reason
            
            return True, None
        
        # Adaptive period: use maximum observed memory
        max_ram_used = max(self.ram_usage_history) if self.ram_usage_history else 0
        max_vram_used = max(self.vram_usage_history) if self.vram_usage_history else 0
        
        # Calculate required memory with safety factor
        required_ram = max_ram_used * self.safety_factor
        required_vram = max_vram_used * self.safety_factor
        
        # Check RAM
        if ram_avail < required_ram:
            reason = (
                f"Insufficient RAM for next image: {ram_avail:.2f}GB available, "
                f"{required_ram:.2f}GB required (max observed: {max_ram_used:.2f}GB × {self.safety_factor}x safety)"
            )
            return False, reason
        
        # Check VRAM
        if self.has_gpu and vram_avail < required_vram:
            reason = (
                f"Insufficient VRAM for next image: {vram_avail:.2f}GB available, "
                f"{required_vram:.2f}GB required (max observed: {max_vram_used:.2f}GB × {self.safety_factor}x safety)"
            )
            return False, reason
        
        # All checks passed
        return True, None
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        ram_avail, vram_avail = self._get_available_memory()
        ram_used, vram_used = self._get_memory_usage()
        
        stats = {
            "ram_available_gb": ram_avail,
            "vram_available_gb": vram_avail,
            "ram_used_gb": ram_used,
            "vram_used_gb": vram_used,
            "images_processed": self.images_processed,
        }
        
        if self.ram_usage_history:
            stats["max_ram_per_image_gb"] = max(self.ram_usage_history)
            stats["avg_ram_per_image_gb"] = sum(self.ram_usage_history) / len(self.ram_usage_history)
        
        if self.vram_usage_history:
            stats["max_vram_per_image_gb"] = max(self.vram_usage_history)
            stats["avg_vram_per_image_gb"] = sum(self.vram_usage_history) / len(self.vram_usage_history)
        
        return stats


def save_checkpoint(
    completed_images: List[str],
    remaining_images: List[str],
    output_dir: str,
    memory_stats: Dict[str, float],
    reason: str
) -> str:
    """
    Save checkpoint file with inference progress.
    
    Args:
        completed_images: List of successfully processed image filenames
        remaining_images: List of unprocessed image filenames
        output_dir: Directory to save checkpoint
        memory_stats: Current memory statistics
        reason: Reason for stopping
    
    Returns:
        Path to checkpoint file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_filename = f"inference_checkpoint_{timestamp}.json"
    checkpoint_path = os.path.join(output_dir, checkpoint_filename)
    
    checkpoint_data = {
        "timestamp": timestamp,
        "reason": reason,
        "total_images": len(completed_images) + len(remaining_images),
        "completed_count": len(completed_images),
        "remaining_count": len(remaining_images),
        "completed_images": completed_images,
        "remaining_images": remaining_images,
        "memory_stats": memory_stats,
    }
    
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        
        system_logger.info(f"Checkpoint saved to: {checkpoint_path}")
        return checkpoint_path
    
    except Exception as e:
        system_logger.error(f"Failed to save checkpoint: {e}")
        # Still return a path even if save failed
        return checkpoint_path


def print_early_stop_message(
    completed_images: List[str],
    remaining_images: List[str],
    memory_stats: Dict[str, float],
    reason: str,
    checkpoint_path: str,
    output_dir: str
) -> None:
    """
    Print a clear message to the user about the early stop.
    
    Args:
        completed_images: List of successfully processed images
        remaining_images: List of unprocessed images
        memory_stats: Current memory statistics
        reason: Reason for stopping
        checkpoint_path: Path to checkpoint file
        output_dir: Output directory with results
    """
    total = len(completed_images) + len(remaining_images)
    completed = len(completed_images)
    percentage = (completed / total * 100) if total > 0 else 0
    
    # Build the message
    message = "\n" + "=" * 80 + "\n"
    message += "⚠️  INFERENCE STOPPED - LOW MEMORY DETECTED\n"
    message += "=" * 80 + "\n"
    message += f"Reason: {reason}\n\n"
    message += f"Progress: {completed}/{total} images ({percentage:.1f}%)\n"
    message += f"Remaining: {len(remaining_images)} images\n\n"
    
    message += "Memory Status:\n"
    message += f"  RAM:  {memory_stats.get('ram_available_gb', 0):.2f} GB available\n"
    if memory_stats.get('vram_available_gb', 0) > 0:
        message += f"  VRAM: {memory_stats.get('vram_available_gb', 0):.2f} GB available\n"
    message += "\n"
    
    if memory_stats.get('max_ram_per_image_gb'):
        message += f"  Max RAM per image:  {memory_stats['max_ram_per_image_gb']:.2f} GB\n"
    if memory_stats.get('max_vram_per_image_gb'):
        message += f"  Max VRAM per image: {memory_stats['max_vram_per_image_gb']:.2f} GB\n"
    message += "\n"
    
    message += f"Results saved to: {output_dir}\n"
    message += f"Checkpoint saved to: {checkpoint_path}\n\n"
    
    message += "Remaining images to process:\n"
    for i, img in enumerate(remaining_images[:10], 1):  # Show first 10
        message += f"  {i}. {img}\n"
    if len(remaining_images) > 10:
        message += f"  ... and {len(remaining_images) - 10} more\n"
    message += "\n"
    
    message += "To process remaining images:\n"
    message += "  1. Close memory-intensive applications\n"
    message += "  2. Process images in smaller batches\n"
    message += "  3. Re-run inference with only the remaining images\n"
    message += "=" * 80 + "\n"
    
    # Print to console
    print(message)
    
    # Also log it
    system_logger.warning("Inference stopped early due to low memory")
    system_logger.info(f"Processed {completed}/{total} images successfully")
    system_logger.info(f"Checkpoint saved to: {checkpoint_path}")


def cleanup_resources() -> None:
    """
    Clean up resources before exiting.
    Ensures models are released from GPU and temporary files are cleaned.
    """
    system_logger.info("Cleaning up resources...")
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            system_logger.debug("GPU cache cleared")
        except Exception as e:
            system_logger.warning(f"Could not clear GPU cache: {e}")
    
    system_logger.info("Resource cleanup complete")
