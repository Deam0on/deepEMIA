"""
GPU availability checking utilities.

This module provides functions for:
- Checking GPU/CUDA availability
- Prompting user for confirmation when GPU is unavailable
- Logging GPU information

The module integrates with PyTorch to detect CUDA-enabled GPUs and provides
interactive prompts for users to decide whether to continue without GPU.
"""

import sys
import torch
from src.utils.logger_utils import system_logger


def check_gpu_availability(require_gpu: bool = False, interactive: bool = True) -> bool:
    """
    Check if GPU/CUDA is available and prompt user if not.
    
    Parameters:
    - require_gpu (bool): If True, will exit if no GPU is available and user doesn't confirm
    - interactive (bool): If True, prompts user for confirmation. If False, just logs warning.
    
    Returns:
    - bool: True if GPU is available or user confirmed to continue, False otherwise
    """
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        # GPU is available - log details
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        
        system_logger.info(f"✓ GPU available: {device_name}")
        system_logger.info(f"  CUDA version: {cuda_version}")
        system_logger.info(f"  Device count: {device_count}")
        
        # Log memory info for each GPU
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # Convert to GB
            system_logger.info(f"  GPU {i}: {props.name}, {total_memory:.1f}GB VRAM")
        
        return True
    
    else:
        # No GPU available
        system_logger.warning("⚠ No GPU/CUDA device detected!")
        system_logger.warning("  This may significantly impact performance for training and inference.")
        system_logger.warning("  Possible causes:")
        system_logger.warning("    - GPU drivers not installed")
        system_logger.warning("    - CUDA not installed or incompatible version")
        system_logger.warning("    - GPU initialization failed")
        system_logger.warning("    - Running on CPU-only instance")
        
        if not interactive:
            # Non-interactive mode - just log warning and continue
            system_logger.warning("  Continuing with CPU-only mode...")
            return True
        
        # Interactive mode - prompt user
        print("\n" + "="*60)
        print("WARNING: No GPU/CUDA device detected!")
        print("="*60)
        print("\nThis will significantly impact performance:")
        print("  • Training may be 10-50x slower")
        print("  • Inference may be 5-20x slower")
        print("\nPossible causes:")
        print("  • GPU drivers not installed")
        print("  • CUDA not installed or incompatible version")
        print("  • GPU initialization failed")
        print("  • Running on CPU-only instance")
        print("\nRecommendation: Check GPU status with 'nvidia-smi'")
        print("="*60)
        
        while True:
            response = input("\nDo you want to continue anyway? (yes/no): ").strip().lower()
            
            if response in ['yes', 'y']:
                system_logger.info("User confirmed to continue without GPU")
                print("Continuing with CPU-only mode...\n")
                return True
            elif response in ['no', 'n']:
                system_logger.info("User chose to abort due to missing GPU")
                print("Aborting. Please check GPU availability and try again.\n")
                return False
            else:
                print("Please enter 'yes' or 'no'")


def get_optimal_device(prefer_gpu=True):
    """
    Get the optimal torch device based on availability.
    
    Parameters:
    - prefer_gpu (bool): Prefer GPU if available
    
    Returns:
    - torch.device: The optimal device
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def log_device_info():
    """
    Log comprehensive device information.
    Useful for debugging and performance optimization.
    """
    if torch.cuda.is_available():
        system_logger.info("=== GPU Device Information ===")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            system_logger.info(f"GPU {i}:")
            system_logger.info(f"  Name: {props.name}")
            system_logger.info(f"  Total Memory: {props.total_memory / (1024**3):.2f}GB")
            system_logger.info(f"  Compute Capability: {props.major}.{props.minor}")
            system_logger.info(f"  Multi-Processors: {props.multi_processor_count}")
    else:
        system_logger.info("=== CPU-Only Mode ===")
        system_logger.info("No GPU devices available")
    
    system_logger.info(f"PyTorch version: {torch.__version__}")
    system_logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        system_logger.info(f"CUDA version: {torch.version.cuda}")
        system_logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
