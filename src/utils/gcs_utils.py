"""
Google Cloud Storage utilities for the deepEMIA project.

This module provides functions for:
- Downloading data from GCS buckets
- Uploading data to GCS buckets
- Managing local and cloud storage synchronization

The module handles:
- File transfers between local and cloud storage
- Automatic archiving of results
- Timestamp-based organization of uploaded data

Note:
    Requires gsutil to be installed and configured with appropriate permissions.
"""

import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.config import get_config
from src.utils.logger_utils import system_logger

config = get_config()
bucket = config["bucket"]

# Resolve paths
local_dataset_path = Path(config["paths"]["local_dataset_root"]).expanduser().resolve()


def run_gsutil_with_retry(cmd: list, max_retries: int = 3, retry_delay: float = 2.0) -> subprocess.CompletedProcess:
    """
    Run gsutil command with retry logic for network failures.
    
    Args:
        cmd: gsutil command as list
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds (exponential backoff)
        
    Returns:
        subprocess.CompletedProcess: Result of the command
        
    Raises:
        subprocess.CalledProcessError: If command fails after all retries
    """
    for attempt in range(max_retries):
        try:
            system_logger.info(f"Running command (attempt {attempt + 1}/{max_retries}): {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result
        except subprocess.CalledProcessError as e:
            if attempt == max_retries - 1:
                system_logger.error(f"Command failed after {max_retries} attempts: {e}")
                system_logger.error(f"STDERR: {e.stderr}")
                raise
            else:
                delay = retry_delay * (2 ** attempt)  # Exponential backoff
                system_logger.warning(f"Command failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
    
    # This should never be reached, but just in case
    raise subprocess.CalledProcessError(1, cmd, "Maximum retries exceeded")


def download_data_from_bucket() -> float:
    """
    Download data from a Google Cloud Storage bucket to a local directory.

    This function:
    1. Removes any existing local dataset directory
    2. Downloads the entire dataset from GCS
    3. Tracks and returns the download duration

    Returns:
        float: Time taken to download data in seconds
    """
    download_start_time = datetime.now()
    dirpath = Path.home() / "DATASET"
    if dirpath.exists() and dirpath.is_dir():
        system_logger.info(f"Removing existing dataset directory: {dirpath}")
        shutil.rmtree(dirpath)

    cmd = [
        "gsutil",
        "-m",
        "cp",
        "-r",
        f"gs://{bucket}/DATASET",
        str(local_dataset_path),
    ]
    try:
        result = run_gsutil_with_retry(cmd)
        system_logger.info("gsutil download completed successfully.")
    except subprocess.CalledProcessError as e:
        system_logger.error(f"gsutil download failed after retries: {e}")
        raise

    download_end_time = datetime.now()
    return (download_end_time - download_start_time).total_seconds()


def upload_data_to_bucket() -> float:
    """
    Upload data from local directories to a Google Cloud Storage bucket.

    This function:
    1. Creates a timestamped archive directory in the bucket
    2. Uploads PNG files if present
    3. Uploads CSV files if present
    4. Uploads output directory contents if present
    5. Tracks and returns the upload duration

    Returns:
        float: Time taken to upload data in seconds
    """
    upload_start_time = datetime.now()
    time_offset = timedelta(hours=2)
    timestamp = (datetime.now() + time_offset).strftime("%Y%m%d_%H%M%S")
    archive_path = f"gs://{bucket}/Archive/{timestamp}/"

    # Upload .png files
    png_files = list(Path.home().glob("*.png"))
    if png_files:
        system_logger.info(f"Uploading {len(png_files)} PNG files to {archive_path}")
        cmd = (
            ["gsutil", "-m", "cp", "-r"] + [str(f) for f in png_files] + [archive_path]
        )
        result = run_gsutil_with_retry(cmd)
        if result.returncode != 0:
            system_logger.error(f"PNG upload failed: {result.stderr}")

    # Upload .csv files
    csv_files = list(Path.home().glob("*.csv"))
    if csv_files:
        system_logger.info(f"Uploading {len(csv_files)} CSV files to {archive_path}")
        cmd = (
            ["gsutil", "-m", "cp", "-r"] + [str(f) for f in csv_files] + [archive_path]
        )
        result = run_gsutil_with_retry(cmd)
        if result.returncode != 0:
            system_logger.error(f"CSV upload failed: {result.stderr}")

    # Upload output directory contents
    output_dir = Path.home() / "output"
    if output_dir.exists() and any(output_dir.iterdir()):
        system_logger.info(f"Uploading output directory contents to {archive_path}")
        cmd = ["gsutil", "-m", "cp", "-r", str(output_dir) + "/*", archive_path]
        result = run_gsutil_with_retry(cmd)
        if result.returncode != 0:
            system_logger.error(f"Output directory upload failed: {result.stderr}")

    upload_end_time = datetime.now()
    return (upload_end_time - upload_start_time).total_seconds()


def upload_inference_results(dataset_name: str, model_info: str, output_dir: Path, current_dir: Path = None) -> float:
    """
    Upload inference results from various locations to a timestamped GCS archive.
    Only uploads essential result files, not individual masks or temporary files.
    
    Args:
        dataset_name: Name of the dataset
        model_info: Model information string (e.g., "R101_single", "combo_multi")
        output_dir: Main output directory (usually split_dir)
        current_dir: Current working directory (optional)
        
    Returns:
        float: Time taken to upload data in seconds
    """
    upload_start_time = datetime.now()
    time_offset = timedelta(hours=2)
    timestamp = (datetime.now() + time_offset).strftime("%Y%m%d_%H%M%S")
    archive_path = f"gs://{bucket}/Archive/{timestamp}_{dataset_name}_{model_info}/"
    
    system_logger.info(f"Starting inference results upload to {archive_path}")
    
    # Define essential files to upload (exclude individual masks and temporary files)
    essential_files = []
    
    # 1. Essential files from output directory (split_dir)
    if output_dir.exists():
        system_logger.info(f"Scanning output directory: {output_dir}")
        
        # Main results files only
        essential_patterns = {
            "measurements_results.csv": "Main measurements and analysis results",
            "R50_flip_results.csv": "RLE encoded masks for all classes",
            "class_color_legend.txt": "Class color coding reference",
            "inference_upload_summary.txt": "Upload summary",
            "*_predictions.png": "Combined prediction visualizations (all classes per image)"
        }
        
        for pattern, description in essential_patterns.items():
            for file_path in output_dir.glob(pattern):
                if file_path.is_file():
                    # Skip individual mask files - we only want the combined prediction images
                    if "_mask_" in file_path.name and file_path.name.endswith('.jpg'):
                        continue
                    essential_files.append({
                        "path": file_path,
                        "description": description
                    })
                    system_logger.info(f"Found essential file: {file_path.name} - {description}")
    
    # 2. Essential files from current working directory
    if current_dir is None:
        current_dir = Path.cwd()
    
    if current_dir.exists():
        system_logger.info(f"Scanning current directory: {current_dir}")
        
        # Only look for main result files, not individual masks
        main_result_patterns = [
            "measurements_results.csv",
            "R50_flip_results.csv", 
            "class_color_legend.txt"
        ]
        
        for pattern in main_result_patterns:
            for file_path in current_dir.glob(pattern):
                if file_path.is_file():
                    # Check if not already in list
                    if not any(ef["path"] == file_path for ef in essential_files):
                        essential_files.append({
                            "path": file_path,
                            "description": f"Main result file: {pattern}"
                        })
                        system_logger.info(f"Found essential file: {file_path.name}")
        
        # Look for prediction images (but not individual masks)
        for file_path in current_dir.glob("*_predictions.png"):
            if file_path.is_file():
                if not any(ef["path"] == file_path for ef in essential_files):
                    essential_files.append({
                        "path": file_path,
                        "description": "Combined prediction visualization"
                    })
                    system_logger.info(f"Found prediction visualization: {file_path.name}")
    
    # 3. Essential files from home directory (fallback)
    home_dir = Path.home()
    home_result_patterns = [
        "measurements_results.csv",
        "R50_flip_results.csv", 
        "class_color_legend.txt"
    ]
    
    for pattern in home_result_patterns:
        for file_path in home_dir.glob(pattern):
            if file_path.is_file():
                if not any(ef["path"] == file_path for ef in essential_files):
                    essential_files.append({
                        "path": file_path,
                        "description": f"Main result file: {pattern}"
                    })
                    system_logger.info(f"Found essential file in home: {file_path.name}")
    
    if not essential_files:
        system_logger.warning("No essential inference result files found to upload")
        return 0.0
    
    system_logger.info(f"Uploading {len(essential_files)} essential result files to {archive_path}")
    
    # Log what we're uploading vs what we're skipping
    if output_dir.exists():
        all_files_count = sum(1 for _ in output_dir.glob("*") if _.is_file())
        skipped_count = all_files_count - len([ef for ef in essential_files if ef["path"].parent == output_dir])
        system_logger.info(f"Uploading {len([ef for ef in essential_files if ef["path"].parent == output_dir])}/{all_files_count} files from output directory")
        system_logger.info(f"Skipped {skipped_count} files (individual masks, temporary files)")
    
    # Upload all essential files in batch using gsutil -m cp
    try:
        file_paths = [str(ef["path"]) for ef in essential_files]
        cmd = [
            "gsutil", "-m", "cp"
        ] + file_paths + [archive_path]
        
        result = run_gsutil_with_retry(cmd)
        system_logger.info(f"Batch upload completed successfully: {len(essential_files)} essential files uploaded")
        successful_uploads = len(essential_files)
        failed_uploads = 0
        
    except subprocess.CalledProcessError as e:
        system_logger.error(f"Batch upload failed: {e}")
        system_logger.info("Attempting individual file uploads as fallback...")
        
        # Fallback to individual uploads
        successful_uploads = 0
        failed_uploads = 0
        
        for file_info in essential_files:
            try:
                cmd = [
                    "gsutil", "-m", "cp",
                    str(file_info["path"]),
                    f"{archive_path}{file_info['path'].name}"
                ]
                
                result = run_gsutil_with_retry(cmd)
                system_logger.info(f"Uploaded: {file_info['path'].name} - {file_info['description']}")
                successful_uploads += 1
                
            except subprocess.CalledProcessError as e:
                system_logger.error(f"Failed to upload {file_info['path'].name}: {e}")
                failed_uploads += 1
    
    # Create and upload a comprehensive summary
    summary_content = f"""Inference Results Upload Summary
=====================================
Dataset: {dataset_name}
Model Configuration: {model_info}
Upload Timestamp: {timestamp}
Total Essential Files Found: {len(essential_files)}
Successfully Uploaded: {successful_uploads}
Failed Uploads: {failed_uploads}
Archive Location: {archive_path}
Upload Method: Selective batch upload (essential files only)

Files Uploaded:
"""
    
    for i, file_info in enumerate(essential_files, 1):
        file_path = file_info["path"]
        description = file_info["description"]
        summary_content += f"{i:3d}. {file_path.name} ({file_path.stat().st_size} bytes) - {description}\n"
    
    summary_content += f"""
Files Excluded from Upload:
- Individual mask images (*_mask_*.jpg)
- Temporary processing files
- Duplicate files from multiple locations

Note: This upload contains only essential result files. Individual particle masks 
are not uploaded to reduce storage usage and upload time. The main measurements 
CSV contains all analysis results, and the RLE CSV contains mask data for 
reconstruction if needed.
"""
    
    summary_path = output_dir / "inference_upload_summary.txt"
    try:
        with open(summary_path, "w") as f:
            f.write(summary_content)
        
        # Upload summary (if not already included)
        if not any(ef["path"] == summary_path for ef in essential_files):
            cmd = [
                "gsutil", "-m", "cp",
                str(summary_path),
                f"{archive_path}inference_upload_summary.txt"
            ]
            run_gsutil_with_retry(cmd)
            system_logger.info("Upload summary created and uploaded")
        
    except Exception as e:
        system_logger.error(f"Failed to create/upload summary: {e}")
    
    # Log final results
    upload_end_time = datetime.now()
    elapsed_time = (upload_end_time - upload_start_time).total_seconds()
    
    if successful_uploads > 0:
        system_logger.info(f"SELECTIVE INFERENCE UPLOAD COMPLETED: {successful_uploads}/{len(essential_files)} essential files uploaded")
        system_logger.info(f"Upload time: {elapsed_time:.2f} seconds")
        system_logger.info(f"Results available at: {archive_path}")
        system_logger.info("Individual mask files excluded to optimize upload size and time")
    else:
        system_logger.error("Essential file uploads failed!")
    
    return elapsed_time
