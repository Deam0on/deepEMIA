"""
Google Cloud Storage utilities for the UW Computer Vision project.

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

import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from src.utils.config import get_config
from src.utils.logger_utils import system_logger

config = get_config()
bucket = config["bucket"]

# Resolve paths
local_dataset_path = Path(config["paths"]["local_dataset_root"]).expanduser().resolve()


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
    system_logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        system_logger.error(f"gsutil download failed: {result.stderr}")
    else:
        system_logger.info("gsutil download completed successfully.")

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
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            system_logger.error(f"PNG upload failed: {result.stderr}")

    # Upload .csv files
    csv_files = list(Path.home().glob("*.csv"))
    if csv_files:
        system_logger.info(f"Uploading {len(csv_files)} CSV files to {archive_path}")
        cmd = (
            ["gsutil", "-m", "cp", "-r"] + [str(f) for f in csv_files] + [archive_path]
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            system_logger.error(f"CSV upload failed: {result.stderr}")

    # Upload output directory contents
    output_dir = Path.home() / "output"
    if output_dir.exists() and any(output_dir.iterdir()):
        system_logger.info(f"Uploading output directory contents to {archive_path}")
        cmd = ["gsutil", "-m", "cp", "-r", str(output_dir) + "/*", archive_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            system_logger.error(f"Output directory upload failed: {result.stderr}")

    upload_end_time = datetime.now()
    return (upload_end_time - upload_start_time).total_seconds()
