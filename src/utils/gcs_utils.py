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
"""

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# Load bucket name from config.yaml
with open(Path.home() / "uw-com-vision" / "config" / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
bucket = config["bucket"]

# Load config once at the start of your program
with open(Path.home() / "uw-com-vision" / "config" / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Resolve paths
local_dataset_path = Path(config["paths"]["local_dataset_root"]).expanduser().resolve()


def download_data_from_bucket():
    """
    Download data from a Google Cloud Storage bucket to a local directory.

    This function:
    1. Removes any existing local dataset directory
    2. Downloads the entire dataset from GCS
    3. Tracks and returns the download duration

    Returns:
        float: Time taken to download data in seconds

    Note:
        Requires gsutil to be installed and configured with appropriate permissions
    """
    download_start_time = datetime.now()
    dirpath = Path.home() / "DATASET"
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    os.system(f"gsutil -m cp -r gs://{bucket}/DATASET {local_dataset_path}")
    download_end_time = datetime.now()

    return (download_end_time - download_start_time).total_seconds()


def upload_data_to_bucket():
    """
    Upload data from local directories to a Google Cloud Storage bucket.

    This function:
    1. Creates a timestamped archive directory in the bucket
    2. Uploads PNG files if present
    3. Uploads CSV files if present
    4. Uploads output directory contents if present
    5. Tracks and returns the upload duration

    The upload process:
    - Uses parallel uploads for better performance
    - Only uploads files that exist locally
    - Organizes uploads in timestamped directories
    - Handles different file types separately

    Returns:
        float: Time taken to upload data in seconds

    Note:
        Requires gsutil to be installed and configured with appropriate permissions
    """
    upload_start_time = datetime.now()
    time_offset = timedelta(hours=2)
    timestamp = (datetime.now() + time_offset).strftime("%Y%m%d_%H%M%S")
    archive_path = f"gs://{bucket}/Archive/{timestamp}/"

    # Check and upload .png files
    if any(fname.endswith(".png") for fname in os.listdir(local_dataset_path)):
        local_path = Path.home() / "*.png"
        os.system(f"gsutil -m cp -r {local_path} {archive_path}")

    # Check and upload .csv files
    if any(fname.endswith(".csv") for fname in os.listdir(local_dataset_path)):
        local_path = Path.home() / "*.csv"
        os.system(f"gsutil -m cp -r {local_path} {archive_path}")

    # Check and upload files in the output directory
    if os.path.exists(local_dataset_path / "output") and os.listdir(
        local_dataset_path / "output"
    ):
        local_path = Path.home() / "output/*"
        os.system(f"gsutil -m cp -r {local_path} {archive_path}")

    upload_end_time = datetime.now()
    return (upload_end_time - upload_start_time).total_seconds()
