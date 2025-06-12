"""
Streamlit utility functions for the UW Computer Vision project.

This module provides backend utilities for the Streamlit GUI, including:
- Dataset management in Google Cloud Storage (GCS)
- File upload/download and zipping from GCS
- Progress and ETA estimation
- Password protection for admin features
- Utility functions for listing and formatting GCS contents

All functions are designed to integrate seamlessly with the Streamlit frontend.
"""

import json
import os
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
import yaml
from google.api_core import page_iterator
from google.cloud import storage

ADMIN_PASSWORD = "admin"

from src.utils.config import get_config

config = get_config()
bucket = config["bucket"]

# Absolute path to main.py
MAIN_SCRIPT_PATH = Path(config["paths"]["main_script"]).expanduser().resolve()
ETA_FILE = Path(config["paths"]["eta_file"]).expanduser().resolve()

# GCS bucket details
GCS_BUCKET_NAME = bucket
GCS_DATASET_FOLDER = "DATASET"
GCS_INFERENCE_FOLDER = "DATASET/INFERENCE"
GCS_ARCHIVE_FOLDER = "Archive"
GCS_DATASET_INFO_PATH = "dataset_info.json"


def create_zip_from_gcs(bucket_name, folder, zip_name="archive.zip"):
    """
    Creates a ZIP archive from files in a GCS bucket folder.

    Parameters:
    - bucket_name (str): Name of the GCS bucket
    - folder (str): Folder in the GCS bucket
    - zip_name (str): Name of the ZIP archive to create

    Returns:
    - bytes: ZIP archive as bytes
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder)

    with TemporaryDirectory() as tempdir:
        zip_path = os.path.join(tempdir, zip_name)
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for blob in blobs:
                if (
                    blob.name.endswith(".png")
                    or blob.name.endswith("results_x_pred_1.csv")
                    or blob.name.endswith("results_x_pred_0.csv")
                ):
                    file_path = os.path.join(tempdir, os.path.basename(blob.name))
                    blob.download_to_filename(file_path)
                    zipf.write(file_path, os.path.basename(blob.name))
        with open(zip_path, "rb") as f:
            bytes_zip = f.read()
    return bytes_zip


def check_password():
    """
    Checks the entered password against the admin password.

    Returns:
    - bool: True if the password is correct, else False
    """

    def password_entered():
        if st.session_state["password"] == ADMIN_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.error("Incorrect password")
        return False
    else:
        return True


def _item_to_value(iterator, item):
    """
    Helper function for GCS directory listing.

    Parameters:
    - iterator: GCS iterator
    - item: Current item

    Returns:
    - item: The item value
    """
    return item


def list_directories(bucket_name, prefix):
    """
    Lists directories in a GCS bucket with the given prefix.

    Parameters:
    - bucket_name (str): Name of the GCS bucket
    - prefix (str): Prefix to filter directories

    Returns:
    - list: List of directory paths
    """
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    extra_params = {"projection": "noAcl", "prefix": prefix, "delimiter": "/"}

    gcs = storage.Client()

    path = "/b/" + bucket_name + "/o"

    iterator = page_iterator.HTTPIterator(
        client=gcs,
        api_request=gcs._connection.api_request,
        path=path,
        items_key="prefixes",
        item_to_value=_item_to_value,
        extra_params=extra_params,
    )

    return [x for x in iterator]


def run_command(command):
    """
    Runs a shell command and captures its output.

    Parameters:
    - command (str): Command to run

    Returns:
    - tuple: (stdout, stderr, success_flag)
    """
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = [], []
    while process.poll() is None:
        output = process.stdout.readline()
        if output:
            stdout.append(output.strip())
    process.stdout.close()
    stderr_output = process.stderr.read()
    process.stderr.close()
    return "\n".join(stdout), stderr_output, process.returncode == 0


def list_png_files_in_gcs_folder(bucket_name, folder):
    """
    Lists .png files in a GCS folder.

    Parameters:
    - bucket_name (str): Name of the GCS bucket
    - folder (str): Folder in the GCS bucket

    Returns:
    - list: List of blob objects for .png files
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder)
    return [blob for blob in blobs if blob.name.endswith(".png")]


def list_specific_csv_files_in_gcs_folder(bucket_name, folder):
    """
    Lists specific .csv files in a GCS folder.

    Parameters:
    - bucket_name (str): Name of the GCS bucket
    - folder (str): Folder in the GCS bucket

    Returns:
    - list: List of blob objects for specific .csv files
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder)
    return [
        blob
        for blob in blobs
        if blob.name.endswith("results_x_pred_1.csv")
        or blob.name.endswith("results_x_pred_0.csv")
    ]


def contains_errors(stderr):
    """
    Checks if stderr contains any errors.

    Parameters:
    - stderr (str): Stderr output as string

    Returns:
    - bool: True if errors found, else False
    """
    error_keywords = ["error", "failed", "exception", "traceback", "critical"]
    return any(keyword in stderr.lower() for keyword in error_keywords)


def load_dataset_names_from_gcs():
    """
    Loads dataset names from a JSON file in GCS.

    Returns:
    - dict: Dataset names and details
    """
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_DATASET_INFO_PATH)

    try:
        data = json.loads(blob.download_as_bytes())
    except Exception as e:
        st.warning(f"dataset_info.json not found. Initializing a new one.")
        data = {}
        save_dataset_names_to_gcs(data)

    return data


def save_dataset_names_to_gcs(data):
    """
    Saves dataset names to a JSON file in GCS.

    Parameters:
    - data (dict): Dataset names and details
    """
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_DATASET_INFO_PATH)
    blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")
    st.write("Dataset info updated in GCS.")


def upload_files_to_gcs(bucket_name, target_folder, files, overwrite):
    """
    Uploads files to a GCS bucket.

    Parameters:
    - bucket_name (str): Name of the GCS bucket
    - target_folder (str): Target folder in the bucket
    - files (list): List of files to upload
    - overwrite (bool): Whether to overwrite existing files

    Returns:
    - bool: True if upload successful, else False
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    if overwrite:
        blobs = bucket.list_blobs(prefix=target_folder)
        for blob in blobs:
            blob.delete()
        st.write(f"Existing files in '{target_folder}' have been deleted.")

    for file in files:
        blob = bucket.blob(f"{target_folder}/{file.name}")
        blob.upload_from_file(file)
        st.write(f"Uploaded {file.name} to {target_folder}")


def format_and_sort_folders(folders):
    """
    Formats and sorts folder names for display.

    Parameters:
    - folders (list): List of folder paths

    Returns:
    - list: Sorted and formatted folder names
    """
    formatted_folders = []
    for folder in folders:
        folder = folder.rstrip("/")
        try:
            timestamp = datetime.strptime(folder.split("/")[-1], "%Y%m%d_%H%M%S")
            formatted_name = timestamp.strftime("%B %d, %Y %H:%M:%S")
            formatted_folders.append((folder, formatted_name))
        except ValueError:
            formatted_folders.append((folder, folder))

    formatted_folders.sort(key=lambda x: x[1], reverse=True)
    return formatted_folders


def estimate_eta(task, num_images=0):
    """
    Estimates the time remaining for a task based on historical data.

    Parameters:
    - task (str): Name of the task
    - num_images (int): Number of images for inference task

    Returns:
    - tuple: (download_eta, task_eta, upload_eta)
    """
    data = read_eta_data()
    if task == "inference":
        avg_time_per_image = data.get(task, {}).get("average_time_per_image", 1)
        buffer = data.get(task, {}).get("buffer", 1)
        task_eta = avg_time_per_image * num_images * buffer
    else:
        task_eta = data.get(task, {}).get("average_time", 60)

    download_eta = data.get("download", {}).get("average_time", 60)
    upload_eta = data.get("upload", {}).get("average_time", 60)

    return download_eta, task_eta, upload_eta


def read_eta_data():
    """
    Reads ETA data from the ETA file.

    Returns:
    - dict: ETA data if file exists, else empty dict
    """
    if os.path.exists(ETA_FILE):
        with open(ETA_FILE, "r") as file:
            return json.load(file)
    return {}
