"""
Streamlit web interface for the UW Computer Vision project.

This module provides a user-friendly web interface for:
- Dataset management
- Model training and evaluation
- Running inference
- Visualizing results
- Downloading predictions and visualizations

The interface integrates with Google Cloud Storage for data management and
provides real-time progress tracking and ETA estimation.
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from io import BytesIO
from PIL import Image
import time
import subprocess
import streamlit as st
from streamlit_functions import (check_password, contains_errors,
                                 create_zip_from_gcs, estimate_eta,
                                 format_and_sort_folders, list_directories,
                                 list_png_files_in_gcs_folder,
                                 list_specific_csv_files_in_gcs_folder,
                                 load_dataset_names_from_gcs,
                                 save_dataset_names_to_gcs,
                                 upload_files_to_gcs)

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

# Initialize session state
if "show_errors" not in st.session_state:
    st.session_state.show_errors = False
if "stderr" not in st.session_state:
    st.session_state.stderr = ""
if "folders" not in st.session_state:
    st.session_state.folders = []
if "show_images" not in st.session_state:
    st.session_state.show_images = False
if "datasets" not in st.session_state:
    st.session_state.datasets = load_dataset_names_from_gcs()
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False

st.title("DL-IA Control Panel")
st.header("Script controls")
# --- Changed: Default to True ---
use_new_data = st.checkbox("Use new data from bucket", value=True)

dataset_name = st.selectbox("Dataset Name", list(st.session_state.datasets.keys()))


def add_new_dataset(new_dataset_name: str, new_classes: str):
    """
    Adds a new dataset to the session state and saves it to GCS.

    Parameters:
    - new_dataset_name (str): Name of the new dataset
    - new_classes (str): Comma-separated string of class names

    Returns:
    - None
    """
    path1 = f"/home/DATASET/{new_dataset_name}/"
    path2 = path1
    classes = [cls.strip() for cls in new_classes.split(",")] if new_classes else []
    if new_dataset_name and classes:
        st.session_state.datasets[new_dataset_name] = [path1, path2, classes]
        save_dataset_names_to_gcs(st.session_state.datasets)
        st.success(f"Dataset '{new_dataset_name}' added.")
    else:
        st.warning("Please enter a valid dataset name and classes.")


def remove_dataset(dataset_name: str):
    """
    Removes a dataset from the session state and GCS.

    Parameters:
    - dataset_name (str): Name of the dataset to remove

    Returns:
    - None
    """
    del st.session_state.datasets[dataset_name]
    save_dataset_names_to_gcs(st.session_state.datasets)
    st.success(f"Dataset '{dataset_name}' deleted.")
    st.session_state.confirm_delete = False
    st.experimental_rerun()


# Dataset creation and deletion (admin only)
if check_password():
    new_dataset = st.checkbox("New dataset")
    if new_dataset:
        new_dataset_name = st.text_input("Enter new dataset name")
        new_classes = st.text_input("Enter class names (comma-separated)")
        if st.button("Add Dataset"):
            add_new_dataset(new_dataset_name, new_classes)

    confirm_deletion = st.checkbox("Confirm Deletion")
    if st.button("Remove Dataset"):
        if confirm_deletion:
            remove_dataset(dataset_name)
        else:
            st.warning("Please confirm deletion before removing a dataset.")

# Define the task mapping
task_mapping = {
    "inference": "MEASUREMENT",
    "evaluate": "MODEL EVALUATE",
    "prepare": "DATA PREPARATION FOR TRAINING",
    "train": "NEW MODEL TRAINING",
}

tasks_order = ["inference", "evaluate", "prepare", "train"]
reverse_task_mapping = {v: k for k, v in task_mapping.items()}

task_display_names = [task_mapping[task] for task in tasks_order]
selected_task_display = st.selectbox("Select Task", task_display_names)
task = reverse_task_mapping[selected_task_display]

threshold = st.slider(
    "Select Detection Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.65,
    step=0.01,
    help="Adjust the detection threshold for the model.",
)

# --- NEW: RCNN Model Selection ---
rcnn_model = st.selectbox(
    "Select RCNN Backbone",
    options=["101", "50", "combo"],
    index=0,
    help="Choose the RCNN backbone: 101, 50, or combo (both)."
)

def update_progress_bar_and_countdown(
    start_time, eta, phase, progress_bar, countdown_placeholder, total_eta, process=None
):
    """
    Updates the progress bar and countdown timer during task execution.

    Parameters:
    - start_time (float): Start time of the phase
    - eta (float): Estimated time for the phase
    - phase (str): Current phase name
    - progress_bar (st.progress): Streamlit progress bar object
    - countdown_placeholder (st.empty): Streamlit placeholder for countdown
    - total_eta (float): Total estimated time for all phases
    - process (subprocess.Popen, optional): Process to monitor for completion

    Returns:
    - None
    """
    end_time = start_time + eta
    while time.time() < end_time:
        elapsed_time = time.time() - start_time
        remaining_time = end_time - time.time()
        hours, remainder = divmod(remaining_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        countdown_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        countdown_placeholder.text(f"{phase} - Time Remaining: {countdown_str}")
        progress_percentage = min(elapsed_time / total_eta, 1.0)
        progress_bar.progress(progress_percentage)
        time.sleep(1)
        if (
            phase == "Task in progress"
            and process is not None
            and process.poll() is not None
        ):
            break


# Task execution
if st.button("Run Task"):
    visualize_flag = "--visualize"
    upload_flag = "--upload"
    download_flag = "--download"

    download_eta, task_eta, upload_eta = estimate_eta(task)
    total_eta = download_eta + task_eta + upload_eta

    # --- Pass rcnn_model to command ---
    command = (
        f"python3 {MAIN_SCRIPT_PATH} --task {task} --dataset_name {dataset_name} "
        f"--threshold {threshold} --rcnn {rcnn_model} {visualize_flag} {download_flag} {upload_flag}"
    )
    st.info(f"Running: {command}")

    with st.spinner("Running task..."):
        progress_bar = st.progress(0)
        countdown_placeholder = st.empty()
        start_time = time.time()

        # Download Phase
        update_progress_bar_and_countdown(
            start_time,
            download_eta,
            "Downloading",
            progress_bar,
            countdown_placeholder,
            total_eta,
        )

        # Task Phase
        task_start_time = time.time()
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()
        progress_bar.progress(1.0)
        st.text(stdout)
        st.session_state.stderr = stderr

        # --- NEW: Print log output after run is complete ---
        logs_dir = Path(config["paths"].get("logs_dir", "~/logs")).expanduser().resolve()
        log_file = logs_dir / "full.log"
        if log_file.exists():
            with open(log_file, "r") as lf:
                log_content = lf.read()
            st.subheader("Run Log Output")
            st.code(log_content, language="text")
        else:
            st.warning(f"Log file not found: {log_file}")

    if stderr:
        st.error("Errors occurred during execution. See below.")
    else:
        st.success("Task completed successfully.")

# --- Changed: Default to measurement data and overwrite ---
upload_folder_mapping = {
    GCS_INFERENCE_FOLDER: "MEASUREMENT DATA",
    f"{GCS_DATASET_FOLDER}/{dataset_name}": "TRAINING DATA",
}
reverse_upload_folder_mapping = {v: k for k, v in upload_folder_mapping.items()}

if use_new_data:
    st.header("Upload Files to GCS")
    upload_folder_display_names = [
        upload_folder_mapping[GCS_INFERENCE_FOLDER],
        upload_folder_mapping[f"{GCS_DATASET_FOLDER}/{dataset_name}"],
    ]
    # --- Default to MEASUREMENT DATA ---
    selected_upload_folder_display = st.selectbox(
        "Select folder to upload to", upload_folder_display_names, index=0
    )
    upload_folder = reverse_upload_folder_mapping[selected_upload_folder_display]
    # --- Default overwrite to True ---
    overwrite = st.checkbox("Overwrite existing data in the folder", value=True)
    uploaded_files = st.file_uploader(
        "Choose files to upload", accept_multiple_files=True
    )
    if st.button("Upload Files") and uploaded_files:
        upload_files_to_gcs(GCS_BUCKET_NAME, upload_folder, uploaded_files, overwrite)

# Error and warning display
has_errors = contains_errors(st.session_state.stderr)
has_stderr = bool(st.session_state.stderr)
expand_expander = has_errors

if has_stderr:
    with st.expander("Show errors and warnings", expanded=expand_expander):
        if has_errors:
            st.error(st.session_state.stderr)
        else:
            st.warning(st.session_state.stderr)

# List folders in the GCS bucket
st.header("Google Cloud Storage")
if "folders" not in st.session_state or not st.session_state.folders:
    st.session_state.folders = list_directories(GCS_BUCKET_NAME, GCS_ARCHIVE_FOLDER)

formatted_folders = format_and_sort_folders(st.session_state.folders)

if formatted_folders:
    folder_dropdown = st.selectbox(
        "Select Folder",
        formatted_folders,
        format_func=lambda x: x[1],
    )
else:
    st.write("No folders found in the GCS bucket.")

selected_folder = folder_dropdown[0]

# Show inference images
if st.button("Show Inference Images") and st.session_state.folders:
    st.session_state.show_images = True

if st.session_state.show_images:
    st.write(f"Displaying images from folder: {folder_dropdown[1]}")
    image_files = list_png_files_in_gcs_folder(GCS_BUCKET_NAME, selected_folder)
    if image_files:
        for blob in image_files:
            img_bytes = blob.download_as_bytes()
            img = Image.open(BytesIO(img_bytes))
            st.image(img, caption=os.path.basename(blob.name))
    else:
        st.write("No images found in the selected folder.")

    # Download specific CSV files
    csv_files = list_specific_csv_files_in_gcs_folder(GCS_BUCKET_NAME, selected_folder)
    if csv_files:
        for blob in csv_files:
            csv_bytes = blob.download_as_bytes()
            csv_name = os.path.basename(blob.name)
            if csv_name == "results_x_pred_1.csv":
                download_name = (
                    "results_pores.csv"
                    if dataset_name != "hw_patterns"
                    else "results_cyclones.csv"
                )
            elif csv_name == "results_x_pred_0.csv":
                download_name = (
                    "results_throats.csv"
                    if dataset_name != "hw_patterns"
                    else "results_flows.csv"
                )
            else:
                continue
            st.download_button(
                label=f"Download {download_name}",
                data=csv_bytes,
                file_name=download_name,
                mime="text/csv",
            )
    else:
        st.write("No specific CSV files found in the selected folder.")

    # Download all as zip
    if st.button("Download All as Zip"):
        zip_bytes = create_zip_from_gcs(GCS_BUCKET_NAME, selected_folder)
        st.download_button(
            label="Download All",
            data=zip_bytes,
            file_name="inference_results.zip",
            mime="application/zip",
        )

# Help and Tips Section
st.sidebar.title("Help & Tips")
st.sidebar.subheader("Tips")
st.sidebar.write(
    """
- **Dataset Preparation:** Ensure your dataset is well-structured.
- **Inference:** Use the slider to adjust the detection threshold.
- **Uploading Files:** Use the 'Upload Files' section to add new data.
"""
)

st.sidebar.subheader("Tutorials")
st.sidebar.write(
    """
    - **Step 1:** Select the task you want to run (prepare, train, evaluate, inference).
    - **Step 2:** Choose the dataset name from the dropdown.
    - **Step 3:** Adjust the detection threshold using the slider.
    - **Step 4:** Optionally, use new data from the bucket by checking the respective box.
    - **Step 5:** Click 'Run Task' to execute the selected task.
    """
)
