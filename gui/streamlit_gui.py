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
import subprocess
import time
from io import BytesIO
from pathlib import Path

import streamlit as st
import yaml
from PIL import Image

from streamlit_functions import (
    check_password,
    contains_errors,
    create_zip_from_gcs,
    estimate_eta,
    format_and_sort_folders,
    list_directories,
    list_png_files_in_gcs_folder,
    list_specific_csv_files_in_gcs_folder,
    load_dataset_names_from_gcs,
    save_dataset_names_to_gcs,
    upload_files_to_gcs,
)

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

# Streamlit interface
st.title("DL-IA Control Panel")

# Task selection
st.header("Script controls")
use_new_data = st.checkbox("Use new data from bucket", value=False)

dataset_name = st.selectbox("Dataset Name", list(st.session_state.datasets.keys()))

# Wrap the dataset creation and deletion sections with password check
if check_password():
    new_dataset = st.checkbox("New dataset")
    if new_dataset:
        new_dataset_name = st.text_input("Enter new dataset name")
        if new_dataset_name:
            path1 = f"/home//DATASET/{new_dataset_name}/"
            path2 = path1
            new_classes = st.text_input("Enter classes (comma separated)")
            if st.button("Add Dataset"):
                classes = (
                    [cls.strip() for cls in new_classes.split(",")]
                    if new_classes
                    else []
                )
                if new_dataset_name and classes:
                    st.session_state.datasets[new_dataset_name] = [
                        path1,
                        path2,
                        classes,
                    ]
                    save_dataset_names_to_gcs(st.session_state.datasets)
                    st.success(f"Dataset '{new_dataset_name}' added.")
                else:
                    st.warning("Please enter a valid dataset name and classes.")

    confirm_deletion = st.checkbox("Confirm Deletion")
    if st.button("Remove Dataset"):
        if confirm_deletion:
            del st.session_state.datasets[dataset_name]
            save_dataset_names_to_gcs(st.session_state.datasets)
            st.success(f"Dataset '{dataset_name}' deleted.")
            st.session_state.confirm_delete = (
                False  # Automatically uncheck after deletion
            )
            st.experimental_rerun()  # Refresh to reflect deletion
        else:
            st.warning("Please check the confirmation box to delete the dataset.")

# Define the task mapping
task_mapping = {
    "inference": "MEASUREMENT",
    "evaluate": "MODEL EVALUATE",
    "prepare": "DATA PREPARATION FOR TRAINING",
    "train": "NEW MODEL TRAINING",
}

# Change the task order and create a reverse mapping for the command
tasks_order = ["inference", "evaluate", "prepare", "train"]
reverse_task_mapping = {v: k for k, v in task_mapping.items()}

# Update task selection with user-friendly names
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

# Update the task execution button code to handle the combined ETA
if st.button("Run Task"):
    visualize_flag = "--visualize"  # Always true
    upload_flag = "--upload"  # Always true
    download_flag = "--download"

    # Calculate ETAs
    download_eta, task_eta, upload_eta = estimate_eta(task)
    total_eta = download_eta + task_eta + upload_eta

    command = f"python3 {MAIN_SCRIPT_PATH} --task {task} --dataset_name {dataset_name} --threshold {threshold} {visualize_flag} {download_flag} {upload_flag}"
    st.info(f"Running: {command}")

    def update_progress_bar_and_countdown(start_time, eta, phase):
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
            if phase == "Task in progress" and not process.poll() is None:
                # If task phase and the command finished, exit the loop
                break

    with st.spinner("Running task..."):
        progress_bar = st.progress(0)
        countdown_placeholder = st.empty()
        start_time = time.time()

        # Download Phase
        update_progress_bar_and_countdown(start_time, download_eta, "Downloading")

        # Task Phase
        task_start_time = time.time()  # Reset start time for task phase
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        update_progress_bar_and_countdown(task_start_time, task_eta, "Task in progress")

        stdout, stderr = process.communicate()

        # Upload Phase
        start_time = time.time()  # Reset start time for upload phase
        update_progress_bar_and_countdown(start_time, upload_eta, "Uploading")

        # Ensure the progress bar is full at the end
        progress_bar.progress(100)
        countdown_placeholder.text("Task Completed")

    st.text(stdout)
    st.session_state.stderr = stderr  # Store stderr in session state

    # Reset the show_errors state if there are new errors
    if stderr:
        st.session_state.show_errors = True
    else:
        st.success(f"{task.capitalize()} task completed successfully!")

# Conditionally show the upload section
upload_folder_mapping = {
    f"{GCS_DATASET_FOLDER}/{dataset_name}": "TRAINING DATA",
    GCS_INFERENCE_FOLDER: "MEASUREMENT DATA",
}
reverse_upload_folder_mapping = {v: k for k, v in upload_folder_mapping.items()}

if use_new_data:
    st.header("Upload Files to GCS")
    upload_folder_display_names = [
        upload_folder_mapping[f"{GCS_DATASET_FOLDER}/{dataset_name}"],
        upload_folder_mapping[GCS_INFERENCE_FOLDER],
    ]
    selected_upload_folder_display = st.selectbox(
        "Select folder to upload to", upload_folder_display_names
    )
    upload_folder = reverse_upload_folder_mapping[selected_upload_folder_display]
    overwrite = st.checkbox("Overwrite existing data in the folder")
    uploaded_files = st.file_uploader(
        "Choose files to upload", accept_multiple_files=True
    )
    if st.button("Upload Files") and uploaded_files:
        with st.spinner("Uploading files..."):
            upload_files_to_gcs(
                GCS_BUCKET_NAME, upload_folder, uploaded_files, overwrite
            )
        st.success("Files uploaded successfully.")

has_errors = contains_errors(st.session_state.stderr)
has_stderr = bool(st.session_state.stderr)

# Auto-expand if actual errors are present
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

# Apply formatting and sorting
formatted_folders = format_and_sort_folders(st.session_state.folders)

if formatted_folders:
    folder_dropdown = st.selectbox(
        "Select Folder",
        formatted_folders,
        format_func=lambda x: x[1],  # Display the formatted name
    )
else:
    st.write("No folders found in the GCS bucket.")

selected_folder = folder_dropdown[0]

# Button to show inference images
if st.button("Show Inference Images") and st.session_state.folders:
    st.session_state.show_images = True

# Display images if available
if st.session_state.show_images:
    st.write(
        f"Displaying images from folder: {folder_dropdown[1]}"
    )  # Use formatted name for display
    image_files = list_png_files_in_gcs_folder(
        GCS_BUCKET_NAME, selected_folder
    )  # Use original name for operations
    if image_files:
        for blob in image_files:
            img_bytes = blob.download_as_bytes()
            img = Image.open(BytesIO(img_bytes))
            st.image(img, caption=os.path.basename(blob.name))
    else:
        st.write("No images found in the selected folder.")

    # Button to download specific CSV files
    csv_files = list_specific_csv_files_in_gcs_folder(
        GCS_BUCKET_NAME, selected_folder
    )  # Use original name for operations
    if csv_files:
        for blob in csv_files:
            csv_bytes = blob.download_as_bytes()
            csv_name = os.path.basename(blob.name)
            if csv_name == "results_x_pred_1.csv":
                if dataset_name != "hw_patterns":
                    download_name = "results_pores.csv"
                else:
                    download_name = "results_cyclones.csv"
            elif csv_name == "results_x_pred_0.csv":
                if dataset_name != "hw_patterns":
                    download_name = "results_throats.csv"
                else:
                    download_name = "results_flows.csv"
            else:
                continue  # Skip files that don't match the specific names
            st.download_button(
                label=f"Download {download_name}",
                data=csv_bytes,
                file_name=download_name,
                mime="text/csv",
            )
    else:
        st.write("No specific CSV files found in the selected folder.")

    # Add "Download All" button
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
