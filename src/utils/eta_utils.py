"""
Estimated Time of Arrival (ETA) utilities for the UW Computer Vision project.

This module provides functions for:
- Reading and updating ETA data for various tasks
- Tracking and averaging execution times
- Managing task-specific timing information

The module handles:
- Data preparation timing
- Model evaluation timing
- Inference timing per image
- Download/upload timing
- Automatic ETA updates and averaging
"""

import json
import os
from pathlib import Path

import yaml

# Load config once at the start of your program
with open(Path.home() / "uw-com-vision" / "config" / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Resolve paths
ETA_FILE = Path(config["paths"]["eta_file"]).expanduser().resolve()


def read_eta_data():
    """
    Read ETA data from a JSON file.

    This function:
    1. Checks if the ETA file exists
    2. If exists, loads the current ETA data
    3. If not exists, returns default timing values

    Returns:
        dict: ETA data containing timing information for different tasks:
            - prepare: Data preparation timing
            - evaluate: Model evaluation timing
            - inference: Per-image inference timing with buffer
            - download: Data download timing
            - upload: Data upload timing
    """
    if os.path.exists(ETA_FILE):
        with open(ETA_FILE, "r") as file:
            return json.load(file)
    else:
        return {
            "prepare": {"average_time": 300},
            "evaluate": {"average_time": 1800},
            "inference": {"average_time_per_image": 5, "buffer": 1.1},
            "download": {"average_time": 60},
            "upload": {"average_time": 60},
        }


def update_eta_data(task, time_taken, num_images=0):
    """
    Update ETA data with new timings.

    This function:
    1. Reads current ETA data
    2. Updates timing information based on task type
    3. Calculates new averages
    4. Saves updated data to file

    For inference tasks:
    - Calculates per-image timing
    - Maintains a buffer factor
    - Updates average time per image

    For other tasks:
    - Updates simple average timing

    Parameters:
        task (str): Task name ('inference', 'prepare', 'evaluate', 'download', 'upload')
        time_taken (float): Time taken for the task in seconds
        num_images (int, optional): Number of images processed. Required for 'inference' task.
            Defaults to 0.

    Note:
        The function automatically handles different averaging strategies
        for inference vs. other tasks
    """
    data = read_eta_data()

    if task == "inference":
        avg_time_per_image = time_taken / max(num_images, 1)
        current_avg = data.get(task, {}).get(
            "average_time_per_image", avg_time_per_image
        )
        buffer = data.get(task, {}).get("buffer", 1.1)

        new_avg_time_per_image = (current_avg + avg_time_per_image) / 2
        data[task] = {
            "average_time_per_image": new_avg_time_per_image,
            "buffer": buffer,
        }
    else:
        current_avg = data.get(task, {}).get("average_time", time_taken)

        new_avg_time = (current_avg + time_taken) / 2
        data[task] = {"average_time": new_avg_time}

    with open(ETA_FILE, "w") as file:
        json.dump(data, file, indent=2)
