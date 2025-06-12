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

ETA data is stored in a JSON file.
"""

import json
import logging
import os
from pathlib import Path

import yaml

from src.utils.config import get_config

config = get_config()

# Resolve paths
ETA_FILE = Path(config["paths"]["eta_file"]).expanduser().resolve()

DEFAULT_ETA = {
    "prepare": {"average_time": 300},
    "evaluate": {"average_time": 1800},
    "inference": {"average_time_per_image": 5, "buffer": 1.1},
    "download": {"average_time": 60},
    "upload": {"average_time": 60},
}


def read_eta_data() -> dict:
    """
    Read ETA data from a JSON file.

    Returns:
        dict: ETA data containing timing information for different tasks.
    """
    if os.path.exists(ETA_FILE):
        try:
            with open(ETA_FILE, "r") as file:
                data = json.load(file)
            logging.info(f"Loaded ETA data from {ETA_FILE}")
            return data
        except Exception as e:
            logging.error(f"Failed to read ETA file {ETA_FILE}: {e}")
            return DEFAULT_ETA.copy()
    else:
        logging.info(f"ETA file {ETA_FILE} not found. Using default values.")
        return DEFAULT_ETA.copy()


def update_eta_data(task: str, time_taken: float, num_images: int = 0) -> None:
    """
    Update ETA data with new timings.

    Parameters:
        task (str): Task name ('inference', 'prepare', 'evaluate', 'download', 'upload')
        time_taken (float): Time taken for the task in seconds
        num_images (int, optional): Number of images processed. Required for 'inference' task.

    Note:
        The function automatically handles different averaging strategies
        for inference vs. other tasks.
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
        logging.info(
            f"Updated inference ETA: {new_avg_time_per_image:.2f}s/image (buffer: {buffer})"
        )
    else:
        current_avg = data.get(task, {}).get("average_time", time_taken)
        new_avg_time = (current_avg + time_taken) / 2
        data[task] = {"average_time": new_avg_time}
        logging.info(f"Updated {task} ETA: {new_avg_time:.2f}s")

    try:
        with open(ETA_FILE, "w") as file:
            json.dump(data, file, indent=2)
        logging.info(f"Saved updated ETA data to {ETA_FILE}")
    except Exception as e:
        logging.error(f"Failed to write ETA file {ETA_FILE}: {e}")
