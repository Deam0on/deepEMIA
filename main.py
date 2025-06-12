"""
Main pipeline script for the UW Computer Vision project.

This script provides a command-line interface for running various computer vision tasks:
- Dataset preparation
- Model training
- Model evaluation
- Inference on new images

The script integrates with Google Cloud Storage for data management and provides
progress tracking and ETA estimation for long-running tasks.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import yaml

from src.data.datasets import split_dataset
from src.functions.evaluate_model import evaluate_model
from src.functions.inference import run_inference
from src.functions.train_model import train_on_dataset
from src.utils.eta_utils import update_eta_data
from src.utils.gcs_utils import download_data_from_bucket, upload_data_to_bucket

# Load config once at the start of your program
with open(Path.home() / "uw-com-vision" / "config" / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Resolve paths
SPLIT_DIR = Path(config["paths"]["split_dir"]).expanduser().resolve()
CATEGORY_JSON = Path(config["paths"]["category_json"]).expanduser().resolve()
ETA_FILE = Path(config["paths"]["eta_file"]).expanduser().resolve()
local_dataset_path = Path(config["paths"]["local_dataset_root"]).expanduser().resolve()

# Load bucket name from config.yaml
with open(Path.home() / "uw-com-vision" / "config" / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
bucket = config["bucket"]


def setup_config():
    """
    Interactive setup for first-time configuration.
    Prompts the user for bucket name and scale_bar_rois settings,
    then writes them to config/config.yaml.
    """
    print("=== UW Computer Vision Project Setup ===")
    config_path = Path.home() / "uw-com-vision" / "config" / "config.yaml"
    if config_path.exists():
        overwrite = input(f"Config file already exists at {config_path}. Overwrite? (y/n): ").strip().lower()
        if overwrite != "y":
            print("Setup cancelled.")
            return

    bucket = input("Enter your Google Cloud Storage bucket name: ").strip()

    print("\nConfigure scale_bar_rois (press Enter to use defaults):")
    x_start = input("  x_start_factor [default 0.667]: ").strip() or "0.667"
    y_start = input("  y_start_factor [default 0.866]: ").strip() or "0.866"
    width = input("  width_factor [default 1]: ").strip() or "1"
    height = input("  height_factor [default 0.067]: ").strip() or "0.067"

    config = {
        "bucket": bucket,
        "paths": {
            "main_script": "~/uw-com-vision/main.py",
            "split_dir": "~/split_dir",
            "category_json": "~/uw-com-vision/dataset_info.json",
            "eta_file": "~/uw-com-vision/config/eta_data.json",
            "local_dataset_root": "~",
        },
        "scale_bar_rois": {
            "default": {
                "x_start_factor": float(x_start),
                "y_start_factor": float(y_start),
                "width_factor": float(width),
                "height_factor": float(height),
            }
        }
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"Configuration saved to {config_path}.")


def main():
    """
    Main function that parses command line arguments and executes the requested task.

    Tasks include:
    - prepare: Split dataset into train and test sets
    - train: Train a model on the dataset
    - evaluate: Evaluate the trained model
    - inference: Run inference on new data

    The function handles:
    - Data download/upload from/to Google Cloud Storage
    - Progress tracking and ETA estimation
    - Task execution with appropriate parameters
    """
    parser = argparse.ArgumentParser(
        description="Pipeline for preparing data, training, evaluating, and running inference on models."
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["prepare", "train", "evaluate", "inference", "setup"],
        help="Task to perform:\n"
        "- 'prepare': Prepare the dataset by splitting into train and test sets.\n"
        "- 'train': Train a model on the dataset.\n"
        "- 'evaluate': Evaluate the trained model on the test set.\n"
        "- 'inference': Run inference on new data using the trained model.\n"
        "- 'setup': Run first-time configuration setup.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to use (e.g., 'polyhipes').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Threshold for inference. Default is 0.65.",
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="json",
        choices=["json", "coco"],
        help="The format of the dataset annotations. 'json' for the custom one-JSON-per-image format, 'coco' for the standard COCO format.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Flag to visualize results during evaluation and inference. Saves visualizations of predictions. Default is False.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=True,
        help="Flag to download data from Google Cloud Storage before executing the task. Default is True.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        default=True,
        help="Flag to upload results to Google Cloud Storage after executing the task. Default is True.",
    )
    parser.add_argument(
        "--id",
        dest="draw_id",
        action="store_true",
        help="Draw instance ID on inference overlays",
    )
    parser.set_defaults(draw_id=False)
    parser.add_argument(
        "--rcnn",
        type=str,
        default="101",
        choices=["50", "101", "combo"],
        help="RCNN backbone to use: '50', '101', or 'combo' (both). Default is '101'."
    )

    args = parser.parse_args()

    if args.task == "setup":
        setup_config()
        return

    local_path = Path.home() / "uw-com-vision"
    os.system(f"gsutil -m cp -r gs://{bucket}/dataset_info.json {local_path}")

    img_dir = os.path.join(local_dataset_path / "DATASET", args.dataset_name)
    output_dir = SPLIT_DIR

    total_start_time = datetime.now()
    download_time_taken = 0
    upload_time_taken = 0

    print(f"Running task: {args.task} on dataset: {args.dataset_name}")

    if args.download:
        print(f"Downloading data for dataset {args.dataset_name} from bucket...")
        download_time_taken = download_data_from_bucket()

    if args.task == "prepare":
        print(f"Preparing dataset {args.dataset_name}...")
        task_start_time = datetime.now()
        split_dataset(img_dir, args.dataset_name)
        task_end_time = datetime.now()

    elif args.task == "train":
        print(
            f"Training model on dataset {args.dataset_name} using '{args.dataset_format}' format and RCNN {args.rcnn}..."
        )
        train_on_dataset(
            args.dataset_name, output_dir, dataset_format=args.dataset_format, rcnn=args.rcnn
        )

    elif args.task == "evaluate":
        print(
            f"Evaluating model on dataset {args.dataset_name} using '{args.dataset_format}' format..."
        )
        task_start_time = datetime.now()
        evaluate_model(
            args.dataset_name,
            output_dir,
            args.visualize,
            dataset_format=args.dataset_format,
        )
        task_end_time = datetime.now()

    elif args.task == "inference":
        print(
            f"Running inference on dataset {args.dataset_name} using '{args.dataset_format}' format and RCNN {args.rcnn}..."
        )

        os.system("rm -f *.png")
        os.system("rm -f *.csv")
        os.system("rm -f *.jpg")

        num_images = len(
            [f for f in os.listdir(img_dir) if f.endswith((".tif", ".png", ".jpg"))]
        )

        task_start_time = datetime.now()
        run_inference(
            args.dataset_name,
            output_dir,
            visualize=args.visualize,
            threshold=args.threshold,
            draw_id=args.draw_id,
            dataset_format=args.dataset_format,
            rcnn=args.rcnn,  # <-- pass the rcnn argument
        )

        task_end_time = datetime.now()

        inference_time_taken = (task_end_time - task_start_time).total_seconds()
        update_eta_data("inference", inference_time_taken, num_images)

    total_end_time = datetime.now()
    total_time_taken = (total_end_time - total_start_time).total_seconds()

    if args.upload:
        print(f"Uploading results for dataset {args.dataset_name} to bucket...")
        upload_time_taken = upload_data_to_bucket()

    if args.task != "inference":
        update_eta_data(args.task, total_time_taken)

    if args.download:
        update_eta_data("download", download_time_taken)
    if args.upload:
        update_eta_data("upload", upload_time_taken)


if __name__ == "__main__":
    main()
