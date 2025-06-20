"""
UW Computer Vision Project - Main Pipeline Script

This script provides a command-line interface for running dataset preparation, model training,
evaluation, and inference tasks. It integrates with Google Cloud Storage and tracks progress/ETA.
system_logger is configured to print simplified logs to the terminal and full logs to logs/full.log.
"""

import argparse
import atexit
import glob
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from src.data.datasets import split_dataset
from src.functions.evaluate_model import evaluate_model
from src.functions.inference import run_inference
from src.functions.train_model import train_on_dataset
from src.utils.config import get_config
from src.utils.eta_utils import update_eta_data
from src.utils.gcs_utils import download_data_from_bucket, upload_data_to_bucket
from src.utils.logger_utils import system_logger

config = get_config()
bucket = config["bucket"]
SPLIT_DIR = Path(config["paths"]["split_dir"]).expanduser().resolve()
CATEGORY_JSON = Path(config["paths"]["category_json"]).expanduser().resolve()
ETA_FILE = Path(config["paths"]["eta_file"]).expanduser().resolve()
local_dataset_path = Path(config["paths"]["local_dataset_root"]).expanduser().resolve()
LOGS_DIR = Path(config["paths"].get("logs_dir", "~/logs")).expanduser().resolve()


def setup_config():
    """
    Interactive setup for first-time configuration.
    Prompts the user for bucket name and scale_bar_rois settings,
    then writes them to config/config.yaml.
    """
    print("=== UW Computer Vision Project Setup ===")
    config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"
    if config_path.exists():
        overwrite = (
            input(f"Config file already exists at {config_path}. Overwrite? (y/n): ")
            .strip()
            .lower()
        )
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
            "main_script": "~/deepEMIA/main.py",
            "split_dir": "~/split_dir",
            "category_json": "~/deepEMIA/dataset_info.json",
            "eta_file": "~/deepEMIA/config/eta_data.json",
            "local_dataset_root": "~",
        },
        "scale_bar_rois": {
            "default": {
                "x_start_factor": float(x_start),
                "y_start_factor": float(y_start),
                "width_factor": float(width),
                "height_factor": float(height),
            }
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"Configuration saved to {config_path}.")


def main():
    """
    Main function that parses command line arguments and executes the requested task.

    Handles:
    - Data download/upload from/to Google Cloud Storage
    - Progress tracking and ETA estimation
    - Task execution with appropriate parameters
    """
    parser = argparse.ArgumentParser(
        description="Computer Vision Pipeline: Prepare data, train, evaluate, and run inference.",
        epilog="""
Examples:
  Prepare a dataset:
    python main.py --task prepare --dataset_name <dataset_name>

  Train a model (default: R101 backbone):
    python main.py --task train --dataset_name <dataset_name> --rcnn 101

  Train with both backbones (combo):
    python main.py --task train --dataset_name <dataset_name> --rcnn combo

  Evaluate a model:
    python main.py --task evaluate --dataset_name <dataset_name> --visualize

  Run inference (with threshold and dual model):
    python main.py --task inference --dataset_name <dataset_name> --threshold 0.65 --rcnn combo --visualize

  Run setup (first-time configuration):
    python main.py --task setup

Tasks:
  prepare    Prepare the dataset by splitting into train and test sets.
  train      Train a model on the dataset.
  evaluate   Evaluate the trained model on the test set.
  inference  Run inference on new data using the trained model.
  setup      Run first-time configuration setup.

RCNN Backbone options:
  50      R50 (prefer small particles)
  101     R101 (prefer large particles, default)
  combo   Dual model (universal, merges both)

For more details, see the README or documentation.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="RCNN backbone to use: '50', '101', or 'combo' (both). Default is '101'.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation during training (applies to both images and annotations).",
    )
    parser.add_argument(
        "--pass",
        dest="pass_mode",
        nargs="+",
        choices=["single", "multi"],
        default=["single"],
        help=(
            "Inference pass mode:\n"
            "  'single' : One pass per image (default).\n"
            "  'multi [max_iters]' : Iterative deduplication, repeat until <10% new masks or max_iters (default 10).\n"
            "    Example: --pass multi 5"
        ),
    )

    args = parser.parse_args()

    # Parse pass_mode and max_iters
    if args.pass_mode[0] == "multi":
        pass_mode = "multi"
        try:
            max_iters = int(args.pass_mode[1])
        except (IndexError, ValueError):
            max_iters = 10  # Default if not provided
    else:
        pass_mode = "single"
        max_iters = 1  # Not used in single mode

    if args.task == "setup":
        setup_config()
        return

    local_path = Path.home() / "deepEMIA"
    try:
        subprocess.run(
            [
                "gsutil",
                "-m",
                "cp",
                "-r",
                f"gs://{bucket}/dataset_info.json",
                str(local_path),
            ],
            check=True,
        )
        system_logger.info("Successfully copied dataset_info.json from GCS.")
    except subprocess.CalledProcessError as e:
        system_logger.error(f"Failed to copy dataset_info.json from GCS: {e}")
        raise

    img_dir = local_dataset_path / "DATASET" / args.dataset_name
    output_dir = SPLIT_DIR

    total_start_time = datetime.now()
    download_time_taken = 0
    upload_time_taken = 0

    system_logger.info(f"Running task: {args.task} on dataset: {args.dataset_name}")

    if args.task == "prepare":
        system_logger.info(f"Preparing dataset {args.dataset_name}...")
        task_start_time = datetime.now()
        split_dataset(img_dir, args.dataset_name)
        task_end_time = datetime.now()

    elif args.task == "train":
        # Download data
        system_logger.info(f"Downloading training data for {args.dataset_name}...")
        download_time_taken = download_data_from_bucket()

        # Train
        system_logger.info(
            f"Training model on dataset {args.dataset_name} using '{args.dataset_format}' format and RCNN {args.rcnn}..."
        )
        train_on_dataset(
            args.dataset_name,
            output_dir,
            dataset_format=args.dataset_format,
            rcnn=args.rcnn,
            augment=args.augment,
        )

        # Delete dataset after training
        dataset_path = local_dataset_path / "DATASET" / args.dataset_name
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            system_logger.info(
                f"Deleted training data at {dataset_path} after training."
            )

    elif args.task == "evaluate":
        system_logger.info(
            f"Evaluating model on dataset {args.dataset_name} using '{args.dataset_format}' format..."
        )
        task_start_time = datetime.now()
        evaluate_model(
            args.dataset_name,
            output_dir,
            args.visualize,
            dataset_format=args.dataset_format,
            rcnn=args.rcnn,
        )
        task_end_time = datetime.now()

    elif args.task == "inference":
        system_logger.info(
            f"Running inference on dataset {args.dataset_name} using '{args.dataset_format}' format and RCNN {args.rcnn}..."
        )

        # Remove .png, .csv, .jpg files in the current directory
        for pattern in ("*.png", "*.csv", "*.jpg"):
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                    system_logger.info(f"Removed file: {file_path}")
                except Exception as e:
                    system_logger.warning(f"Could not remove file {file_path}: {e}")

        # Download inference data
        inference_path = local_dataset_path / "DATASET" / "INFERENCE"
        system_logger.info("Downloading inference data...")
        download_time_taken = download_data_from_bucket()

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
            rcnn=args.rcnn,
            pass_mode=pass_mode,
            max_iters=max_iters,  # <-- Pass max_iters to run_inference
        )

        task_end_time = datetime.now()
        inference_time_taken = (task_end_time - task_start_time).total_seconds()
        update_eta_data("inference", inference_time_taken, num_images)

        # Delete inference data after inference
        if inference_path.exists():
            shutil.rmtree(inference_path)
            system_logger.info(
                f"Deleted inference data at {inference_path} after inference."
            )

    total_end_time = datetime.now()
    total_time_taken = (total_end_time - total_start_time).total_seconds()

    if args.upload:
        system_logger.info(
            f"Uploading results for dataset {args.dataset_name} to bucket..."
        )
        upload_time_taken = upload_data_to_bucket()

        # Upload logs directory to the bucket
        logs_dir = LOGS_DIR
        if logs_dir.exists():
            try:
                subprocess.run(
                    ["gsutil", "-m", "cp", "-r", str(logs_dir), f"gs://{bucket}/logs/"],
                    check=True,
                )
                system_logger.info(f"Uploaded logs directory to gs://{bucket}/logs/")
            except subprocess.CalledProcessError as e:
                system_logger.warning(f"Failed to upload logs directory: {e}")

            # # Delete logs directory after upload
            # try:
            #     shutil.rmtree(logs_dir)
            #     system_logger.info(f"Deleted local logs directory: {logs_dir}")
            # except Exception as e:
            #     system_logger.warning(
            #         f"Could not delete local logs directory {logs_dir}: {e}"
            #     )

        # Delete result files after upload
        for pattern in ("*.png", "*.csv"):
            for file_path in Path.home().glob(pattern):
                try:
                    file_path.unlink()
                    system_logger.info(f"Deleted result file: {file_path}")
                except Exception as e:
                    system_logger.warning(
                        f"Could not delete result file {file_path}: {e}"
                    )
        output_dir = Path.home() / "output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
            system_logger.info(f"Deleted output directory: {output_dir}")

    if args.task != "inference":
        update_eta_data(args.task, total_time_taken)

    if args.download:
        update_eta_data("download", download_time_taken)
    if args.upload:
        update_eta_data("upload", upload_time_taken)


if __name__ == "__main__":

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Failed to create log directory: {LOGS_DIR} ({e})")

    print(f"system_logger to: {LOGS_DIR / 'full.log'}")
    main()
