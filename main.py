"""
deepEMIA Project - Main Pipeline Script

This script provides a command-line interface for running dataset preparation, model training,
evaluation, and inference tasks. It integrates with Google Cloud Storage and tracks progress/ETA.
system_logger is configured to print simplified logs to the terminal and full logs to logs/full.log.
"""

import argparse
import os
import json
import shutil
import random
import subprocess
from datetime import datetime
from pathlib import Path

from src.data.datasets import split_dataset
from src.functions.evaluate_model import evaluate_model
from src.functions.inference import run_inference
from src.functions.train_model import train_on_dataset
from src.utils.config import get_config
from src.utils.eta_utils import update_eta_data
from src.utils.gcs_utils import download_data_from_bucket, upload_data_to_bucket
from src.utils.logger_utils import system_logger
from src.utils.gpu_check import check_gpu_availability, log_device_info

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
    print("=== deepEMIA Project Setup ===")
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

    print("\nConfigure scalebar thresholds (press Enter to use defaults):")
    intensity = input("  intensity threshold [default 150]: ").strip() or "150"
    proximity = input("  proximity threshold [default 50]: ").strip() or "50"

    print("\nConfigure measurement settings:")
    measure_contrast = (
        input("  measure_contrast_distribution [default false] (true/false): ").strip()
        or "false"
    )
    measure_contrast = measure_contrast.lower() == "true"

    print("\nConfigure inference settings:")
    class_specific_default = (
        input("  use_class_specific_inference [default true] (true/false): ").strip()
        or "true"
    )
    use_class_specific_default = class_specific_default.lower() == "true"

    print("\nConfigure RCNN hyperparameters (press Enter to use defaults):")
    print("  R50 settings:")
    r50_base_lr = input("    base_lr [default 0.00025]: ").strip() or "0.00025"
    r50_ims_per_batch = input("    ims_per_batch [default 2]: ").strip() or "2"
    r50_warmup_iters = input("    warmup_iters [default 1000]: ").strip() or "1000"
    r50_gamma = input("    gamma [default 0.1]: ").strip() or "0.1"
    r50_batch_size_per_image = (
        input("    batch_size_per_image [default 64]: ").strip() or "64"
    )

    print("  R101 settings:")
    r101_base_lr = input("    base_lr [default 0.00025]: ").strip() or "0.00025"
    r101_ims_per_batch = input("    ims_per_batch [default 2]: ").strip() or "2"
    r101_warmup_iters = input("    warmup_iters [default 1000]: ").strip() or "1000"
    r101_gamma = input("    gamma [default 0.1]: ").strip() or "0.1"
    r101_batch_size_per_image = (
        input("    batch_size_per_image [default 64]: ").strip() or "64"
    )

    config = {
        "bucket": bucket,
        "paths": {
            "main_script": "~/deepEMIA/main.py",
            "split_dir": "~/split_dir",
            "category_json": "~/deepEMIA/dataset_info.json",
            "eta_file": "~/deepEMIA/config/eta_data.json",
            "logs_dir": "~/logs",
            "output_dir": "~/deepEMIA/output",
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
        "scalebar_thresholds": {
            "intensity": int(intensity),
            "proximity": int(proximity),
        },
        "measure_contrast_distribution": measure_contrast,
        "inference_settings": {
            "use_class_specific_inference": use_class_specific_default,
            "class_specific_settings": {
                "class_0": {
                    "confidence_threshold": 0.5,
                    "iou_threshold": 0.7,
                    "min_size": 25,
                },
                "class_1": {
                    "confidence_threshold": 0.3,
                    "iou_threshold": 0.5,
                    "min_size": 5,
                    "use_multiscale": True,
                },
            },
        },
        "rcnn_hyperparameters": {
            "default": {
                "R50": {
                    "base_lr": float(r50_base_lr),
                    "ims_per_batch": int(r50_ims_per_batch),
                    "warmup_iters": int(r50_warmup_iters),
                    "gamma": float(r50_gamma),
                    "batch_size_per_image": int(r50_batch_size_per_image),
                },
                "R101": {
                    "base_lr": float(r101_base_lr),
                    "ims_per_batch": int(r101_ims_per_batch),
                    "warmup_iters": int(r101_warmup_iters),
                    "gamma": float(r101_gamma),
                    "batch_size_per_image": int(r101_batch_size_per_image),
                },
            },
            "best": {
                "R50": {},
                "R101": {},
            },
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Configuration saved to {config_path}.")


def main():
    """
    Main function that parses command line arguments and executes the requested task.
    """
    parser = argparse.ArgumentParser(
        description="deepEMIA - Deep Learning Computer Vision Pipeline for Scientific Image Analysis.\n"
        "This tool provides dataset preparation, model training, evaluation, and inference capabilities.\n\n"
        "For an easier, interactive experience, use: python cli_main.py\n"
        "The CLI wizard guides you through all options step-by-step.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core task argument
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["setup", "prepare", "train", "evaluate", "inference"],
        help="Task to perform",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset to use",
    )

    parser.add_argument(
        "--dataset_format",
        type=str,
        default="json",
        choices=["json", "coco"],
        help="Dataset annotation format",
    )

    parser.add_argument(
        "--config_variant",
        type=str,
        default="default",
        help="Configuration variant for the dataset (e.g., 'default', 'class_0_max', 'balanced')",
    )

    # Model arguments
    parser.add_argument(
        "--rcnn",
        type=str,
        default="101",
        choices=["50", "101", "combo"],
        help="RCNN backbone architecture (50, 101, or combo)",
    )

    # Training arguments
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation during training",
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable hyperparameter optimization with Optuna",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of Optuna trials for optimization",
    )

    # Inference arguments
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Confidence threshold for inference",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization overlays",
    )

    parser.add_argument(
        "--id",
        action="store_true",
        dest="draw_id",
        help="Draw instance IDs on visualizations",
    )

    parser.add_argument(
        "--draw_scalebar",
        action="store_true",
        help="Draw scale bar ROI and detection on images (for debugging)",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results",
    )

    # Cloud storage arguments
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download data from GCS before processing",
    )

    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload results to GCS after processing",
    )

    # Logging arguments
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        choices=["INFO", "DEBUG"],
        help="Logging verbosity level",
    )

    args = parser.parse_args()

    # Set logging verbosity
    if args.verbosity == "DEBUG":
        from src.utils.logger_utils import set_console_log_level
        import logging
        set_console_log_level(logging.DEBUG)

    # Load config with dataset and variant support
    if args.task != "setup":
        config = get_config(dataset_name=args.dataset_name, variant=args.config_variant)
    
    # Check GPU before any heavy operations
    system_logger.info("Checking GPU availability...")
    check_gpu_availability(require_gpu=False, interactive=True)
    log_device_info()

    # Execute task
    try:
        if args.task == "setup":
            setup_config()

        elif args.task == "prepare":
            if not args.dataset_name:
                system_logger.error("--dataset_name required for prepare task")
                return
            
            if args.download:
                download_data_from_bucket()
            
            img_dir = Path.home() / "DATASET" / args.dataset_name
            split_dataset(str(img_dir), args.dataset_name, test_size=0.2)
            
            if args.upload:
                upload_data_to_bucket()

        elif args.task == "train":
            if not args.dataset_name:
                system_logger.error("--dataset_name required for train task")
                return
            
            if args.download:
                download_data_from_bucket()
            
            train_on_dataset(
                args.dataset_name,
                args.output_dir or str(SPLIT_DIR),
                dataset_format=args.dataset_format,
                rcnn=args.rcnn,
                augment=args.augment,
                optimize=args.optimize,
                n_trials=args.n_trials,
            )
            
            if args.upload:
                upload_data_to_bucket()

        elif args.task == "evaluate":
            if not args.dataset_name:
                system_logger.error("--dataset_name required for evaluate task")
                return
            
            if args.download:
                download_data_from_bucket()
            
            evaluate_model(
                args.dataset_name,
                args.output_dir or str(SPLIT_DIR),
                visualize=args.visualize,
                dataset_format=args.dataset_format,
                rcnn=int(args.rcnn) if args.rcnn != "combo" else 101,
            )
            
            if args.upload:
                upload_data_to_bucket()

        elif args.task == "inference":
            if not args.dataset_name:
                system_logger.error("--dataset_name required for inference task")
                return
            
            if args.download:
                download_data_from_bucket()
            
            run_inference(
                args.dataset_name,
                args.output_dir or str(SPLIT_DIR),
                visualize=args.visualize,
                threshold=args.threshold,
                draw_id=args.draw_id,
                dataset_format=args.dataset_format,
                draw_scalebar=args.draw_scalebar,
            )
            
            if args.upload:
                upload_data_to_bucket()

    except Exception as e:
        system_logger.error(f"Task failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Failed to create log directory: {LOGS_DIR} ({e})")

    print(f"system_logger to: {LOGS_DIR / 'full.log'}")
    main()
