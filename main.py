"""
deepEMIA Project - Main Pipeline Script

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
        epilog="""
QUICK START EXAMPLES:

Setup (First Time):
  python main.py --task setup
  
Prepare Dataset:
  python main.py --task prepare --dataset_name polyhipes

Training:
  # Basic training
  python main.py --task train --dataset_name polyhipes --rcnn 101
  
  # Advanced training with optimization
  python main.py --task train --dataset_name polyhipes --rcnn combo --optimize --n-trials 20 --augment

Evaluation:
  python main.py --task evaluate --dataset_name polyhipes --visualize --rcnn combo

Inference:
  # Run inference with automatic iteration control (configured in config.yaml)
  python main.py --task inference --dataset_name polyhipes --threshold 0.7 --visualize
  
  # Inference with instance IDs displayed
  python main.py --task inference --dataset_name polyhipes --threshold 0.65 --visualize --id

TASK DESCRIPTIONS:

setup      First-time configuration (bucket, scale bar settings, hyperparameters)
prepare    Split dataset into train/test sets and register with Detectron2
train      Train instance segmentation models (R50, R101, or both)
evaluate   Evaluate trained models on test set with COCO metrics
inference  Run inference on new images with measurements and analysis

RCNN BACKBONE OPTIONS:

50         Fast R-CNN with ResNet-50 backbone (good for small particles)
101        Fast R-CNN with ResNet-101 backbone (good for large particles, default)
combo      Dual model approach - uses both R50 and R101 for universal detection

ADVANCED FEATURES:

• Hyperparameter Optimization: Use --optimize --n-trials N for automated tuning
• Data Augmentation: Use --augment for enhanced training robustness  
• Iteration Control: Automatic via config.yaml iterative_stopping settings
• Universal Class Processing: Inference always uses class-specific processing with size heuristics
• Visualization: Use --visualize to save prediction overlays
• Instance IDs: Use --id to draw instance identifiers on visualizations

CLOUD INTEGRATION:

• Automatic GCS sync with --download/--upload flags
• Progress tracking with ETA estimation
• Comprehensive logging to ~/logs/

For detailed documentation, see README.md
For guided interactive mode: python cli_main.py
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["prepare", "train", "evaluate", "inference", "setup"],
        help="Task to perform:\n"
        "• 'prepare': Split dataset into train/test sets and register with Detectron2\n"
        "• 'train': Train instance segmentation models (supports R50, R101, or combo)\n"
        "• 'evaluate': Evaluate trained models on test set with COCO metrics\n"
        "• 'inference': Run inference on new images with advanced measurements\n"
        "• 'setup': Interactive first-time configuration (bucket, hyperparameters, etc.)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=False,
        help="Dataset name to use (e.g., 'polyhipes', 'crystals'). Must exist in dataset_info.json. Not required for setup task.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Detection confidence threshold for inference (0.0-1.0). Higher = fewer, more confident detections. [default: 0.65]",
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="json",
        choices=["json", "coco"],
        help="Dataset annotation format: 'json' (one JSON per image) or 'coco' (standard COCO format). [default: json]",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Generate and save visualization overlays during evaluation/inference. Shows predictions on original images.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=True,
        help="Download required data from Google Cloud Storage before task execution. [default: True]",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        default=True,
        help="Upload results and logs to Google Cloud Storage after task completion. [default: True]",
    )
    parser.add_argument(
        "--id",
        dest="draw_id",
        action="store_true",
        help="Draw instance ID numbers on inference visualization overlays for easier tracking.",
    )
    parser.set_defaults(draw_id=False)
    parser.add_argument(
        "--rcnn",
        type=str,
        default="101",
        choices=["50", "101", "combo"],
        help="RCNN backbone architecture for train/evaluate tasks:\n"
        "• '50': ResNet-50 (faster, good for small particles)\n"
        "• '101': ResNet-101 (slower, good for large particles)\n"
        "• 'combo': Both models [default: 101]\n"
        "Note: Inference task auto-detects available models",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation during training (rotation, flip, brightness). Improves model robustness.",
    )
    # --- Optuna HPO flags ---
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization during training. Automatically finds best parameters.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of Optuna optimization trials to run. More trials = better optimization but longer time. [default: 10]",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Console logging verbosity level. File logs always include DEBUG. [default: info]",
    )
    parser.add_argument(
        "--no-gpu-check",
        action="store_true",
        help="Skip GPU availability check (for automated/non-interactive execution).",
    )
    parser.add_argument(
        "--draw-scalebar",
        action="store_true",
        default=False,
        help="Draw scale bar ROI and detection results on output images for debugging. Shows the ROI box and detected scale bar line."
    )
    parser.add_argument(
        "--config_variant",
        type=str,
        default="default",
        help="Configuration variant for the dataset (e.g., 'default', 'class0_maximized', 'balanced')"
    )
    
    args = parser.parse_args()

    # Set console logging level based on verbosity argument
    from src.utils.logger_utils import set_console_log_level
    import logging
    verbosity_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    set_console_log_level(verbosity_map.get(args.verbosity.lower(), logging.INFO))

    # === GPU AVAILABILITY CHECK ===
    # Check GPU before any heavy operations (skip for setup task)
    if args.task != "setup" and not args.no_gpu_check:
        system_logger.info("Checking GPU availability...")
        log_device_info()
        
        # Determine if this task requires GPU
        gpu_intensive_tasks = ['train', 'inference', 'evaluate']
        requires_gpu = args.task in gpu_intensive_tasks
        
        if not check_gpu_availability(require_gpu=requires_gpu, interactive=True):
            system_logger.error("Execution aborted due to GPU unavailability")
            import sys
            sys.exit(1)
    
    # === END GPU CHECK ===

    # Validate arguments
    if args.task != "setup" and not args.dataset_name:
        parser.error(f"--dataset_name is required for task '{args.task}'")

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

        # Train or optimize
        system_logger.info(
            f"Training model on dataset {args.dataset_name} using '{args.dataset_format}' format and RCNN {args.rcnn}..."
        )
        train_on_dataset(
            args.dataset_name,
            output_dir,
            dataset_format=args.dataset_format,
            rcnn=args.rcnn,
            augment=args.augment,
            optimize=args.optimize,
            n_trials=args.n_trials,
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
            f"Running inference on dataset {args.dataset_name} using '{args.dataset_format}' format with auto-detected models..."
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
            draw_scalebar=args.draw_scalebar,
        )

        task_end_time = datetime.now()
        inference_time_taken = (task_end_time - task_start_time).total_seconds()
        update_eta_data("inference", inference_time_taken, num_images)

        # UPDATED: Use dedicated inference upload function
        if args.upload:
            system_logger.info("Uploading inference results to GCP...")

            # Import the new function
            from src.utils.gcs_utils import upload_inference_results

            # Determine model info for remote path
            # Model info reflects automatic iteration control
            model_info = "auto_models_adaptive"

            try:
                upload_time_taken = upload_inference_results(
                    dataset_name=args.dataset_name,
                    model_info=model_info,
                    output_dir=output_dir,
                    current_dir=Path.cwd(),
                )

                if upload_time_taken > 0:
                    system_logger.info(
                        f"Inference results uploaded successfully in {upload_time_taken:.2f} seconds"
                    )
                else:
                    system_logger.warning("No files were uploaded")

            except Exception as e:
                system_logger.error(f"Failed to upload inference results: {e}")

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
