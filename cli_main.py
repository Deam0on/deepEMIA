#!/usr/bin/env python3
"""
Interactive CLI Wizard for deepEMIA

This script provides a user-friendly interface for running the main pipeline
without needing to remember complex command-line arguments.
"""

import os
import subprocess
import sys
from pathlib import Path
import json
import yaml


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    """Print the application header."""
    print("=" * 60)
    print("    deepEMIA - Interactive CLI")
    print("=" * 60)
    print()


def download_dataset_info():
    """
    Download dataset_info.json from GCS before proceeding.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    try:
        # Read config to get bucket name
        config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"
        if not config_path.exists():
            print("Warning: Config file not found. Please run setup first.")
            return False

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        bucket = config.get("bucket")
        if not bucket:
            print("Warning: Bucket not configured. Please run setup first.")
            return False

        local_path = Path.home() / "deepEMIA"
        local_path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading dataset_info.json from gs://{bucket}...")

        result = subprocess.run(
            [
                "gsutil",
                "-m",
                "cp",
                "-r",
                f"gs://{bucket}/dataset_info.json",
                str(local_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            print("Successfully downloaded dataset_info.json")
            return True
        else:
            print(f"Failed to download dataset_info.json: {result.stderr}")
            print("You can still enter dataset names manually.")
            return False

    except Exception as e:
        print(f"Error downloading dataset_info.json: {e}")
        print("You can still enter dataset names manually.")
        return False


def get_user_choice(prompt, choices, default=None):
    """
    Get user choice from a list of options.

    Args:
        prompt (str): The prompt to display to the user.
        choices (list): List of available choices.
        default (str, optional): Default choice if user presses enter.

    Returns:
        str: The selected choice.
    """
    while True:
        if prompt:
            print(prompt)
        for i, choice in enumerate(choices, 1):
            marker = " (default)" if default and choice == default else ""
            print(f"  {i}. {choice}{marker}")

        if default:
            choice_input = input(
                f"\nEnter choice (1-{len(choices)}) [default: {choices.index(default) + 1}]: "
            ).strip()
            if not choice_input:
                return default
        else:
            choice_input = input(f"\nEnter choice (1-{len(choices)}): ").strip()

        try:
            choice_idx = int(choice_input) - 1
            if 0 <= choice_idx < len(choices):
                return choices[choice_idx]
            else:
                print(f"Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print("Please enter a valid number")
        print()


def get_yes_no(prompt, default=None):
    """
    Get yes/no input from user.

    Args:
        prompt (str): The prompt to display to the user.
        default (bool, optional): Default value if user presses enter.

    Returns:
        bool: True for yes, False for no.
    """
    while True:
        if default is not None:
            suffix = " [Y/n]" if default else " [y/N]"
            response = input(f"{prompt}{suffix}: ").strip().lower()
            if not response:
                return default
        else:
            response = input(f"{prompt} (y/n): ").strip().lower()

        if response in ["y", "yes", "true", "1"]:
            return True
        elif response in ["n", "no", "false", "0"]:
            return False
        else:
            print("Please enter 'y' or 'n'")


def get_string_input(prompt, default=None, required=True):
    """
    Get string input from user.

    Args:
        prompt (str): The prompt to display to the user.
        default (str, optional): Default value if user presses enter.
        required (bool): Whether the input is required.

    Returns:
        str: The user's input.
    """
    while True:
        if default:
            response = input(f"{prompt} [default: {default}]: ").strip()
            if not response:
                return default
        else:
            response = input(f"{prompt}: ").strip()

        if response or not required:
            return response
        else:
            print("This field is required.")


def get_float_input(prompt, default=None, min_val=None, max_val=None):
    """
    Get float input from user.

    Args:
        prompt (str): The prompt to display to the user.
        default (float, optional): Default value if user presses enter.
        min_val (float, optional): Minimum allowed value.
        max_val (float, optional): Maximum allowed value.

    Returns:
        float: The user's input as a float.
    """
    while True:
        if default is not None:
            response = input(f"{prompt} [default: {default}]: ").strip()
            if not response:
                return default
        else:
            response = input(f"{prompt}: ").strip()

        try:
            value = float(response)
            if min_val is not None and value < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number")


def get_int_input(prompt, default=None, min_val=None, max_val=None):
    """
    Get integer input from user.

    Args:
        prompt (str): The prompt to display to the user.
        default (int, optional): Default value if user presses enter.
        min_val (int, optional): Minimum allowed value.
        max_val (int, optional): Maximum allowed value.

    Returns:
        int: The user's input as an integer.
    """
    while True:
        if default is not None:
            response = input(f"{prompt} [default: {default}]: ").strip()
            if not response:
                return default
        else:
            response = input(f"{prompt}: ").strip()

        try:
            value = int(response)
            if min_val is not None and value < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid integer")


def setup_task():
    """
    Handle setup task.

    Returns:
        list or None: Command arguments for setup task, or None if cancelled.
    """
    print("\nSETUP TASK")
    print("This will configure your project for first-time use.")
    print(
        "You'll be prompted to enter your Google Cloud Storage bucket and other settings."
    )

    if get_yes_no("\nProceed with setup?", default=True):
        return ["--task", "setup"]
    return None


def get_available_datasets():
    """
    Get list of available datasets from dataset_info.json.

    Returns:
        list: List of available dataset names.
    """
    try:
        # Read config to get path to dataset_info.json
        config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Get dataset_info.json path
        category_json_path = Path(config["paths"]["category_json"]).expanduser()

        # Check if file exists, if not try to download
        if not category_json_path.exists():
            print(f"Warning: Dataset info file not found at {category_json_path}")
            if download_dataset_info():
                # Try again after download
                if not category_json_path.exists():
                    print("Error: Dataset info file still not found after download")
                    return []
            else:
                return []

        # Read dataset info
        with open(category_json_path, "r") as f:
            dataset_info = json.load(f)

        # Extract dataset names
        datasets = list(dataset_info.keys())
        return datasets
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load dataset list: {e}")
        return []


def get_dataset_selection_with_retry(prompt_text="Select dataset", max_retries=3):
    """
    Get dataset selection with retry logic for error recovery.

    Args:
        prompt_text (str): Prompt text for dataset selection.
        max_retries (int): Maximum number of retry attempts.

    Returns:
        str: Selected dataset name.
    """
    for attempt in range(max_retries):
        try:
            return get_dataset_selection(prompt_text)
        except Exception as e:
            print(f"Error loading datasets (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                retry = get_yes_no("Retry loading datasets?", default=True)
                if not retry:
                    break
            else:
                print("Max retries exceeded.")

    print("Falling back to manual dataset name entry.")
    return get_string_input("Enter dataset name manually")


def get_dataset_selection(prompt_text="Select dataset"):
    """
    Get dataset selection from available datasets.

    Args:
        prompt_text (str): Prompt text for dataset selection.

    Returns:
        str: Selected dataset name.
    """
    datasets = get_available_datasets()

    if not datasets:
        print("No datasets found or dataset_info.json not available.")
        print("Tip: Make sure you have run setup and have network access to GCS.")
        return get_string_input("Enter dataset name manually")

    return get_user_choice(
        f"\n{prompt_text}:", datasets, default=datasets[0] if datasets else None
    )


def prepare_task():
    """
    Handle prepare task.

    Returns:
        list or None: Command arguments for prepare task, or None if cancelled.
    """
    print("\nPREPARE TASK")
    print("This will split your dataset into training and testing sets.")
    print(
        "Typically uses 80% for training and 20% for testing with stratified splitting."
    )

    dataset_name = get_dataset_selection_with_retry("Select dataset to prepare")

    # Dataset format
    dataset_format = get_user_choice(
        "\nDataset annotation format:",
        ["json (one JSON per image - most common)", "coco (standard COCO format)"],
        default="json (one JSON per image - most common)",
    )
    dataset_format_value = dataset_format.split()[0]

    args = [
        "--task",
        "prepare",
        "--dataset_name",
        dataset_name,
        "--dataset_format",
        dataset_format_value,
    ]

    # Download/Upload options
    print("\nCloud Storage Options:")
    download = get_yes_no("Download dataset from Google Cloud Storage?", default=True)
    upload = get_yes_no(
        "Upload prepared dataset splits to Google Cloud Storage?", default=True
    )

    if download:
        args.append("--download")
    if upload:
        args.append("--upload")

    return args


def train_task():
    """
    Handle train task.

    Returns:
        list or None: Command arguments for train task, or None if cancelled.
    """
    print("\nTRAIN TASK")
    print("This will train instance segmentation models on your dataset.")
    print("Training uses Detectron2 with ResNet backbones and can take several hours.")

    dataset_name = get_dataset_selection_with_retry("Select dataset to train on")

    # Show current hyperparameters for this dataset
    try:
        # Import here to avoid issues if modules aren't available
        sys.path.append(str(Path(__file__).parent / "src"))
        from src.functions.train_model import get_hyperparameter_info

        print(f"\nCurrent hyperparameters for '{dataset_name}':")
        print("-" * 50)
        info = get_hyperparameter_info(dataset_name=dataset_name)
        print(info)
        print("-" * 50)
    except Exception as e:
        print(f"\nNote: Could not load hyperparameter info: {e}")

    # RCNN backbone
    print("\nRCNN Backbone Architecture:")
    print("   R50: Faster training/inference, good for small particles")
    print("   R101: Slower but more accurate, good for large/complex particles")
    print("   Combo: Trains both models - best overall performance")
    rcnn = get_user_choice(
        "Select RCNN backbone:",
        [
            "50 (R50 - faster, good for small particles)",
            "101 (R101 - slower, good for large particles)",
            "combo (both R50 and R101 - recommended)",
        ],
        default="101 (R101 - slower, good for large particles)",
    )
    rcnn_value = rcnn.split()[0]  # Extract just the number/combo

    # Dataset format
    dataset_format = get_user_choice(
        "\nDataset annotation format:",
        ["json (one JSON per image)", "coco (standard COCO format)"],
        default="json (one JSON per image)",
    )
    dataset_format_value = dataset_format.split()[0]

    # Advanced training options
    print("\nAdvanced Training Options:")

    # Augmentation
    print(
        "   Data Augmentation: Improves model robustness with rotation, flip, brightness changes"
    )
    augment = get_yes_no("Enable data augmentation during training?", default=False)

    # Hyperparameter optimization
    print(
        "   Hyperparameter Optimization: Uses Optuna to automatically find best parameters"
    )
    print(
        "   Note: This will create/update dataset-specific hyperparameters (best_"
        + dataset_name
        + ")"
    )
    optimize = get_yes_no(
        "Run hyperparameter optimization (takes longer but improves results)?",
        default=False,
    )
    n_trials = None
    if optimize:
        n_trials = get_int_input(
            "Number of optimization trials (more trials = better results but longer training)",
            default=10,
            min_val=1,
            max_val=100,
        )

    args = [
        "--task",
        "train",
        "--dataset_name",
        dataset_name,
        "--rcnn",
        rcnn_value,
        "--dataset_format",
        dataset_format_value,
    ]

    if augment:
        args.append("--augment")

    if optimize:
        args.extend(["--optimize", "--n-trials", str(n_trials)])

    # Download/Upload
    print("\nCloud Storage Options:")
    download = get_yes_no(
        "Download training data from Google Cloud Storage?", default=True
    )
    upload = get_yes_no(
        "Upload trained models and results to Google Cloud Storage?", default=True
    )

    if download:
        args.append("--download")
    if upload:
        args.append("--upload")

    return args


def evaluate_task():
    """
    Handle evaluate task.

    Returns:
        list or None: Command arguments for evaluate task, or None if cancelled.
    """
    print("\nEVALUATE TASK")
    print("This will evaluate your trained model on the test set using COCO metrics.")
    print("You'll get precision, recall, mAP, and other performance metrics.")

    dataset_name = get_dataset_selection_with_retry("Select dataset to evaluate")

    # RCNN backbone
    rcnn = get_user_choice(
        "\nSelect RCNN backbone to evaluate:",
        ["50 (R50 model)", "101 (R101 model)", "combo (both models)"],
        default="101 (R101 model)",
    )
    rcnn_value = rcnn.split()[0]

    # Dataset format
    dataset_format = get_user_choice(
        "\nDataset annotation format:",
        ["json (one JSON per image)", "coco (standard COCO format)"],
        default="json (one JSON per image)",
    )
    dataset_format_value = dataset_format.split()[0]

    # Visualization
    print("\nVisualization Options:")
    visualize = get_yes_no(
        "Generate visualization outputs showing predictions vs ground truth?",
        default=True,
    )

    args = [
        "--task",
        "evaluate",
        "--dataset_name",
        dataset_name,
        "--rcnn",
        rcnn_value,
        "--dataset_format",
        dataset_format_value,
    ]

    if visualize:
        args.append("--visualize")

    # Download/Upload
    print("\nCloud Storage Options:")
    download = get_yes_no(
        "Download evaluation data from Google Cloud Storage?", default=True
    )
    upload = get_yes_no(
        "Upload evaluation results to Google Cloud Storage?", default=True
    )

    if download:
        args.append("--download")
    if upload:
        args.append("--upload")

    return args


def inference_task():
    """
    Handle inference task.

    Returns:
        list or None: Command arguments for inference task, or None if cancelled.
    """
    print("\nINFERENCE TASK")
    print("This will run inference on new images using your trained model.")
    print("The system will detect, measure, and analyze particles in your images.")
    print(
        "\nNOTE: System now auto-detects available models and always uses class-specific inference"
    )
    print("      Dataset-specific settings (scale bar ROI, spatial constraints) are applied automatically")

    dataset_name = get_dataset_selection_with_retry("Select dataset for inference")

    # Detection threshold
    print("\nDetection Threshold Configuration:")
    print("   Higher threshold = fewer, more confident detections")
    print("   Lower threshold = more detections, including uncertain ones")
    threshold = get_float_input(
        "Detection confidence threshold (0.0-1.0)",
        default=0.65,
        min_val=0.0,
        max_val=1.0,
    )

    # Dataset format
    dataset_format = get_user_choice(
        "\nDataset annotation format:",
        ["json (one JSON per image - most common)", "coco (standard COCO format)"],
        default="json (one JSON per image - most common)",
    )
    dataset_format_value = dataset_format.split()[0]

    # Visualization options
    print("\nVisualization Options:")
    visualize = get_yes_no(
        "Generate visualization overlays showing detections?", default=True
    )
    draw_id = False
    if visualize:
        draw_id = get_yes_no(
            "Draw instance ID numbers on visualizations?", default=False
        )

    # Logging verbosity
    print("\nLogging Verbosity:")
    print("   INFO: Standard output (recommended)")
    print("   DEBUG: Detailed output for troubleshooting")
    verbosity = get_user_choice(
        "Select logging verbosity:",
        ["INFO (standard)", "DEBUG (detailed)"],
        default="INFO (standard)",
    )
    verbosity_level = verbosity.split()[0].lower()

    args = [
        "--task",
        "inference",
        "--dataset_name",
        dataset_name,
        "--threshold",
        str(threshold),
        "--dataset_format",
        dataset_format_value,
        "--verbosity",
        verbosity_level,
    ]

    if visualize:
        args.append("--visualize")
    if draw_id:
        args.append("--id")

    # Download/Upload
    print("\nCloud Storage Options:")
    download = get_yes_no(
        "Download inference data from Google Cloud Storage?", default=True
    )
    upload = get_yes_no("Upload results to Google Cloud Storage?", default=True)

    if download:
        args.append("--download")
    if upload:
        args.append("--upload")

    return args


def execute_command(args):
    """
    Execute the command in the current terminal.

    Args:
        args (list): Command arguments to pass to main.py.

    Returns:
        bool: True if command executed successfully, False otherwise.
    """
    command = f"python main.py {' '.join(args)}"

    print(f"\nCommand to execute:")
    print(f"{command}")
    print()

    if not get_yes_no("Execute this command?", default=True):
        print("Task cancelled.")
        return False

    print("\nExecuting in current terminal...")
    try:
        script_dir = Path(__file__).parent
        main_py = script_dir / "main.py"

        subprocess.run([sys.executable, str(main_py)] + args, check=True)
        print("\nTask completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTask failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nTask interrupted by user")
        return False


def main():
    """Main function."""
    print("Starting deepEMIA Interactive CLI...")
    print("This wizard will guide you through all available operations step-by-step.")
    print("For direct command-line usage: python main.py --help")

    while True:
        clear_screen()
        print_header()

        print("Welcome to deepEMIA - Deep Learning Image Analysis Tool")
        print(
            "   A complete pipeline for scientific image analysis with instance segmentation"
        )
        print()

        # Task selection
        task = get_user_choice(
            "Select a task to perform:",
            [
                "setup - First-time configuration (bucket, hyperparameters, scale bars)",
                "prepare - Split dataset into train/test sets",
                "train - Train instance segmentation models",
                "evaluate - Evaluate trained models with COCO metrics",
                "inference - Run inference on new images with measurements",
                "exit - Exit the wizard",
            ],
        )

        if task.startswith("exit"):
            print("\nThank you for using deepEMIA! Goodbye!")
            break

        # Get task-specific arguments
        task_name = task.split()[0]

        # For non-setup tasks, ensure we have dataset info
        if task_name != "setup":
            # Try to download dataset_info.json if needed
            config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"
            dataset_info_path = Path.home() / "deepEMIA" / "dataset_info.json"

            if config_path.exists() and not dataset_info_path.exists():
                print("\nChecking for dataset information...")
                download_dataset_info()
            elif not config_path.exists():
                print("Warning: Configuration not found. Please run setup first.")
                print("   Setup will configure your GCS bucket and other settings.")
                continue

        # Show task-specific information
        if task_name == "setup":
            print("\nSetup will configure:")
            print("   • Google Cloud Storage bucket")
            print("   • Scale bar detection settings")
            print("   • Model hyperparameters")
            print("   • Other project settings")
            args = setup_task()
        elif task_name == "prepare":
            print("\nPrepare will:")
            print("   • Split your dataset into train/test sets")
            print("   • Register datasets with Detectron2")
            print("   • Validate annotations")
            args = prepare_task()
        elif task_name == "train":
            print("\nTraining will:")
            print("   • Train instance segmentation models")
            print("   • Support R50, R101, or combo backbones")
            print("   • Optional hyperparameter optimization")
            print("   • Optional data augmentation")
            args = train_task()
        elif task_name == "evaluate":
            print("\nEvaluation will:")
            print("   • Test model performance on test set")
            print("   • Generate COCO metrics (mAP, precision, recall)")
            print("   • Optional visualization outputs")
            args = evaluate_task()
        elif task_name == "inference":
            print("\nInference will:")
            print("   • Detect and segment particles in images")
            print("   • Measure geometric properties")
            print("   • Generate CSV results and visualizations")
            print("   • Support single or multi-pass modes")
            args = inference_task()

        if args is None:
            print("\nTask cancelled.")
            input("Press Enter to return to main menu...")
            continue

        # Execute the command
        success = execute_command(args)

        if success:
            print("\nTask completed successfully!")
        else:
            print("\nTask failed. Check the output above for details.")
            print("Tip: Check ~/logs/ for detailed error information.")

        input("\nPress Enter to return to main menu...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye! (Interrupted by user)")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check ~/logs/ for detailed error information.")
        input("Press Enter to exit...")
