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
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header."""
    print("=" * 60)
    print("    deepEMIA - Interactive CLI")
    print("=" * 60)
    print()

def get_user_choice(prompt, choices, default=None):
    """Get user choice from a list of options."""
    while True:
        print(prompt)
        for i, choice in enumerate(choices, 1):
            marker = " (default)" if default and choice == default else ""
            print(f"  {i}. {choice}{marker}")
        
        if default:
            choice_input = input(f"\nEnter choice (1-{len(choices)}) [default: {choices.index(default) + 1}]: ").strip()
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
    """Get yes/no input from user."""
    while True:
        if default is not None:
            suffix = " [Y/n]" if default else " [y/N]"
            response = input(f"{prompt}{suffix}: ").strip().lower()
            if not response:
                return default
        else:
            response = input(f"{prompt} (y/n): ").strip().lower()
        
        if response in ['y', 'yes', 'true', '1']:
            return True
        elif response in ['n', 'no', 'false', '0']:
            return False
        else:
            print("Please enter 'y' or 'n'")

def get_string_input(prompt, default=None, required=True):
    """Get string input from user."""
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
    """Get float input from user."""
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
    """Get integer input from user."""
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
    """Handle setup task."""
    print("\n🔧 SETUP TASK")
    print("This will configure your project for first-time use.")
    print("You'll be prompted to enter your Google Cloud Storage bucket and other settings.")
    
    if get_yes_no("\nProceed with setup?", default=True):
        return ["--task", "setup", "--dataset_name", "dummy"]
    return None

def get_available_datasets():
    """Get list of available datasets from dataset_info.json."""
    try:
        # Read config to get path to dataset_info.json
        config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Get dataset_info.json path
        category_json_path = Path(config["paths"]["category_json"]).expanduser()
        
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
    """Get dataset selection with retry logic for error recovery."""
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
    """Get dataset selection from available datasets."""
    datasets = get_available_datasets()
    
    if not datasets:
        print("No datasets found. Please ensure dataset_info.json is properly configured.")
        return get_string_input("Enter dataset name manually")
    
    return get_user_choice(
        f"\n{prompt_text}:",
        datasets,
        default=datasets[0] if datasets else None
    )

def prepare_task():
    """Handle prepare task."""
    print("\n📋 PREPARE TASK")
    print("This will split your dataset into training and testing sets.")
    
    dataset_name = get_dataset_selection_with_retry("Select dataset to prepare")
    
    args = ["--task", "prepare", "--dataset_name", dataset_name]
    
    # Download/Upload options
    download = get_yes_no("Download data from Google Cloud Storage?", default=True)
    upload = get_yes_no("Upload results to Google Cloud Storage?", default=True)
    
    if download:
        args.append("--download")
    if upload:
        args.append("--upload")
    
    return args

def train_task():
    """Handle train task."""
    print("\n🎯 TRAIN TASK")
    print("This will train a model on your dataset.")
    
    dataset_name = get_dataset_selection_with_retry("Select dataset to train on")
    
    # RCNN backbone
    rcnn = get_user_choice(
        "\nSelect RCNN backbone:",
        ["50 (R50 - better for small particles)", 
         "101 (R101 - better for large particles)", 
         "combo (both R50 and R101)"],
        default="101 (R101 - better for large particles)"
    )
    rcnn_value = rcnn.split()[0]  # Extract just the number/combo
    
    # Dataset format
    dataset_format = get_user_choice(
        "\nDataset format:",
        ["json", "coco"],
        default="json"
    )
    
    # Augmentation
    augment = get_yes_no("Enable data augmentation?", default=False)
    
    # Hyperparameter optimization
    optimize = get_yes_no("Run hyperparameter optimization (Optuna)?", default=False)
    n_trials = None
    if optimize:
        n_trials = get_int_input("Number of optimization trials", default=10, min_val=1)
    
    args = [
        "--task", "train",
        "--dataset_name", dataset_name,
        "--rcnn", rcnn_value,
        "--dataset_format", dataset_format
    ]
    
    if augment:
        args.append("--augment")
    
    if optimize:
        args.extend(["--optimize", "--n-trials", str(n_trials)])
    
    # Download/Upload
    download = get_yes_no("Download training data from GCS?", default=True)
    upload = get_yes_no("Upload results to GCS?", default=True)
    
    if download:
        args.append("--download")
    if upload:
        args.append("--upload")
    
    return args

def evaluate_task():
    """Handle evaluate task."""
    print("\n📊 EVALUATE TASK")
    print("This will evaluate your trained model on the test set.")
    
    dataset_name = get_dataset_selection_with_retry("Select dataset to evaluate")
    
    # RCNN backbone
    rcnn = get_user_choice(
        "\nSelect RCNN backbone to evaluate:",
        ["50", "101", "combo"],
        default="101"
    )
    
    # Dataset format
    dataset_format = get_user_choice(
        "\nDataset format:",
        ["json", "coco"],
        default="json"
    )
    
    # Visualization
    visualize = get_yes_no("Generate visualization outputs?", default=True)
    
    args = [
        "--task", "evaluate",
        "--dataset_name", dataset_name,
        "--rcnn", rcnn,
        "--dataset_format", dataset_format
    ]
    
    if visualize:
        args.append("--visualize")
    
    return args

def inference_task():
    """Handle inference task."""
    print("\n🔍 INFERENCE TASK")
    print("This will run inference on new images using your trained model.")
    
    dataset_name = get_dataset_selection_with_retry("Select dataset for inference")
    
    # Detection threshold
    threshold = get_float_input("Detection threshold", default=0.65, min_val=0.0, max_val=1.0)
    
    # RCNN backbone
    rcnn = get_user_choice(
        "\nSelect RCNN backbone:",
        ["50 (faster, good for small particles)", 
         "101 (slower, good for large particles)", 
         "combo (both models, most accurate)"],
        default="combo (both models, most accurate)"
    )
    rcnn_value = rcnn.split()[0]
    
    # Dataset format
    dataset_format = get_user_choice(
        "\nDataset format:",
        ["json", "coco"],
        default="json"
    )
    
    # Pass mode
    pass_mode = get_user_choice(
        "\nInference pass mode:",
        ["single (one pass, faster)", "multi (iterative deduplication, more accurate)"],
        default="single (one pass, faster)"
    )
    
    max_iters = None
    if pass_mode.startswith("multi"):
        max_iters = get_int_input("Maximum iterations for multi-pass", default=10, min_val=1)
    
    # Visualization options
    visualize = get_yes_no("Generate visualization outputs?", default=True)
    draw_id = get_yes_no("Draw instance IDs on visualizations?", default=False)
    
    args = [
        "--task", "inference",
        "--dataset_name", dataset_name,
        "--threshold", str(threshold),
        "--rcnn", rcnn_value,
        "--dataset_format", dataset_format
    ]
    
    # Pass mode
    if pass_mode.startswith("multi"):
        args.extend(["--pass", "multi", str(max_iters)])
    else:
        args.extend(["--pass", "single"])
    
    if visualize:
        args.append("--visualize")
    if draw_id:
        args.append("--id")
    
    # Download/Upload
    download = get_yes_no("Download inference data from GCS?", default=True)
    upload = get_yes_no("Upload results to GCS?", default=True)
    
    if download:
        args.append("--download")
    if upload:
        args.append("--upload")
    
    return args

def main():
    """Main function."""
    while True:
        clear_screen()
        print_header()
        
        # Task selection
        task = get_user_choice(
            "Select a task to perform:",
            [
                "setup - First-time configuration",
                "prepare - Split dataset into train/test",
                "train - Train a model",
                "evaluate - Evaluate trained model",
                "inference - Run inference on new data",
                "exit - Exit the wizard"
            ]
        )
        
        if task.startswith("exit"):
            print("\nGoodbye! 👋")
            break
        
        # Get task-specific arguments
        task_name = task.split()[0]
        
        if task_name == "setup":
            args = setup_task()
        elif task_name == "prepare":
            args = prepare_task()
        elif task_name == "train":
            args = train_task()
        elif task_name == "evaluate":
            args = evaluate_task()
        elif task_name == "inference":
            args = inference_task()
        
        if args is None:
            continue
        
        # Confirm and execute
        print(f"\n📋 Command to execute:")
        print(f"python main.py {' '.join(args)}")
        print()
        
        if get_yes_no("Execute this command?", default=True):
            print("\n🚀 Executing...")
            try:
                # Get the directory where this script is located
                script_dir = Path(__file__).parent
                main_py = script_dir / "main.py"
                
                subprocess.run([sys.executable, str(main_py)] + args, check=True)
                print("\n✅ Task completed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Task failed with exit code {e.returncode}")
            except KeyboardInterrupt:
                print("\n⚠️ Task interrupted by user")
            
            input("\nPress Enter to continue...")
        else:
            print("❌ Task cancelled.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()