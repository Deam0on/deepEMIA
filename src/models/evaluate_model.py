"""
Model evaluation module for the UW Computer Vision project.

This module handles:
- Model evaluation on test datasets
- Performance metrics calculation
- Prediction visualization
- Results saving and reporting

The module integrates with Detectron2 for evaluation and provides
utilities for visualizing and saving model predictions.
"""

import csv
import os
from pathlib import Path

import yaml
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, MetadataCatalog,
                             build_detection_test_loader)
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer

from src.data.data_preparation import (choose_and_use_model,
                                   get_trained_model_paths, read_dataset_info,
                                   register_datasets)

# Load config once at the start of your program
with open(Path.home() / "uw-com-vision" / "config" / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Constant paths
SPLIT_DIR = Path(config["paths"]["split_dir"]).expanduser().resolve()
CATEGORY_JSON = Path(config["paths"]["category_json"]).expanduser().resolve()


def evaluate_model(dataset_name, output_dir, visualize=False):
    """
    Evaluates the model on the specified dataset and optionally visualizes predictions.

    Parameters:
    - dataset_name (str): Name of the dataset to evaluate
    - output_dir (str): Directory to save evaluation results and visualizations
    - visualize (bool): Whether to generate and save prediction visualizations

    The function performs:
    1. Dataset registration and model loading
    2. COCO-style evaluation
    3. Metrics calculation and saving
    4. Optional prediction visualization

    Returns:
    - None
    """
    # Load dataset information
    dataset_info = read_dataset_info(CATEGORY_JSON)

    # Register the datasets
    register_datasets(dataset_info, dataset_name)

    # Get paths to trained models
    trained_model_paths = get_trained_model_paths(SPLIT_DIR)

    # Set detection threshold
    threshold = 0.45

    # Choose and load the model
    predictor = choose_and_use_model(trained_model_paths, dataset_name, threshold)

    # Initialize configuration
    cfg = get_cfg()

    # Set up COCO evaluator
    evaluator = COCOEvaluator(f"{dataset_name}_test", cfg, False, output_dir=output_dir)

    # Ensure no cached data is used
    coco_format_cache = os.path.join(SPLIT_DIR, f"{dataset_name}_test_coco_format.json")
    if os.path.exists(coco_format_cache):
        os.remove(coco_format_cache)

    # Build the validation data loader
    val_loader = build_detection_test_loader(cfg, f"{dataset_name}_test")

    # Perform inference and evaluate
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(f"Evaluation metrics: {metrics}")

    # Save metrics to CSV
    csv_path = os.path.join(output_dir, "metrics.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, mode="w", newline="") as csv_file:
        fieldnames = ["metric", "value"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})

    print(f"Metrics saved to {csv_path}")

    # Visualize predictions if requested
    if visualize:
        # visualize_predictions(predictor, dataset_name, output_dir)
        pass


def visualize_predictions(predictor, dataset_name, output_dir):
    """
    Visualizes predictions made by the model on the test dataset.

    Parameters:
    - predictor (object): The predictor object used for inference
    - dataset_name (str): Name of the dataset to visualize
    - output_dir (str): Directory to save the visualizations

    The function:
    1. Loads test dataset images
    2. Generates predictions
    3. Creates visualizations with bounding boxes and masks
    4. Saves visualizations to the output directory

    Returns:
    - None
    """
    # Get dataset dictionaries and metadata
    dataset_dicts = DatasetCatalog.get(f"{dataset_name}_test")
    metadata = MetadataCatalog.get(f"{dataset_name}_test")

    # Iterate over the dataset and visualize predictions
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        vis_output = v.get_image()[:, :, ::-1]

        # Save the visualization
        os.makedirs(output_dir, exist_ok=True)
        vis_path = os.path.join(output_dir, os.path.basename(d["file_name"]))
        cv2.imwrite(vis_path, vis_output)
        print(f"Saved visualization to {vis_path}")
