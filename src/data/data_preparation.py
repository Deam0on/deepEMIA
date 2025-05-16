"""
Data preparation module for the UW Computer Vision project.

This module handles:
- Dataset splitting into train and test sets
- Dataset registration for Detectron2
- Model loading and preparation
- Dataset information management

The module integrates with Detectron2 for computer vision tasks and provides
utilities for handling various data formats and model types.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import shapely.affinity
import shapely.geometry
import torch
import yaml
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from sklearn.model_selection import train_test_split

# Load config once at the start of your program
with open(Path.home() / "uw-com-vision" / "config" / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Constant paths
SPLIT_DIR = Path(config["paths"]["split_dir"]).expanduser().resolve()
CATEGORY_JSON = Path(config["paths"]["category_json"]).expanduser().resolve()


def split_dataset(img_dir, dataset_name, test_size=0.2, seed=42):
    """
    Splits the dataset into training and testing sets and saves the split information.

    Parameters:
    - img_dir (str): Directory containing images
    - dataset_name (str): Name of the dataset
    - test_size (float): Proportion of the dataset to include in the test split
    - seed (int): Random seed for reproducibility

    Returns:
    - tuple: (train_files, test_files) Lists of training and testing label files
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    # List all label files in the image directory
    label_files = [f for f in os.listdir(img_dir) if f.endswith(".json")]

    # Split the label files into training and testing sets
    train_files, test_files = train_test_split(
        label_files, test_size=test_size, random_state=seed
    )

    # Create directory to save the split information if it doesn't exist
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # Path to save the split JSON file
    split_file = os.path.join(SPLIT_DIR, f"{dataset_name}_split.json")
    split_data = {"train": train_files, "test": test_files}

    # Save the split data to a JSON file
    with open(split_file, "w") as f:
        json.dump(split_data, f)

    print(f"Training & Testing data successfully split into {split_file}")

    return train_files, test_files


def register_datasets(dataset_info, dataset_name, test_size=0.2):
    """
    Registers the selected dataset in the Detectron2 framework.

    Parameters:
    - dataset_info (dict): Dictionary containing dataset names and their info
    - dataset_name (str): Name of the dataset to register
    - test_size (float): Proportion of the dataset to include in the test split

    Raises:
    - ValueError: If the dataset name is not found in dataset_info
    """
    if dataset_name not in dataset_info:
        raise ValueError(f"Dataset '{dataset_name}' not found in dataset_info.")

    img_dir, label_dir, thing_classes = dataset_info[dataset_name]

    print(f"Processing dataset: {dataset_name}, Info: {dataset_info[dataset_name]}")

    # Load or split the dataset
    split_file = os.path.join(SPLIT_DIR, f"{dataset_name}_split.json")
    category_key = dataset_name

    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            split_data = json.load(f)
        train_files = split_data["train"]
        test_files = split_data["test"]
    else:
        # Create split data if it doesn't exist
        train_files, test_files = split_dataset(
            img_dir, dataset_name, test_size=test_size
        )
        split_data = {"train": train_files, "test": test_files}
        os.makedirs(SPLIT_DIR, exist_ok=True)
        with open(split_file, "w") as f:
            json.dump(split_data, f)
        print(f"Split created and saved at {split_file}")

    # Register training dataset
    DatasetCatalog.register(
        f"{dataset_name}_train",
        lambda img_dir=img_dir, label_dir=label_dir, files=train_files: get_split_dicts(
            img_dir, label_dir, files, CATEGORY_JSON, category_key
        ),
    )
    MetadataCatalog.get(f"{dataset_name}_train").set(thing_classes=thing_classes)

    # Register testing dataset
    DatasetCatalog.register(
        f"{dataset_name}_test",
        lambda img_dir=img_dir, label_dir=label_dir, files=test_files: get_split_dicts(
            img_dir, label_dir, files, CATEGORY_JSON, category_key
        ),
    )
    MetadataCatalog.get(f"{dataset_name}_test").set(thing_classes=thing_classes)


def get_split_dicts(img_dir, label_dir, files, category_json, category_key):
    """
    Generates a list of dictionaries for Detectron2 dataset registration.

    Parameters:
    - img_dir (str): Directory containing images
    - label_dir (str): Directory containing labels
    - files (list): List of label files to process
    - category_json (str): Path to the JSON file containing category information
    - category_key (str): Key in JSON to select category names

    Returns:
    - list: List of dictionaries with image and annotation data

    Raises:
    - ValueError: If the category key is not found in the JSON file
    """
    # Load category names and create a mapping to category IDs
    dataset_info = read_dataset_info(category_json)

    if category_key not in dataset_info:
        raise ValueError(f"Category key '{category_key}' not found in JSON")

    category_names = dataset_info[category_key][
        2
    ]  # Extract category names from the JSON
    category_name_to_id = {name: idx for idx, name in enumerate(category_names)}

    print(f"Category Mapping: {category_name_to_id}")

    dataset_dicts = []
    for idx, file in enumerate(files):
        json_file = os.path.join(label_dir, file)
        with open(json_file) as f:
            imgs_anns = json.load(f)

        record = {}
        filename = os.path.join(img_dir, imgs_anns["metadata"]["name"])
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = imgs_anns["metadata"]["height"]
        record["width"] = imgs_anns["metadata"]["width"]

        annos = imgs_anns["instances"]
        objs = []

        for anno in annos:
            categoryName = anno["className"]
            type = anno["type"]

            if type == "ellipse":
                cx = anno["cx"]
                cy = anno["cy"]
                rx = anno["rx"]
                ry = anno["ry"]
                theta = anno["angle"]
                ellipse = ((cx, cy), (rx, ry), theta)
                circ = shapely.geometry.Point(ellipse[0]).buffer(1)
                ell = shapely.affinity.scale(
                    circ, int(ellipse[1][0]), int(ellipse[1][1])
                )
                ellr = shapely.affinity.rotate(ell, ellipse[2])
                px, py = ellr.exterior.coords.xy
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
            elif type == "polygon":
                px = anno["points"][0:-1:2]
                py = anno["points"][1:-1:2]
                px.append(anno["points"][0])
                py.append(anno["points"][-1])
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

            if categoryName in category_name_to_id:
                category_id = category_name_to_id[categoryName]
            else:
                print(f"Warning: Category Name Not Found: {categoryName}")
                continue

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def get_trained_model_paths(base_dir):
    """
    Retrieves paths to trained models in a given base directory.

    Parameters:
    - base_dir (str): Directory containing trained models

    Returns:
    - dict: Dictionary with dataset names as keys and model paths as values
    """
    model_paths = {}
    for dataset_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, dataset_name)
        model_path = os.path.join(model_dir, "model_final.pth")
        if os.path.exists(model_path):
            model_paths[dataset_name] = model_path
    return model_paths


def load_model(cfg, model_path, dataset_name, is_quantized=False):
    """
    Loads a trained model. If quantized fails, fallback must be handled by caller.

    Parameters:
    - cfg (CfgNode): Detectron2 config object
    - model_path (str): Path to model file
    - dataset_name (str): Dataset name for metadata
    - is_quantized (bool): Whether the model is quantized

    Returns:
    - object: Predictor object for making predictions
    """
    if is_quantized:
        try:
            model = torch.load(model_path, map_location=cfg.MODEL.DEVICE)
            model.eval()

            class QuantizedPredictor:
                def __init__(self, model):
                    self.model = model

                def __call__(self, image):
                    with torch.no_grad():
                        image_tensor = (
                            torch.from_numpy(image)
                            .permute(2, 0, 1)
                            .float()
                            .unsqueeze(0)
                        )
                        image_tensor = image_tensor.to(
                            next(self.model.parameters()).device
                        )
                        inputs = [
                            {
                                "image": image_tensor[0],
                                "height": image.shape[0],
                                "width": image.shape[1],
                            }
                        ]
                        return self.model(inputs)[0]

            return QuantizedPredictor(model)

        except Exception as e:
            print(f"Failed to load or initialize quantized model: {e}")
            raise RuntimeError("Quantized model load failed.")

    # fallback or standard model
    cfg.MODEL.WEIGHTS = model_path
    thing_classes = MetadataCatalog.get(f"{dataset_name}_train").thing_classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    return DefaultPredictor(cfg)


def choose_and_use_model(model_paths, dataset_name, threshold):
    """
    Chooses and loads the appropriate model for a given dataset.

    Parameters:
    - model_paths (dict): Dictionary of available model paths
    - dataset_name (str): Name of the dataset
    - threshold (float): Confidence threshold for predictions

    Returns:
    - tuple: (predictor, metadata) The loaded model and its metadata
    """
    if dataset_name not in model_paths:
        print(f"No model found for dataset {dataset_name}")
        return None

    base_model_path = model_paths[dataset_name]
    quantized_model_path = base_model_path.replace(
        "model_final.pth", "model_final_quantized.pth"
    )

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        )
    )
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    # try quantized if no CUDA and file exists
    if not torch.cuda.is_available() and os.path.exists(quantized_model_path):
        try:
            print(f"Trying quantized model for {dataset_name}")
            return load_model(
                cfg, quantized_model_path, dataset_name, is_quantized=True
            )
        except RuntimeError:
            print(f"Falling back to standard model for {dataset_name}")

    print(f"Using standard model for {dataset_name}")
    return load_model(cfg, base_model_path, dataset_name, is_quantized=False)


def read_dataset_info(file_path):
    """
    Reads dataset information from a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file

    Returns:
    - dict: Dataset information
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        # Convert list values back to tuples for consistency with the original data
        dataset_info = {
            k: tuple(v) if isinstance(v, list) else v for k, v in data.items()
        }
        print("Dataset Info:", dataset_info)
    return dataset_info
