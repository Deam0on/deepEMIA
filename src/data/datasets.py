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
import yaml
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from sklearn.model_selection import train_test_split

from src.utils.config import get_config
from src.utils.logger_utils import system_logger

config = get_config()

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
    random.seed(seed)
    label_files = [f for f in os.listdir(img_dir) if f.endswith(".json")]
    train_files, test_files = train_test_split(
        label_files, test_size=test_size, random_state=seed
    )
    os.makedirs(SPLIT_DIR, exist_ok=True)
    split_file = os.path.join(SPLIT_DIR, f"{dataset_name}_split.json")
    split_data = {"train": train_files, "test": test_files}
    with open(split_file, "w") as f:
        json.dump(split_data, f)
    system_logger.info(f"Training & Testing data successfully split into {split_file}")
    return train_files, test_files


def register_datasets(dataset_info, dataset_name, test_size=0.2, dataset_format="json"):
    """
    Registers the selected dataset in the Detectron2 framework.

    Parameters:
    - dataset_info (dict): Dictionary containing dataset names and their info
    - dataset_name (str): Name of the dataset to register
    - test_size (float): Proportion of the dataset to include in the test split


    For COCO, DATASET needs to be in the format:

    DATASET/
    └── my_crystal_dataset_coco/      <-- Your new dataset_name
        ├── annotations/
        │   ├── instances_train.json  <-- COCO JSON for the training set
        │   └── instances_test.json   <-- COCO JSON for the testing set
        ├── train/                    <-- Directory with training images
        │   ├── image1.jpg
        │   └── image2.jpg
        └── test/                     <-- Directory with testing images
            ├── image3.jpg
            └── image4.jpg

    Raises:
    - ValueError: If the dataset name is not found in dataset_info
    """
    if dataset_format == "coco":
        system_logger.info(f"Registering COCO dataset: {dataset_name}")
        base_path = os.path.join(os.path.expanduser("~"), "DATASET", dataset_name)
        train_json_path = os.path.join(base_path, "annotations", "instances_train.json")
        train_images_path = os.path.join(base_path, "train")
        test_json_path = os.path.join(base_path, "annotations", "instances_test.json")
        test_images_path = os.path.join(base_path, "test")
        register_coco_instances(
            f"{dataset_name}_train", {}, train_json_path, train_images_path
        )
        register_coco_instances(
            f"{dataset_name}_test", {}, test_json_path, test_images_path
        )
        system_logger.info("COCO dataset registration complete.")

    elif dataset_format == "json":
        system_logger.info(f"Registering custom JSON dataset: {dataset_name}")
        if dataset_name not in dataset_info:
            raise ValueError(f"Dataset '{dataset_name}' not found in dataset_info.")

        img_dir, label_dir, thing_classes = dataset_info[dataset_name]
        system_logger.info(
            f"Processing dataset: {dataset_name}, Info: {dataset_info[dataset_name]}"
        )

        split_file = os.path.join(SPLIT_DIR, f"{dataset_name}_split.json")
        category_key = dataset_name

        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                split_data = json.load(f)
            train_files = split_data["train"]
            test_files = split_data["test"]
        else:
            train_files, test_files = split_dataset(
                img_dir, dataset_name, test_size=test_size
            )
            split_data = {"train": train_files, "test": test_files}
            os.makedirs(SPLIT_DIR, exist_ok=True)
            with open(split_file, "w") as f:
                json.dump(split_data, f)
            system_logger.info(f"Split created and saved at {split_file}")

        DatasetCatalog.register(
            f"{dataset_name}_train",
            lambda d="train": get_split_dicts(
                img_dir, label_dir, split_data[d], CATEGORY_JSON, category_key
            ),
        )
        MetadataCatalog.get(f"{dataset_name}_train").set(thing_classes=thing_classes)

        DatasetCatalog.register(
            f"{dataset_name}_test",
            lambda d="test": get_split_dicts(
                img_dir, label_dir, split_data[d], CATEGORY_JSON, category_key
            ),
        )
        MetadataCatalog.get(f"{dataset_name}_test").set(thing_classes=thing_classes)
        system_logger.info("Custom JSON dataset registration complete.")

    else:
        raise ValueError(f"Unknown dataset_format: {dataset_format}")


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
    dataset_info = read_dataset_info(category_json)
    if category_key not in dataset_info:
        raise ValueError(f"Category key '{category_key}' not found in JSON")

    category_names = dataset_info[category_key][2]
    category_name_to_id = {name: idx for idx, name in enumerate(category_names)}
    system_logger.info(f"Category Mapping: {category_name_to_id}")

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
                system_logger.warning(f"Category Name Not Found: {categoryName}")
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
        dataset_info = {
            k: tuple(v) if isinstance(v, list) else v for k, v in data.items()
        }
        system_logger.info(f"Dataset Info: {dataset_info}")
    return dataset_info
