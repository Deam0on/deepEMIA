## IMPORTS
import copy
import csv
import itertools
import json
import os
import random
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import albumentations as A
import cv2
import detectron2.data.transforms as T
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import binary_fill_holes
from shapely.affinity import rotate, scale
from shapely.geometry import Point
from skimage.measure import label
from skimage.morphology import dilation, erosion
from sklearn.model_selection import train_test_split
from torch.quantization import quantize_dynamic
from torch.utils.data import DataLoader

from data_preparation import (
    choose_and_use_model,
    get_split_dicts,
    get_trained_model_paths,
    load_model,
    read_dataset_info,
    register_datasets,
    split_dataset,
)

# Constant paths
SPLIT_DIR = Path.home() / "split_dir"
CATEGORY_JSON = Path.home() / "uw-com-vision" / "dataset_info.json"

# Set dynamic threading and quantization engine
torch.set_num_threads(os.cpu_count() // 2)
torch.backends.quantized.engine = "qnnpack"


def get_albumentations_transform():
    return A.Compose(
        [
            A.Resize(800, 800),
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def custom_mapper(dataset_dicts):
    """
    Custom data mapper function using Albumentations for faster CPU transforms.
    """
    dataset_dicts = copy.deepcopy(dataset_dicts)
    image = cv2.imread(dataset_dicts["file_name"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_albumentations_transform()
    augmented = transform(image=image)
    image = augmented["image"]

    dataset_dicts["image"] = image

    annos = [
        utils.transform_instance_annotations(obj, None, image.shape[1:])
        for obj in dataset_dicts.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]

    instances = utils.annotations_to_instances(annos, image.shape[1:])
    dataset_dicts["instances"] = utils.filter_empty_instances(instances)

    return dataset_dicts


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        dataset = build_detection_train_loader(
            cfg,
            mapper=custom_mapper,
            sampler=None,
            total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
        ).dataset  # Extract the dataset only

        cpu_count = os.cpu_count() or 2
        num_workers = max(1, cpu_count // 2)

        return DataLoader(
            dataset,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=False,
        )


def train_on_dataset(dataset_name, output_dir):
    """
    Trains a model on the specified dataset.

    Parameters:
    - dataset_name: Name of the dataset.
    - output_dir: Directory to save the trained model.
    """
    # Read dataset information
    dataset_info = read_dataset_info(CATEGORY_JSON)

    # Register datasets
    register_datasets(dataset_info, dataset_name)

    # Debug prints for verification
    print(DatasetCatalog.get(f"{dataset_name}_train"))
    print(DatasetCatalog.get(f"{dataset_name}_test"))

    # Path for the split file
    split_file = os.path.join(SPLIT_DIR, f"{dataset_name}_split.json")
    print(f"Split file for {dataset_name}: {split_file}")

    # Configuration for training
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
    cfg.DATASETS.TEST = (f"{dataset_name}_test",)
    cpu_count = os.cpu_count() or 2
    cfg.DATALOADER.NUM_WORKERS = max(1, cpu_count // 2)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    )
    cfg.SOLVER.IMS_PER_BATCH = 8 if torch.cuda.is_available() else 2
    cfg.SOLVER.BASE_LR = 0.0001 * (cfg.SOLVER.IMS_PER_BATCH / 2)
    num_images = len(DatasetCatalog.get(f"{dataset_name}_train"))
    # cfg.SOLVER.MAX_ITER = max(1000, int(100 * num_images))
    # MAX_ITER logic
    num_images = len(DatasetCatalog.get(f"{dataset_name}_train"))
    if num_images < 100:
        cfg.SOLVER.MAX_ITER = max(1000, int(200 * num_images))
    else:
        cfg.SOLVER.MAX_ITER = max(1000, int(100 * num_images))
    #cfg.SOLVER.STEPS = []

    # LR scheduler and warmup for better CPU stability
    cfg.SOLVER.STEPS = [int(0.6 * cfg.SOLVER.MAX_ITER), int(0.8 * cfg.SOLVER.MAX_ITER)]
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_FACTOR = 1e-3
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16 * cfg.SOLVER.IMS_PER_BATCH

    # Set the number of classes
    thing_classes = MetadataCatalog.get(f"{dataset_name}_train").thing_classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Output directory for the dataset
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = dataset_output_dir

    # Initialize and start the trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator(f"{dataset_name}_test", output_dir=dataset_output_dir)
    DefaultTrainer.test(cfg, trainer.model, evaluators=[evaluator])

    # Save the trained model
    model_path = os.path.join(dataset_output_dir, "model_final.pth")
    torch.save(trainer.model.state_dict(), model_path)
    print(f"Model trained on {dataset_name} saved to {model_path}")

    # Quantize the trained model
    quantized_model = quantize_dynamic(trainer.model, {nn.Linear}, dtype=torch.qint8)

    # Save quantized model separately
    quantized_model_path = os.path.join(dataset_output_dir, "model_final_quantized.pth")
    torch.save(quantized_model, quantized_model_path)
    print(f"Quantized model saved to {quantized_model_path}")
