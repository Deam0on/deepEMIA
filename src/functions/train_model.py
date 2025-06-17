"""
Model training module for the UW Computer Vision project.

This module handles:
- Model training with Detectron2
- Data augmentation using Albumentations
- Model quantization for CPU inference
- Custom training pipeline with CPU optimizations

The module provides a complete training pipeline with support for:
- Custom data augmentation
- CPU-optimized training
- Model quantization
- Evaluation during training
"""

## IMPORTS
import os
import shutil
from pathlib import Path

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from src.data.datasets import read_dataset_info, register_datasets
from src.functions.inference import CustomTrainer
from src.utils.config import get_config
from src.utils.logger_utils import system_logger
from src.data.custom_mapper import custom_mapper

config = get_config()

# Resolve paths
SPLIT_DIR = Path(config["paths"]["split_dir"]).expanduser().resolve()
CATEGORY_JSON = Path(config["paths"]["category_json"]).expanduser().resolve()

# Set dynamic threading and quantization engine
torch.set_num_threads(os.cpu_count() // 2)
torch.backends.quantized.engine = "qnnpack"


def get_albumentations_transform() -> A.Compose:
    """
    Creates an Albumentations transform pipeline for data augmentation.

    Returns:
    - A.Compose: Albumentations transform pipeline.
    """
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


def check_disk_space(path: str, min_gb: int = 5) -> None:
    """
    Check if there is at least min_gb GB free at the given path.

    Raises:
        RuntimeError: If there is not enough disk space.
    """
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)
    if free_gb < min_gb:
        system_logger.error(
            f"Not enough disk space: only {free_gb:.2f} GB free at {path}. Minimum required: {min_gb} GB."
        )
        raise RuntimeError("Insufficient disk space for training.")
    else:
        system_logger.info(f"Disk space check passed: {free_gb:.2f} GB free at {path}.")


class AugTrainer(DefaultTrainer):
    def __init__(self, cfg, augment=False):
        self.augment = augment
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg, augment=False):
        # Pass augment flag to the custom mapper
        return build_detection_train_loader(cfg, mapper=lambda d: custom_mapper(d, augment=augment))

    def build_hooks(self):
        # If you have custom hooks, add them here
        return super().build_hooks()


def train_on_dataset(
    dataset_name,
    output_dir,
    dataset_format="json",
    rcnn="101",
    augment=False,
    # ...other args...
):
    """
    Trains a model on the specified dataset with the selected backbone(s).

    Parameters:
    - dataset_name (str): Name of the dataset to train on
    - output_dir (str): Directory to save the trained models
    - dataset_format (str): Annotation format
    - rcnn (str): Backbone to use: "50", "101", or "combo"
    """
    check_disk_space(output_dir, min_gb=1)

    # Read dataset information
    dataset_info = read_dataset_info(CATEGORY_JSON)

    # Register datasets
    register_datasets(dataset_info, dataset_name, dataset_format=dataset_format)

    # Path for the split file
    split_file = os.path.join(SPLIT_DIR, f"{dataset_name}_split.json")
    system_logger.info(f"Split file for {dataset_name}: {split_file}")

    train_data = DatasetCatalog.get(f"{dataset_name}_train")
    test_data = DatasetCatalog.get(f"{dataset_name}_test")
    system_logger.info(f"Training images: {len(train_data)}")
    system_logger.info(f"Test images: {len(test_data)}")
    categories = MetadataCatalog.get(f"{dataset_name}_train").thing_classes
    system_logger.info(f"Categories: {categories}")

    # Log augmentation status
    if augment:
        system_logger.info("Data augmentation is ENABLED for training (using custom_mapper with flips, rotation, brightness).")
    else:
        system_logger.info("Data augmentation is DISABLED for training.")

    def train_with_backbone(
        backbone_name: str, config_file: str, model_suffix: str
    ) -> None:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
        cfg.DATASETS.TEST = (f"{dataset_name}_test",)
        cpu_count = os.cpu_count() or 2
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
        cfg.DATALOADER.NUM_WORKERS = max(1, cpu_count // 2)
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        cfg.SOLVER.IMS_PER_BATCH = 8 if torch.cuda.is_available() else 2
        cfg.SOLVER.BASE_LR = 0.0001 * (cfg.SOLVER.IMS_PER_BATCH / 2)
        num_images = len(DatasetCatalog.get(f"{dataset_name}_train"))
        if num_images < 100:
            cfg.SOLVER.MAX_ITER = max(1000, int(200 * num_images))
        else:
            cfg.SOLVER.MAX_ITER = max(1000, int(100 * num_images))
        cfg.SOLVER.STEPS = [
            int(0.6 * cfg.SOLVER.MAX_ITER),
            int(0.8 * cfg.SOLVER.MAX_ITER),
        ]
        cfg.SOLVER.GAMMA = 0.1
        cfg.SOLVER.WARMUP_ITERS = 1000
        cfg.SOLVER.WARMUP_FACTOR = 1e-3
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16 * cfg.SOLVER.IMS_PER_BATCH
        thing_classes = MetadataCatalog.get(f"{dataset_name}_train").thing_classes
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        dataset_output_dir = os.path.join(
            output_dir, dataset_name, f"rcnn_{model_suffix}"
        )
        os.makedirs(dataset_output_dir, exist_ok=True)
        cfg.OUTPUT_DIR = dataset_output_dir

        system_logger.info(f"Training with backbone: {backbone_name}")
        system_logger.info(
            f"Classes: {MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes}"
        )
        # Log hyperparameters
        system_logger.info(
            f"Training hyperparameters: "
            f"Backbone={backbone_name}, "
            f"Base LR={cfg.SOLVER.BASE_LR}, "
            f"Batch Size={cfg.SOLVER.IMS_PER_BATCH}, "
            f"Max Iter={cfg.SOLVER.MAX_ITER}, "
            f"Steps={cfg.SOLVER.STEPS}, "
            f"Gamma={cfg.SOLVER.GAMMA}, "
            f"Warmup Iters={cfg.SOLVER.WARMUP_ITERS}, "
            f"Warmup Factor={cfg.SOLVER.WARMUP_FACTOR}, "
            f"ROI Batch Size={cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}, "
            f"Num Classes={cfg.MODEL.ROI_HEADS.NUM_CLASSES}, "
            f"Device={cfg.MODEL.DEVICE}"
        )
        trainer = AugTrainer(cfg, augment=augment)
        trainer.build_train_loader = lambda: build_detection_train_loader(
            cfg, mapper=lambda d: custom_mapper(d, augment=augment)
        )
        trainer.resume_or_load(resume=False)
        trainer.train()

        evaluator = COCOEvaluator(f"{dataset_name}_test", output_dir=dataset_output_dir)
        DefaultTrainer.test(cfg, trainer.model, evaluators=[evaluator])

        # Use Detectron2's checkpoint
        src_ckpt = os.path.join(dataset_output_dir, "model_final.pth")
        dst_ckpt = os.path.join(dataset_output_dir, f"model_final_{model_suffix}.pth")
        if os.path.exists(src_ckpt):
            shutil.copy(src_ckpt, dst_ckpt)
            system_logger.info(f"Copied Detectron2 checkpoint to {dst_ckpt}")
        else:
            system_logger.warning(
                f"Detectron2 checkpoint {src_ckpt} not found after training."
            )

    if rcnn == "50":
        train_with_backbone(
            "R50", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", "r50"
        )
    elif rcnn == "101":
        train_with_backbone(
            "R101", "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", "r101"
        )
    elif rcnn == "combo":
        train_with_backbone(
            "R50", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", "r50"
        )
        train_with_backbone(
            "R101", "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", "r101"
        )
    else:
        raise ValueError("Invalid value for rcnn. Choose from '50', '101', or 'combo'.")
