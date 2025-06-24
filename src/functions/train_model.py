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
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

import optuna
import yaml

from src.data.custom_mapper import custom_mapper
from src.data.datasets import read_dataset_info, register_datasets
from src.functions.inference import CustomTrainer
from src.utils.config import get_config
from src.utils.logger_utils import system_logger

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
        return build_detection_train_loader(
            cfg, mapper=lambda d: custom_mapper(d, augment=augment)
        )

    def build_hooks(self):
        # If you have custom hooks, add them here
        return super().build_hooks()


def train_with_backbone(
    backbone_name: str,
    config_file: str,
    model_suffix: str,
    dataset_name: str,
    output_dir: str,
    augment: bool,
    base_lr: float = 0.0001,
    ims_per_batch: int = 2,
    max_iter: int = None,
    warmup_iters: int = 1000,
    warmup_factor: float = 1e-3,
    gamma: float = 0.1,
    batch_size_per_image: int = None,
    return_metric: bool = False,
):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
    cfg.DATASETS.TEST = (f"{dataset_name}_test",)
    cpu_count = os.cpu_count() or 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.DATALOADER.NUM_WORKERS = max(1, cpu_count // 2)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    num_images = len(DatasetCatalog.get(f"{dataset_name}_train"))
    if max_iter is None:
        if num_images < 100:
            cfg.SOLVER.MAX_ITER = max(1000, int(200 * num_images))
        else:
            cfg.SOLVER.MAX_ITER = max(1000, int(100 * num_images))
    else:
        cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = [
        int(0.6 * cfg.SOLVER.MAX_ITER),
        int(0.8 * cfg.SOLVER.MAX_ITER),
    ]
    cfg.SOLVER.GAMMA = gamma
    cfg.SOLVER.WARMUP_ITERS = warmup_iters
    cfg.SOLVER.WARMUP_FACTOR = warmup_factor
    if batch_size_per_image is None:
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16 * cfg.SOLVER.IMS_PER_BATCH
    else:
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
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
    if augment:
        trainer = AugTrainer(cfg, augment=augment)
        trainer.build_train_loader = lambda: build_detection_train_loader(
            cfg, mapper=lambda d: custom_mapper(d, augment=augment)
        )
    else:
        trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator(f"{dataset_name}_test", output_dir=dataset_output_dir)
    results = DefaultTrainer.test(cfg, trainer.model, evaluators=[evaluator])

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

    # Return metric for Optuna
    if return_metric:
        # Use bbox AP as main metric (change as needed)
        return results["bbox"]["AP"] if "bbox" in results else results["segm"]["AP"]
    return None


def optuna_objective(trial, dataset_name, output_dir, backbone="R50", augment=False):
    # Suggest hyperparameters
    base_lr = trial.suggest_loguniform("base_lr", 1e-5, 1e-2)
    ims_per_batch = trial.suggest_categorical("ims_per_batch", [2, 4, 8])
    warmup_iters = trial.suggest_int("warmup_iters", 500, 2000)
    gamma = trial.suggest_float("gamma", 0.05, 0.2)
    batch_size_per_image = trial.suggest_categorical("batch_size_per_image", [32, 64, 128])

    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" if backbone == "R50" else "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    model_suffix = backbone.lower()

    ap = train_with_backbone(
        backbone_name=backbone,
        config_file=config_file,
        model_suffix=model_suffix,
        dataset_name=dataset_name,
        output_dir=output_dir,
        augment=augment,
        base_lr=base_lr,
        ims_per_batch=ims_per_batch,
        warmup_iters=warmup_iters,
        gamma=gamma,
        batch_size_per_image=batch_size_per_image,
        return_metric=True,
    )
    return ap


def optimize_hyperparameters(
    dataset_name,
    output_dir,
    n_trials=10,
    backbone="R50",
    augment=False,
):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(
            trial, dataset_name, output_dir, backbone=backbone, augment=augment
        ),
        n_trials=n_trials,
    )
    system_logger.info(f"Best trial: {study.best_trial.value}")
    system_logger.info(f"Best params: {study.best_trial.params}")
    # Save best params
    for t in study.trials:
        system_logger.info(f"Trial {t.number}: value={t.value}, params={t.params}")
    save_best_rcnn_hyperparameters(backbone, study.best_trial.params)
    system_logger.info(f"Best hyperparameters for {backbone} saved to config/config.yaml")
    return study.best_trial


def train_on_dataset(
    dataset_name,
    output_dir,
    dataset_format="json",
    rcnn="101",
    augment=False,
    optimize=False,
    n_trials=10,
):
    """
    Trains a model on the specified dataset with the selected backbone(s).
    If optimize=True, runs Optuna HPO instead of standard training.
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
        system_logger.info(
            "Data augmentation is ENABLED for training (using custom_mapper with flips, rotation, brightness)."
        )
    else:
        system_logger.info("Data augmentation is DISABLED for training.")

    if optimize:
        backbone = "R50" if rcnn == "50" else "R101"
        optimize_hyperparameters(
            dataset_name=dataset_name,
            output_dir=output_dir,
            n_trials=n_trials,
            backbone=backbone,
            augment=augment,
        )
        return

    def _train(backbone_name, config_file, model_suffix):
        params = load_rcnn_hyperparameters(backbone_name, use_best=True)
        train_with_backbone(
            backbone_name=backbone_name,
            config_file=config_file,
            model_suffix=model_suffix,
            dataset_name=dataset_name,
            output_dir=output_dir,
            augment=augment,
            base_lr=params.get("base_lr", 0.0001),
            ims_per_batch=params.get("ims_per_batch", 2),
            warmup_iters=params.get("warmup_iters", 1000),
            gamma=params.get("gamma", 0.1),
            batch_size_per_image=params.get("batch_size_per_image", 64),
        )

    if rcnn == "50":
        _train("R50", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", "r50")
    elif rcnn == "101":
        _train("R101", "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", "r101")
    elif rcnn == "combo":
        _train("R50", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", "r50")
        _train("R101", "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", "r101")
    else:
        raise ValueError("Invalid value for rcnn. Choose from '50', '101', or 'combo'.")

def load_rcnn_hyperparameters(rcnn_type, use_best=True):
    config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    section = "best" if use_best and config_data.get("rcnn_hyperparameters", {}).get("best", {}).get(rcnn_type) else "default"
    params = config_data["rcnn_hyperparameters"][section][rcnn_type]
    return params

def save_best_rcnn_hyperparameters(rcnn_type, best_params):
    config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    # Save current best as backup if not present
    if "rcnn_hyperparameters" not in config_data:
        config_data["rcnn_hyperparameters"] = {"default": {}, "best": {}}
    if rcnn_type not in config_data["rcnn_hyperparameters"]["default"]:
        config_data["rcnn_hyperparameters"]["default"][rcnn_type] = best_params
    # Save best params
    config_data["rcnn_hyperparameters"]["best"][rcnn_type] = best_params
    with open(config_path, "w") as f:
        yaml.safe_dump(config_data, f)
