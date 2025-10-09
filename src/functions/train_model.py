"""
Model training module for the deepEMIA project.

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
from typing import Dict, Optional, Union

import albumentations as A
import torch
import yaml
import optuna
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

from src.data.custom_mapper import custom_mapper
from src.data.datasets import read_dataset_info, register_datasets
from src.utils.config import get_config
from src.utils.logger_utils import system_logger
from src.utils.exceptions import TrainingError, ModelLoadError, ConfigurationError
from src.utils.constants import DefaultHyperparameters, ProcessingLimits

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

    dataset_output_dir = os.path.join(output_dir, dataset_name, f"rcnn_{model_suffix}")
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

    system_logger.info(
        f"Starting training for {backbone_name} on dataset '{dataset_name}'..."
    )
    trainer.train()
    system_logger.info(
        f"Training completed successfully for {backbone_name} on dataset '{dataset_name}'"
    )

    system_logger.info("Evaluating on test set...")
    evaluator = COCOEvaluator(f"{dataset_name}_test", output_dir=dataset_output_dir)
    results = DefaultTrainer.test(cfg, trainer.model, evaluators=[evaluator])

    # Log evaluation results
    system_logger.info(f"Evaluation complete for {backbone_name} on '{dataset_name}'")

    if "bbox" in results:
        bbox_results = results["bbox"]
        system_logger.info(
            f"BBox - AP: {bbox_results.get('AP', 'N/A'):.4f}, "
            f"AP50: {bbox_results.get('AP50', 'N/A'):.4f}, "
            f"AP75: {bbox_results.get('AP75', 'N/A'):.4f}"
        )

    if "segm" in results:
        segm_results = results["segm"]
        system_logger.info(
            f"Segm - AP: {segm_results.get('AP', 'N/A'):.4f}, "
            f"AP50: {segm_results.get('AP50', 'N/A'):.4f}, "
            f"AP75: {segm_results.get('AP75', 'N/A'):.4f}"
        )

    # Primary metric for summary
    primary_ap = (
        results["bbox"]["AP"]
        if "bbox" in results
        else results["segm"]["AP"] if "segm" in results else "N/A"
    )

    # Use Detectron2's checkpoint
    src_ckpt = os.path.join(dataset_output_dir, "model_final.pth")
    dst_ckpt = os.path.join(dataset_output_dir, f"model_final_{model_suffix}.pth")
    if os.path.exists(src_ckpt):
        shutil.copy(src_ckpt, dst_ckpt)
    else:
        system_logger.warning(
            f"Detectron2 checkpoint {src_ckpt} not found after training."
        )

    system_logger.info(f"Training complete: {backbone_name}, AP={primary_ap:.4f}, output: {dataset_output_dir}")

    # Return metric for Optuna
    if return_metric:
        # Use bbox AP as main metric (change as needed)
        return results["bbox"]["AP"] if "bbox" in results else results["segm"]["AP"]
    return None


def optuna_objective(trial, dataset_name, output_dir, backbone="R50", augment=False):
    """
    Optuna objective function for hyperparameter optimization.

    Parameters:
    - trial (optuna.Trial): Optuna trial object for suggesting hyperparameters
    - dataset_name (str): Name of the dataset to train on
    - output_dir (str): Directory to save trained models
    - backbone (str): RCNN backbone architecture ("R50" or "R101")
    - augment (bool): Whether to apply data augmentation

    Returns:
    - float: Average Precision (AP) score to maximize
    """
    # Suggest hyperparameters
    base_lr = trial.suggest_loguniform("base_lr", 1e-5, 1e-2)
    ims_per_batch = trial.suggest_categorical("ims_per_batch", [2, 4, 8])
    warmup_iters = trial.suggest_int("warmup_iters", 500, 2000)
    gamma = trial.suggest_float("gamma", 0.05, 0.2)
    batch_size_per_image = trial.suggest_categorical(
        "batch_size_per_image", [32, 64, 128]
    )

    config_file = (
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        if backbone == "R50"
        else "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    )
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
    system_logger.info(f"Optimizing hyperparameters: {backbone} on {dataset_name}, {n_trials} trials")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(
            trial, dataset_name, output_dir, backbone=backbone, augment=augment
        ),
        n_trials=n_trials,
    )

    system_logger.info(f"Optimization complete: best AP={study.best_trial.value:.4f}, params={study.best_trial.params}")

    save_best_rcnn_hyperparameters(
        backbone, study.best_trial.params, dataset_name=dataset_name
    )

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

    train_data = DatasetCatalog.get(f"{dataset_name}_train")
    test_data = DatasetCatalog.get(f"{dataset_name}_test")
    categories = MetadataCatalog.get(f"{dataset_name}_train").thing_classes
    system_logger.info(
        f"Dataset: {dataset_name}, train: {len(train_data)}, test: {len(test_data)}, "
        f"categories: {len(categories)}, augment: {augment}"
    )

    if optimize:
        backbone = "R50" if rcnn == "50" else "R101"
        optimize_hyperparameters(
            dataset_name=dataset_name,
            output_dir=output_dir,
            n_trials=n_trials,
            backbone=backbone,
            augment=augment,
        )
        system_logger.info(f"Hyperparameter optimization complete: {dataset_name}, {backbone}, {n_trials} trials")
        return

    def _train(backbone_name, config_file, model_suffix):
        params = load_rcnn_hyperparameters(
            backbone_name, dataset_name=dataset_name, use_best=True
        )
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

    system_logger.info(
        f"Training complete: {dataset_name}, "
        f"{'R50' if rcnn == '50' else 'R101' if rcnn == '101' else 'R50+R101'}, "
        f"output: {output_dir}/{dataset_name}/"
    )


def load_rcnn_hyperparameters(
    rcnn_type: str, dataset_name: str = None, use_best: bool = True
) -> Dict[str, Union[float, int]]:
    """
    Load RCNN hyperparameters from configuration file.

    Args:
        rcnn_type: Type of RCNN model ('R50' or 'R101')
        dataset_name: Name of the dataset (for dataset-specific parameters)
        use_best: Whether to use best parameters if available

    Returns:
        Dictionary of hyperparameters

    Raises:
        ConfigurationError: If configuration cannot be loaded or is invalid
    """
    try:
        config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        rcnn_config = config_data.get("rcnn_hyperparameters", {})
        if not rcnn_config:
            raise ConfigurationError("No RCNN hyperparameters found in configuration")

        # Priority order for loading hyperparameters:
        # 1. Dataset-specific best (best_<dataset_name>)
        # 2. Global best (best)
        # 3. Default

        params = None
        source = None

        if use_best and dataset_name:
            # Try dataset-specific best first
            dataset_best_key = f"best_{dataset_name}"
            if (
                dataset_best_key in rcnn_config
                and rcnn_type in rcnn_config[dataset_best_key]
            ):
                dataset_params = rcnn_config[dataset_best_key][rcnn_type]
                if dataset_params:  # Make sure it's not empty
                    params = dataset_params
                    source = f"dataset-specific best ({dataset_best_key})"

        if params is None and use_best:
            # Try global best
            if "best" in rcnn_config and rcnn_type in rcnn_config["best"]:
                best_params = rcnn_config["best"][rcnn_type]
                if best_params:  # Make sure it's not empty
                    params = best_params
                    source = "global best"

        if params is None:
            # Fall back to default
            if "default" not in rcnn_config:
                raise ConfigurationError("No default hyperparameters section found")

            if rcnn_type not in rcnn_config["default"]:
                raise ConfigurationError(
                    f"No default hyperparameters found for {rcnn_type}"
                )

            params = rcnn_config["default"][rcnn_type]
            source = "default"

        # Validate required parameters
        required_params = [
            "base_lr",
            "ims_per_batch",
            "warmup_iters",
            "gamma",
            "batch_size_per_image",
        ]
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            raise ConfigurationError(
                f"Missing required hyperparameters: {missing_params}"
            )

        dataset_info = f" for dataset '{dataset_name}'" if dataset_name else ""
        system_logger.info(
            f"Loaded {source} hyperparameters for {rcnn_type}{dataset_info}"
        )
        return params

    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing configuration file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading hyperparameters: {e}")


def save_best_rcnn_hyperparameters(
    rcnn_type: str, best_params: Dict[str, Union[float, int]], dataset_name: str = None
) -> None:
    """
    Save best RCNN hyperparameters to configuration file.

    Args:
        rcnn_type: Type of RCNN model ('R50' or 'R101')
        best_params: Dictionary of hyperparameters to save
        dataset_name: Name of the dataset (for dataset-specific parameters)

    Raises:
        ConfigurationError: If configuration cannot be loaded or saved
    """
    try:
        config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"

        # Read existing config
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Initialize structure if needed
        if "rcnn_hyperparameters" not in config_data:
            config_data["rcnn_hyperparameters"] = {"default": {}, "best": {}}

        rcnn_config = config_data["rcnn_hyperparameters"]
        if "default" not in rcnn_config:
            rcnn_config["default"] = {}
        if "best" not in rcnn_config:
            rcnn_config["best"] = {}

        # Save current as default if not present
        if rcnn_type not in rcnn_config["default"]:
            rcnn_config["default"][rcnn_type] = best_params.copy()

        # Determine where to save best params
        if dataset_name:
            # Save dataset-specific best params
            best_key = f"best_{dataset_name}"
            if best_key not in rcnn_config:
                rcnn_config[best_key] = {}

            rcnn_config[best_key][rcnn_type] = best_params
            system_logger.info(
                f"Saved dataset-specific best hyperparameters for {rcnn_type} (dataset: {dataset_name})"
            )
        else:
            # Save global best params
            rcnn_config["best"][rcnn_type] = best_params
            system_logger.info(f"Saved global best hyperparameters for {rcnn_type}")

        # Write back to file
        with open(config_path, "w") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error writing configuration file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error saving hyperparameters: {e}")


def list_available_hyperparameters() -> Dict[str, Dict[str, dict]]:
    """
    List all available hyperparameter configurations.

    Returns:
        Dictionary containing all hyperparameter configurations organized by type
    """
    try:
        config_path = Path.home() / "deepEMIA" / "config" / "config.yaml"
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        rcnn_config = config_data.get("rcnn_hyperparameters", {})

        # Organize configurations
        configurations = {
            "default": rcnn_config.get("default", {}),
            "global_best": rcnn_config.get("best", {}),
            "dataset_specific": {},
        }

        # Find all dataset-specific configurations
        for key, value in rcnn_config.items():
            if key.startswith("best_") and key not in ["best"]:
                dataset_name = key[5:]  # Remove "best_" prefix
                configurations["dataset_specific"][dataset_name] = value

        # Log summary
        system_logger.info("Available hyperparameter configurations:")
        system_logger.info(f"  Default: {list(configurations['default'].keys())}")
        system_logger.info(
            f"  Global best: {list(configurations['global_best'].keys())}"
        )
        system_logger.info(
            f"  Dataset-specific: {list(configurations['dataset_specific'].keys())}"
        )

        return configurations

    except Exception as e:
        system_logger.error(f"Error listing hyperparameters: {e}")
        return {"default": {}, "global_best": {}, "dataset_specific": {}}


def get_hyperparameter_info(dataset_name: str = None, rcnn_type: str = None) -> str:
    """
    Get formatted information about hyperparameters for a dataset/model combination.

    Args:
        dataset_name: Name of the dataset (optional)
        rcnn_type: Type of RCNN model ('R50' or 'R101', optional)

    Returns:
        Formatted string with hyperparameter information
    """
    try:
        info_lines = []

        if rcnn_type:
            rcnn_types = [rcnn_type]
        else:
            rcnn_types = ["R50", "R101"]

        for rcnn in rcnn_types:
            try:
                params = load_rcnn_hyperparameters(
                    rcnn, dataset_name=dataset_name, use_best=True
                )
                info_lines.append(
                    f"\n{rcnn} hyperparameters"
                    + (f" for '{dataset_name}'" if dataset_name else "")
                    + ":"
                )
                for key, value in params.items():
                    info_lines.append(f"  {key}: {value}")
            except ConfigurationError as e:
                info_lines.append(f"\n{rcnn}: Error loading hyperparameters - {e}")

        return "\n".join(info_lines)

    except Exception as e:
        return f"Error getting hyperparameter info: {e}"
