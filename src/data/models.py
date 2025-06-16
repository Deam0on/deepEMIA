"""
Model and data utility module for the UW Computer Vision project.

This module handles:
- Dataset splitting into train and test sets
- Dataset registration for Detectron2
- Model loading and preparation (including quantized models)
- Dataset information management

The module integrates with Detectron2 for computer vision tasks and provides
utilities for handling various data formats and model types.
"""

from src.utils.logger_utils import system_logger
import os
from pathlib import Path

import torch
import yaml
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

from src.utils.config import get_config

config = get_config()

# Constant paths
SPLIT_DIR = Path(config["paths"]["split_dir"]).expanduser().resolve()
CATEGORY_JSON = Path(config["paths"]["category_json"]).expanduser().resolve()


def get_trained_model_paths(base_dir: str, rcnn: int = 101) -> dict:
    """
    Retrieves paths to trained models in a given base directory.

    Parameters:
    - base_dir (str): Directory containing trained models
    - rcnn (int): RCNN backbone, 50 or 101

    Returns:
    - dict: Dictionary with dataset names as keys and model paths as values
    """
    model_paths = {}
    suffix = f"r{rcnn}"
    for dataset_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, dataset_name, f"rcnn_r{rcnn}")
        model_path = os.path.join(model_dir, f"model_final_{suffix}.pth")
        if os.path.exists(model_path):
            model_paths[dataset_name] = model_path
    return model_paths


def load_model(cfg, model_path: str, dataset_name: str, is_quantized: bool = False):
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
            system_logger.error(f"Failed to load or initialize quantized model: {e}")
            raise RuntimeError("Quantized model load failed.")

    # fallback or standard model
    cfg.MODEL.WEIGHTS = model_path
    metadata = MetadataCatalog.get(f"{dataset_name}_train")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)

    return DefaultPredictor(cfg)


def choose_and_use_model(
    model_paths: dict, dataset_name: str, threshold: float, metadata, rcnn: int = 101):
    
    """
    Chooses and loads the appropriate model for a given dataset.

    Parameters:
    - model_paths (dict): Dictionary of available model paths
    - dataset_name (str): Name of the dataset
    - threshold (float): Confidence threshold for predictions
    - metadata: Metadata object for the dataset

    Returns:
    - tuple: (predictor, metadata) The loaded model and its metadata
    """
    if dataset_name not in model_paths:
        system_logger.error(f"No model found for dataset {dataset_name}")
        return None, None

    base_model_path = model_paths[dataset_name]
    quantized_model_path = base_model_path.replace(
        "model_final.pth", "model_final_quantized.pth"
    )

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            f"COCO-InstanceSegmentation/mask_rcnn_R_{rcnn}_FPN_3x.yaml"
        )
    )
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    # Get the metadata here, after registration is complete.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)

    predictor = None
    if not torch.cuda.is_available() and os.path.exists(quantized_model_path):
        try:
            system_logger.info(f"Trying quantized model for {dataset_name}")
            predictor = load_model(
                cfg, quantized_model_path, dataset_name, is_quantized=True
            )
        except RuntimeError:
            system_logger.warning(f"Falling back to standard model for {dataset_name}")
            predictor = load_model(
                cfg, base_model_path, dataset_name, is_quantized=False
            )
    else:
        system_logger.info(f"Using standard model for {dataset_name}")
        predictor = load_model(cfg, base_model_path, dataset_name, is_quantized=False)

    return predictor, metadata
