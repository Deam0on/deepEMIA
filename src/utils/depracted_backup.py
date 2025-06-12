"""
Deprecated Backup Utilities

This module contains backup or deprecated functions for mask extraction and encoding.
"""

import cv2
import torch
import numpy as np
import logging

from src.utils.mask_utils import rle_encode

# Ensure these are defined or imported from the correct module
# from src.utils.constants import THRESHOLDS, MIN_PIXELS

def get_masks(fn: str, predictor) -> list:
    """
    Gets predicted masks for an image using a trained model.

    Parameters:
    - fn (str): File name of the image
    - predictor (object): Predictor object for inference

    Returns:
    - list: List of RLE encoded masks
    """
    im = cv2.imread(fn)
    if im is None:
        logging.error(f"Failed to read image: {fn}")
        return []
    pred = predictor(im)
    pred_class = torch.mode(pred["instances"].pred_classes)[0]
    take = pred["instances"].scores >= THRESHOLDS[pred_class]
    pred_masks = pred["instances"].pred_masks[take]
    pred_masks = pred_masks.cpu().numpy()
    res = []
    used = np.zeros(im.shape[:2], dtype=int)
    for mask in pred_masks:
        mask = mask * (1 - used)
        if mask.sum() >= MIN_PIXELS[pred_class]:
            used += mask
            res.append(rle_encode(mask))
    return res
