"""
Mask Utilities

This module provides utility functions for mask encoding, decoding, and post-processing,
including Run-Length Encoding (RLE) and mask cleaning for segmentation tasks.
"""

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from skimage.morphology import dilation, erosion

from src.utils.logger_utils import system_logger
from src.utils.constants import DefaultThresholds


def rle_encoding(x):
    """
    Encodes a binary array into run-length encoding.

    Parameters:
    - x (numpy.ndarray): Binary array (1 - mask, 0 - background)

    Returns:
    - list: Run-length encoding list
    """
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def postprocess_masks(ori_mask, ori_score, image, min_crys_size=None):
    """
    Post-processes masks by removing overlaps, filling small holes, and smoothing boundaries.

    Parameters:
    - ori_mask (numpy.ndarray): Original mask predictions
    - ori_score (numpy.ndarray): Confidence scores for the masks
    - image (numpy.ndarray): Original image for reference
    - min_crys_size (int): Minimum size for valid masks

    Returns:
    - list: List of processed masks
    """
    if min_crys_size is None:
        min_crys_size = DefaultThresholds.MIN_CRYSTAL_SIZE

    image = image[:, :, ::-1]
    height, width = image.shape[:2]

    score_threshold = 0.5

    if len(ori_mask) == 0 or ori_score.all() < score_threshold:
        return []

    keep_ind = np.where(np.sum(ori_mask, axis=(0, 1)) > min_crys_size)[0]
    if len(keep_ind) < len(ori_mask):
        if keep_ind.shape[0] != 0:
            ori_mask = ori_mask[: keep_ind.shape[0]]
            ori_score = ori_score[: keep_ind.shape[0]]
        else:
            return []

    overlap = np.zeros([height, width])
    masks = []

    # Removes overlaps from masks with lower scores
    for i in range(len(ori_mask)):
        mask = binary_fill_holes(ori_mask[i]).astype(np.uint8)
        mask = erosion(dilation(mask))
        overlap += mask
        mask[overlap > 1] = 0
        out_label = label(mask)
        if out_label.max() > 1:
            mask[:] = 0
        masks.append(mask)

    return masks
