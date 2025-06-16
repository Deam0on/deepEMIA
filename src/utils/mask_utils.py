"""
Mask Utilities

This module provides utility functions for mask encoding, decoding, and post-processing,
including Run-Length Encoding (RLE) and mask cleaning for segmentation tasks.
"""

import itertools
from src.utils.logger_utils import system_logger

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from skimage.morphology import dilation, erosion


def binary_mask_to_rle(binary_mask: np.ndarray) -> dict:
    """
    Converts a binary mask to Run-Length Encoding (RLE).

    Parameters:
    - binary_mask (numpy.ndarray): 2D binary mask

    Returns:
    - dict: Dictionary with RLE counts and mask size
    """
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")
    for i, (value, elements) in enumerate(
        itertools.groupby(binary_mask.ravel(order="F"))
    ):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def rle_encode(img):
    """
    Encodes a binary image into Run-Length Encoding (RLE).

    Parameters:
    - img (numpy.ndarray): Binary image (1 - mask, 0 - background)

    Returns:
    - str: Run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes Run-Length Encoding (RLE) into a binary mask.

    Parameters:
    - mask_rle (str): Run-length as string formatted (start length)
    - shape (tuple): (height, width) of array to return

    Returns:
    - numpy.ndarray: Binary mask (1 - mask, 0 - background)
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


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


def postprocess_masks(ori_mask, ori_score, image, min_crys_size=2):
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
