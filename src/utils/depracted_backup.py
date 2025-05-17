import cv2
import torch
from src.utils.mask_utils import rle_encode

def get_masks(fn, predictor):
    """
    Gets predicted masks for an image using a trained model.

    Parameters:
    - fn (str): File name of the image
    - predictor (object): Predictor object for inference

    Returns:
    - list: List of RLE encoded masks
    """
    im = cv2.imread(fn)
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