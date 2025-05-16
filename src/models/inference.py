"""
Inference module for the UW Computer Vision project.

This module handles:
- Model inference on new images
- Mask prediction and processing
- Run-length encoding/decoding
- Post-processing of predictions
- Scale bar and arrow detection
- Wavelength analysis

The module provides a comprehensive pipeline for:
- Loading and preprocessing images
- Running model inference
- Post-processing predictions
- Analyzing results
- Saving predictions and visualizations
"""

## IMPORTS
import copy
import csv
import itertools
import os
import re
from math import sqrt
from pathlib import Path

import cv2
import detectron2.data.transforms as T
import easyocr
import imutils
import numpy as np
import pandas as pd
import yaml
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode, Visualizer
from imutils import perspective
from numpy import sqrt
from scipy.ndimage import binary_fill_holes
from scipy.spatial import distance as dist
from skimage.measure import label
from skimage.morphology import dilation, erosion

from data.data_preparation import (choose_and_use_model,
                                   get_trained_model_paths, read_dataset_info,
                                   register_datasets)

# Load config once at the start of your program
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Resolve paths
SPLIT_DIR = Path(config["paths"]["split_dir"]).expanduser().resolve()
CATEGORY_JSON = Path(config["paths"]["category_json"]).expanduser().resolve()


def custom_mapper(dataset_dicts):
    """
    Custom data mapper function for Detectron2. Applies various transformations to the image and annotations.

    Parameters:
    - dataset_dicts (dict): Dictionary containing image and annotation data

    Returns:
    - dict: Updated dictionary with transformed image and annotations
    """
    dataset_dicts = copy.deepcopy(dataset_dicts)  # it will be modified by code below
    image = utils.read_image(dataset_dicts["file_name"], format="BGR")
    transform_list = [
        T.Resize((800, 800)),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomRotation(angle=[90, 90]),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    ]

    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dicts["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dicts.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]

    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dicts["instances"] = utils.filter_empty_instances(instances)
    return dataset_dicts


class CustomTrainer(DefaultTrainer):
    """
    Custom trainer class extending Detectron2's DefaultTrainer to use a custom data mapper.

    This trainer implements:
    - Custom data loading pipeline
    - Data augmentation during training
    - Instance segmentation support
    """

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Builds a custom training data loader with the custom mapper.

        Parameters:
        - cfg (CfgNode): Detectron2 configuration object

        Returns:
        - DataLoader: PyTorch DataLoader with custom mapper
        """
        return build_detection_train_loader(cfg, mapper=custom_mapper)


def get_image_folder_path(base_path=Path.home() / "DATASET" / "INFERENCE"):
    """
    Determines the path to the folder containing images for inference.

    Parameters:
    - base_path (Path): Base path where the INFERENCE folder is located

    Returns:
    - str: Path to the folder containing the images

    Raises:
    - FileNotFoundError: If no images are found in either INFERENCE or INFERENCE/UPLOAD folders
    """
    # Define the two possible paths
    inference_path = os.path.join(base_path)
    upload_path = os.path.join(base_path, "UPLOAD")

    # Check if the INFERENCE folder contains images
    if any(
        os.path.isfile(os.path.join(inference_path, f))
        for f in os.listdir(inference_path)
    ):
        return inference_path

    # Check if the UPLOAD subfolder contains images
    elif os.path.exists(upload_path) and any(
        os.path.isfile(os.path.join(upload_path, f)) for f in os.listdir(upload_path)
    ):
        return upload_path

    # If no images found in either folder, raise an exception
    else:
        raise FileNotFoundError(
            "No images found in INFERENCE or INFERENCE/UPLOAD folders."
        )


def binary_mask_to_rle(binary_mask):
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


def midpoint(ptA, ptB):
    """
    Calculates the midpoint between two points.

    Parameters:
    - ptA (tuple): First point coordinates (x, y)
    - ptB (tuple): Second point coordinates (x, y)

    Returns:
    - tuple: Midpoint coordinates (x, y)
    """
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def GetInference(predictor, im, x_pred, metadata, test_img):
    """
    Performs inference on an image and processes the results.

    Parameters:
    - predictor (object): Model predictor
    - im (numpy.ndarray): Input image
    - x_pred (numpy.ndarray): Previous predictions
    - metadata (object): Dataset metadata
    - test_img (bool): Whether this is a test image

    Returns:
    - tuple: (predictions, visualizations)
    """
    outputs = predictor(im)

    # Get all instances
    inst_out = outputs["instances"]

    # Filter instances by predicted class
    filtered_instances = inst_out[inst_out.pred_classes == x_pred]

    v = Visualizer(
        im[:, :, ::-1], metadata=metadata, scale=1, instance_mode=ColorMode.SEGMENTATION
    )
    out = v.draw_instance_predictions(filtered_instances.to("cpu"))
    cv2.imwrite(
        test_img + "_" + str(x_pred) + "__pred.png", out.get_image()[:, :, ::-1]
    )


def GetCounts(predictor, im, TList, PList):
    """
    Counts instances in the image based on predictions.

    Parameters:
    - predictor (object): Model predictor
    - im (numpy.ndarray): Input image
    - TList (list): List of thresholds
    - PList (list): List of pixel thresholds

    Returns:
    - tuple: (counts, predictions)
    """
    outputs = predictor(im)
    classes = outputs["instances"].pred_classes.to("cpu").numpy()
    TotalCount = sum(classes == 0) + sum(classes == 1)
    TCount = sum(classes == 0)
    PCount = sum(classes == 1)
    TList.append(TCount)
    PList.append(PCount)


def rgb_to_hsv(r, g, b):
    """
    Converts RGB color to HSV.

    Parameters:
    - r (int): Red component (0-255)
    - g (int): Green component (0-255)
    - b (int): Blue component (0-255)

    Returns:
    - tuple: (hue, saturation, value)
    """
    MAX_PIXEL_VALUE = 255.0

    r = r / MAX_PIXEL_VALUE
    g = g / MAX_PIXEL_VALUE
    b = b / MAX_PIXEL_VALUE

    max_val = max(r, g, b)
    min_val = min(r, g, b)
    v = max_val

    if max_val == 0.0:
        s = 0
        h = 0
    elif (max_val - min_val) == 0.0:
        s = 0
        h = 0
    else:
        s = (max_val - min_val) / max_val

        if max_val == r:
            h = 60 * ((g - b) / (max_val - min_val)) + 0
        elif max_val == g:
            h = 60 * ((b - r) / (max_val - min_val)) + 120
        else:
            h = 60 * ((r - g) / (max_val - min_val)) + 240

    if h < 0:
        h = h + 360.0

    h = h / 2
    s = s * MAX_PIXEL_VALUE
    v = v * MAX_PIXEL_VALUE

    return h, s, v


def hue_to_wavelength(hue):
    """
    Converts hue to wavelength in nanometers.

    Parameters:
    - hue (float): Hue value (0-360)

    Returns:
    - float: Wavelength in nanometers
    """
    assert hue >= 0
    assert hue <= 270

    wavelength = 620 - 170 / 270 * hue
    return wavelength


def rgb_to_wavelength(r, g, b):
    """
    Converts RGB color to wavelength.

    Parameters:
    - r (int): Red component (0-255)
    - g (int): Green component (0-255)
    - b (int): Blue component (0-255)

    Returns:
    - float: Wavelength in nanometers
    """
    h, s, v = rgb_to_hsv(r, g, b)
    wavelength = hue_to_wavelength(h)
    return wavelength


def detect_arrows(image):
    """
    Detects arrows in the image.

    Parameters:
    - image (numpy.ndarray): Input image

    Returns:
    - list: List of detected arrow coordinates
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    flow_vectors = []

    for contour in contours:
        if cv2.contourArea(contour) < 100:  # Filter out small contours
            continue

        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

        # Calculate the direction vector
        direction = (vx[0], vy[0])
        flow_vectors.append(direction)

    return flow_vectors


def detect_scale_bar_sem(image):
    """
    Detects scale bars in SEM images.

    Parameters:
    - image (numpy.ndarray): Input image

    Returns:
    - tuple: (scale_bar_length, scale_bar_unit)
    """
    h, w = image.shape[:2]
    roi = image[h // 2 : h, w // 2 : w].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    reader = easyocr.Reader(["en"], verbose=False)
    result = reader.readtext(gray, detail=1, paragraph=False)

    unit_pattern = re.compile(r"(\d+)\s*(nm|µm|um)", re.IGNORECASE)
    scale_value = "0"
    unit = "um"
    scale_center = None

    for bbox, text, _ in result:
        match = unit_pattern.search(text)
        if match:
            scale_value = match.group(1)
            unit = match.group(2).lower()
            x_coords = [int(p[0]) for p in bbox]
            y_coords = [int(p[1]) for p in bbox]
            center_x = sum(x_coords) // len(x_coords)
            center_y = sum(y_coords) // len(y_coords)
            scale_center = (center_x, center_y)
            cv2.rectangle(
                roi,
                (min(x_coords), min(y_coords)),
                (max(x_coords), max(y_coords)),
                (255, 0, 0),
                2,
            )
            break

    bin_img = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bar_px_length = 0
    bar_found = False

    if scale_center:
        for c in contours:
            x, y, w_c, h_c = cv2.boundingRect(c)
            aspect_ratio = w_c / float(h_c)
            if aspect_ratio > 10 and 2 <= h_c <= 15 and y > scale_center[1]:
                left_tick = bin_img[y : y + h_c, max(x - 5, 0) : x + 5]
                right_tick = bin_img[
                    y : y + h_c, x + w_c - 5 : min(x + w_c + 5, roi.shape[1])
                ]
                if (
                    np.count_nonzero(left_tick) > 10
                    and np.count_nonzero(right_tick) > 10
                ):
                    bar_px_length = w_c
                    cv2.line(
                        roi, (x, y + h_c // 2), (x + w_c, y + h_c // 2), (0, 255, 0), 2
                    )
                    bar_found = True
                    break

    unit_multipliers = {"nm": 0.001, "um": 1.0, "µm": 1.0, "mm": 1000.0}
    real_um = float(scale_value) * unit_multipliers.get(unit, 1.0)
    um_per_pixel = real_um / bar_px_length if bar_found else 1.0

    annotated_image = image.copy()
    annotated_image[h // 2 : h, w // 2 : w] = roi

    return um_per_pixel, str(real_um), bar_px_length, annotated_image


def run_inference(dataset_name, output_dir, visualize=False, threshold=0.65):
    """
    Runs inference on a dataset and saves the results.

    Parameters:
    - dataset_name (str): Name of the dataset
    - output_dir (str): Directory to save results
    - visualize (bool): Whether to generate visualizations
    - threshold (float): Confidence threshold for predictions

    The function:
    1. Loads the model and dataset
    2. Processes each image
    3. Generates predictions
    4. Saves results and visualizations
    5. Performs post-processing and analysis

    Returns:
    - None
    """
    dataset_info = read_dataset_info(CATEGORY_JSON)
    register_datasets(dataset_info, dataset_name)

    trained_model_paths = get_trained_model_paths(SPLIT_DIR)
    selected_model_dataset = dataset_name  # User-selected model
    predictor = choose_and_use_model(
        trained_model_paths, selected_model_dataset, threshold
    )

    metadata = MetadataCatalog.get(f"{dataset_name}_train")

    image_folder_path = get_image_folder_path()

    # Path to save outputs
    path = output_dir
    os.makedirs(path, exist_ok=True)
    inpath = image_folder_path
    images_name = [f for f in os.listdir(inpath) if f.endswith(".tif")]

    Img_ID = []
    EncodedPixels = []

    conv = lambda l: " ".join(map(str, l))

    for name in images_name:
        image = cv2.imread(os.path.join(inpath, name))
        outputs = predictor(image)
        masks = postprocess_masks(
            np.asarray(outputs["instances"].to("cpu")._fields["pred_masks"]),
            outputs["instances"].to("cpu")._fields["scores"].numpy(),
            image,
        )

        if masks:
            for i in range(len(masks)):
                Img_ID.append(name.replace(".tif", ""))
                EncodedPixels.append(conv(rle_encoding(masks[i])))

    df = pd.DataFrame({"ImageId": Img_ID, "EncodedPixels": EncodedPixels})
    df.to_csv(os.path.join(path, "R50_flip_results.csv"), index=False, sep=",")

    num_classes = len(MetadataCatalog.get(f"{dataset_name}_train").thing_classes)
    for x_pred in range(num_classes):
        TList = []
        PList = []
        csv_filename = f"results_x_pred_{x_pred}.csv"
        test_img_path = image_folder_path

        with open(csv_filename, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow(
                [
                    "Major axis length",
                    "Minor axis length",
                    "Eccentricity",
                    "C. Length",
                    "C. Width",
                    "Circular eq. diameter",
                    "Acpect ratio",
                    "Circularity",
                    "Chord length",
                    "Ferret diameter",
                    "Roundness",
                    "Sphericity",
                    "Detected scale bar",
                    "File name",
                ]
            )

            for idx, test_img in enumerate(os.listdir(test_img_path), 1):
                print(
                    f"Inferencing image {idx} out of {len(os.listdir(test_img_path))}"
                )
                input_path = os.path.join(test_img_path, test_img)
                im = cv2.imread(input_path)

                h, w = im.shape[:2]

                # Define proportional region where the scale bar and text are located
                x_start = int(w * 0.667)
                y_start = int(h * 0.866)
                x_end = w
                y_end = int(y_start + h * 0.067)

                # Draw detection ROI border in bright red
                cv2.rectangle(im, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

                roi = im[y_start:y_end, x_start:x_end].copy()
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # # Detect text in the image
                # reader = easyocr.Reader(["en"])
                # result = reader.readtext(
                #     gray_roi,
                #     detail=1,  # Get bounding boxes
                #     paragraph=False,
                #     contrast_ths=0.85,
                #     adjust_contrast=0.85,
                #     add_margin=0.25,
                #     width_ths=0.25,
                #     decoder="beamsearch",
                # )
                reader = easyocr.Reader(["en"], verbose=False)
                result = reader.readtext(gray_roi, detail=1, paragraph=False)

                if result:
                    # Extract the first recognized text that looks like a scale (e.g., "500nm")
                    for detection in result:
                        bbox, text, _ = detection
                        text_clean = re.sub("[^0-9]", "", text)  # Extract numeric part
                        if text_clean:
                            pxum_r = text
                            psum = text_clean
                            x_min = int(
                                min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
                            )
                            y_min = int(
                                min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
                            )
                            x_max = int(
                                max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
                            )
                            y_max = int(
                                max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
                            )
                            text_box_center = (
                                (x_min + x_max) // 2,
                                (y_min + y_max) // 2,
                            )
                            break
                    else:
                        pxum_r = ""
                        psum = "0"
                        text_box_center = None
                else:
                    pxum_r = ""
                    psum = "0"
                    text_box_center = None

                # Use Canny edge detection
                edges = cv2.Canny(gray_roi, 50, 150, apertureSize=3)

                lines_list = []
                scale_len = 0
                um_pix = 1

                if text_box_center:
                    # Focus on lines near the detected text box
                    proximity_threshold = (
                        50  # Distance from text box to search for lines
                    )
                    lines = cv2.HoughLinesP(
                        edges,
                        1,
                        np.pi / 180,
                        threshold=100,
                        minLineLength=50,
                        maxLineGap=5,
                    )

                longest_line = None
                max_length = 0

                if lines is not None and text_box_center:
                    for points in lines:
                        x1, y1, x2, y2 = points[0]
                        line_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        dist_to_text = sqrt(
                            (line_center[0] - text_box_center[0]) ** 2
                            + (line_center[1] - text_box_center[1]) ** 2
                        )
                        if dist_to_text < proximity_threshold:
                            length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                            if length > max_length:
                                max_length = length
                                longest_line = (x1, y1, x2, y2)

                if longest_line:
                    x1, y1, x2, y2 = longest_line

                    # Offset from ROI to original image
                    x1_full = x1 + x_start
                    x2_full = x2 + x_start
                    y1_full = y1 + y_start
                    y2_full = y2 + y_start

                    # Draw longest detected line in green
                    cv2.line(im, (x1_full, y1_full), (x2_full, y2_full), (0, 255, 0), 2)

                    scale_len = max_length
                    um_pix = float(psum) / scale_len if scale_len > 0 else 1.0
                else:
                    um_pix = 1
                    psum = "0"

                # end new here #######################

                # um_pix, psum, bar_px_len, im = detect_scale_bar_sem(im)

                GetInference(predictor, im, x_pred, metadata, test_img)
                GetCounts(predictor, im, TList, PList)

                outputs = predictor(im)
                inst_out = outputs["instances"]
                filtered_instances = inst_out[inst_out.pred_classes == x_pred]
                mask_array = filtered_instances.pred_masks.to("cpu").numpy()
                num_instances = mask_array.shape[0]
                mask_array = np.moveaxis(mask_array, 0, -1)
                output = np.zeros_like(im)

                # global_min_wavelength = float("inf")
                # global_max_wavelength = float("-inf")

                # for i in range(im.shape[0]):
                #     for j in range(im.shape[1]):
                #         b, g, r = im[i, j]
                #         wavelength = rgb_to_wavelength(b, g, r)
                #         if wavelength < global_min_wavelength:
                #             global_min_wavelength = wavelength
                #         if wavelength > global_max_wavelength:
                #             global_max_wavelength = wavelength

                for i in range(num_instances):
                    single_output = np.zeros_like(output)
                    mask = mask_array[:, :, i : (i + 1)]
                    single_output = np.where(mask == True, 255, single_output)

                    mask_filename = os.path.join(output_dir, f"mask_{i}.jpg")
                    cv2.imwrite(mask_filename, single_output)

                    single_im_mask = cv2.cvtColor(single_output, cv2.COLOR_BGR2GRAY)
                    single_cnts = cv2.findContours(
                        single_im_mask.copy(),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    single_cnts = imutils.grab_contours(single_cnts)

                    for c in single_cnts:
                        pixelsPerMetric = 1
                        if cv2.contourArea(c) < 100:
                            continue
                        area = cv2.contourArea(c)
                        perimeter = cv2.arcLength(c, True)

                        orig = single_im_mask.copy()
                        box = cv2.minAreaRect(c)
                        box = (
                            cv2.boxPoints(box)
                            if imutils.is_cv2()
                            else cv2.boxPoints(box)
                        )
                        box = np.array(box, dtype="int")
                        box = perspective.order_points(box)
                        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
                        for x, y in box:
                            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
                        (tl, tr, br, bl) = box
                        (tltrX, tltrY) = midpoint(tl, tr)
                        (blbrX, blbrY) = midpoint(bl, br)
                        (tlblX, tlblY) = midpoint(tl, bl)
                        (trbrX, trbrY) = midpoint(tr, br)
                        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                        dimA = dA / pixelsPerMetric
                        dimB = dB / pixelsPerMetric

                        dimArea = area / pixelsPerMetric
                        dimPerimeter = perimeter / pixelsPerMetric
                        diaFeret = max(dimA, dimB)
                        if (dimA and dimB) != 0:
                            Aspect_Ratio = max(dimB, dimA) / min(dimA, dimB)
                        else:
                            Aspect_Ratio = 0
                        Length = min(dimA, dimB) * um_pix
                        Width = max(dimA, dimB) * um_pix

                        CircularED = np.sqrt(4 * area / np.pi) * um_pix
                        Chords = cv2.arcLength(c, True) * um_pix
                        Roundness = 1 / Aspect_Ratio if Aspect_Ratio != 0 else 0
                        Sphericity = (
                            (2 * np.sqrt(np.pi * dimArea)) / dimPerimeter * um_pix
                        )
                        Circularity = (
                            4 * np.pi * (dimArea / (dimPerimeter) ** 2) * um_pix
                        )
                        Feret_diam = diaFeret * um_pix

                        ellipse = cv2.fitEllipse(c)
                        (x, y), (major_axis, minor_axis), angle = ellipse

                        if major_axis > minor_axis:
                            a = major_axis / 2.0
                            b = minor_axis / 2.0
                        else:
                            a = minor_axis / 2.0
                            b = major_axis / 2.0
                        eccentricity = np.sqrt(1 - (b**2 / a**2))

                        major_axis_length = major_axis / pixelsPerMetric * um_pix
                        minor_axis_length = minor_axis / pixelsPerMetric * um_pix

                        csvwriter.writerow(
                            [
                                major_axis_length,
                                minor_axis_length,
                                eccentricity,
                                Length,
                                Width,
                                CircularED,
                                Aspect_Ratio,
                                Circularity,
                                Chords,
                                Feret_diam,
                                Roundness,
                                Sphericity,
                                psum,
                                test_img,
                            ]
                        )
