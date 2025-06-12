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
import os
import time
from pathlib import Path
import logging

import cv2
import detectron2.data.transforms as T
import imutils
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode, Visualizer
from imutils import perspective
from scipy.spatial import distance as dist

from src.data.datasets import (
    read_dataset_info,
    register_datasets,
)
from src.data.models import (
    choose_and_use_model,
    get_trained_model_paths,
)
from src.utils.mask_utils import (
    postprocess_masks,
    rle_encoding,
)
from src.utils.measurements import midpoint
from src.utils.scalebar_ocr import detect_scale_bar

from src.utils.config import get_config

config = get_config()

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


def GetInference(im, filtered_instances, metadata, test_img, x_pred):
    """
    Annotates each instance with its ID at the center of the bounding box
    and saves the annotated image.

    Parameters:
    - im (np.ndarray): Original image
    - filtered_instances (Instances): Instances filtered by class
    - metadata (Metadata): Metadata for label mapping
    - test_img (str): Image name
    - x_pred (int): Class index
    """
    v = Visualizer(
        im[:, :, ::-1],
        metadata=metadata,
        scale=1.0,
        instance_mode=ColorMode.SEGMENTATION,
    )
    out = v.draw_instance_predictions(filtered_instances)
    img_with_boxes = out.get_image()

    for i, box in enumerate(filtered_instances.pred_boxes.tensor):
        x = int((box[0] + box[2]) / 2)
        y = int((box[1] + box[3]) / 2)
        label = f"{i+1}"
        cv2.putText(
            img_with_boxes,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),  # Bright red
            1,
            cv2.LINE_AA,
        )

    save_path = f"{test_img}_class_{x_pred}_pred.png"
    cv2.imwrite(save_path, img_with_boxes[:, :, ::-1])


def GetInferenceNoID(predictor, im, x_pred, metadata, test_img):
    """
    Performs inference on an image and saves the predicted instances.

    Parameters:
    - predictor: The predictor object used for inference.
    - im: The image to perform inference on.
    - x_pred: The class to filter predicted instances by.
    - metadata: Metadata for visualization.
    - test_img: Path to save the test image.

    Returns:
    - None
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


def merge_instances_with_nms(instances_list, iou_threshold=0.5):
    """
    Merge multiple Detectron2 Instances objects using NMS to remove duplicates.
    """
    from detectron2.structures import Instances, Boxes
    import torch

    # Concatenate all predictions
    all_boxes = []
    all_scores = []
    all_classes = []
    all_masks = []
    image_size = None
    for instances in instances_list:
        if len(instances) == 0:
            continue
        if image_size is None:
            image_size = instances.image_size
        all_boxes.append(instances.pred_boxes.tensor)
        all_scores.append(instances.scores)
        all_classes.append(instances.pred_classes)
        if hasattr(instances, "pred_masks"):
            all_masks.append(instances.pred_masks)

    if not all_boxes:
        return Instances(image_size=(0, 0))

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    classes = torch.cat(all_classes)
    if all_masks:
        masks = torch.cat(all_masks)
    else:
        masks = None

    # NMS per class
    keep = []
    for cls in classes.unique():
        inds = (classes == cls).nonzero(as_tuple=True)[0]
        cls_boxes = boxes[inds]
        cls_scores = scores[inds]
        nms_inds = torchvision.ops.nms(cls_boxes, cls_scores, iou_threshold)
        keep.extend(inds[nms_inds].tolist())

    # Build merged Instances
    merged = Instances(image_size if image_size else (0, 0))
    merged.pred_boxes = Boxes(boxes[keep])
    merged.scores = scores[keep]
    merged.pred_classes = classes[keep]
    if masks is not None:
        merged.pred_masks = masks[keep]
    return merged


def deduplicate_instances(instances_list, iou_threshold=0.5):
    """
    Combine two Detectron2 Instances objects, removing duplicates by mask IoU.
    """
    if not instances_list or all(len(inst) == 0 for inst in instances_list):
        return Instances(image_size=(0, 0))

    # Concatenate all predictions
    all_boxes = []
    all_scores = []
    all_classes = []
    all_masks = []
    image_size = None
    for instances in instances_list:
        if len(instances) == 0:
            continue
        if image_size is None:
            image_size = instances.image_size
        all_boxes.append(instances.pred_boxes.tensor)
        all_scores.append(instances.scores)
        all_classes.append(instances.pred_classes)
        if hasattr(instances, "pred_masks"):
            all_masks.append(instances.pred_masks)

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    classes = torch.cat(all_classes)
    masks = torch.cat(all_masks) if all_masks else None

    # NMS per class (by box)
    keep = []
    for cls in classes.unique():
        inds = (classes == cls).nonzero(as_tuple=True)[0]
        cls_boxes = boxes[inds]
        cls_scores = scores[inds]
        nms_inds = torchvision.ops.nms(cls_boxes, cls_scores, iou_threshold)
        keep.extend(inds[nms_inds].tolist())

    merged = Instances(image_size)
    merged.pred_boxes = Boxes(boxes[keep])
    merged.scores = scores[keep]
    merged.pred_classes = classes[keep]
    if masks is not None:
        merged.pred_masks = masks[keep]
    return merged


def load_predictor(config_file, model_suffix, output_dir, dataset_name, threshold):
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.WEIGHTS = os.path.join(output_dir, dataset_name, f"rcnn_{model_suffix}", f"model_final_{model_suffix}.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    return predictor


def run_inference(
    dataset_name: str,
    output_dir: str,
    visualize: bool = False,
    threshold: float = 0.65,
    draw_id: bool = False,
    dataset_format: str = "json",
    rcnn: str = "combo",
) -> None:
    """
    Runs inference on a dataset and saves the results.

    Parameters:
    - dataset_name (str): Name of the dataset
    - output_dir (str): Directory to save results
    - visualize (bool): Whether to generate visualizations
    - threshold (float): Confidence threshold for predictions
    - draw_id (bool): Whether to draw instance IDs
    - dataset_format (str): Dataset annotation format
    - rcnn (str): Backbone to use: "50", "101", or "combo"

    Returns:
    - None
    """
    dataset_info = read_dataset_info(CATEGORY_JSON)
    register_datasets(dataset_info, dataset_name, dataset_format=dataset_format)

    logging.info("Forcing metadata population from DatasetCatalog...")
    d = DatasetCatalog.get(f"{dataset_name}_train")
    metadata = MetadataCatalog.get(f"{dataset_name}_train")
    logging.info("Metadata populated successfully.")

    # Prepare both predictors
    predictor_r101 = load_predictor("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", "r101", output_dir, dataset_name, threshold)
    predictor_r50 = load_predictor("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", "r50", output_dir, dataset_name, threshold)

    image_folder_path = get_image_folder_path()
    path = output_dir
    os.makedirs(path, exist_ok=True)
    inpath = image_folder_path
    images_name = [f for f in os.listdir(inpath) if f.endswith(".tif")]

    Img_ID = []
    EncodedPixels = []

    with open(Path.home() / "uw-com-vision" / "config" / "config.yaml", "r") as f:
        full_config = yaml.safe_load(f)

    roi_profiles = full_config.get("scale_bar_rois", {})
    roi_config = roi_profiles.get(dataset_name, roi_profiles["default"])
    logging.info(f"Using scale bar ROI profile for '{dataset_name}': {roi_config}")

    conv = lambda l: " ".join(map(str, l))

    for name in images_name:
        logging.info(f"Preparing masks for image {name}")
        image = cv2.imread(os.path.join(inpath, name))

        # Run both predictors
        outputs_r101 = predictor_r101(image)
        outputs_r50 = predictor_r50(image)
        instances_r101 = outputs_r101["instances"].to("cpu")
        instances_r50 = outputs_r50["instances"].to("cpu")

        # Combine and deduplicate
        merged_instances = deduplicate_instances([instances_r101, instances_r50], iou_threshold=0.5)

        if not hasattr(merged_instances, "pred_masks") or len(merged_instances) == 0:
            logging.warning(f"No masks predicted for image {name}. Skipping.")
            continue

        masks = postprocess_masks(
            merged_instances.pred_masks.numpy(),
            merged_instances.scores.numpy(),
            image,
        )

        if masks:
            for i in range(len(masks)):
                Img_ID.append(name.replace(".tif", ""))
                EncodedPixels.append(conv(rle_encoding(masks[i])))

    df = pd.DataFrame({"ImageId": Img_ID, "EncodedPixels": EncodedPixels})
    df.to_csv(os.path.join(path, "results_combo.csv"), index=False, sep=",")

    num_classes = len(MetadataCatalog.get(f"{dataset_name}_train").thing_classes)
    for x_pred in range(num_classes):
        TList = []
        PList = []
        csv_filename = f"results_x_pred_{x_pred}_{rcnn}.csv"
        test_img_path = image_folder_path

        with open(csv_filename, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow(
                [
                    "Instance_ID",
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

            image_list = os.listdir(test_img_path)
            num_images = len(image_list)
            total_time = 0

            for idx, test_img in enumerate(image_list, 1):
                start_time = time.perf_counter()
                logging.info(f"Inferencing image {idx} out of {num_images}")

                input_path = os.path.join(test_img_path, test_img)
                im = cv2.imread(input_path)

                psum, um_pix = detect_scale_bar(im, roi_config)

                all_instances = []
                for predictor in predictors:
                    outputs = predictor(im)
                    all_instances.append(outputs["instances"].to("cpu"))

                if rcnn == "combo":
                    merged_instances = merge_instances_with_nms(all_instances)
                else:
                    merged_instances = all_instances[0]

                filtered_instances = merged_instances[merged_instances.pred_classes == x_pred]

                if draw_id:
                    GetInference(im, filtered_instances, metadata, test_img, x_pred)
                else:
                    # Use the first predictor for visualization if needed
                    GetInferenceNoID(predictors[0], im, x_pred, metadata, test_img)
                GetCounts(predictors[0], im, TList, PList)

                pred_masks = filtered_instances.pred_masks.numpy()
                pred_boxes = filtered_instances.pred_boxes.tensor.numpy()
                scores = filtered_instances.scores.numpy()
                num_instances = len(filtered_instances)

                output = np.zeros_like(im)

                for i in range(num_instances):
                    instance_id = i + 1
                    binary_mask = (pred_masks[i] > 0).astype(
                        np.uint8
                    ) * 255  # ensure binary uint8
                    single_im_mask = binary_mask.copy()
                    mask_3ch = np.stack([single_im_mask] * 3, axis=-1)
                    mask_filename = os.path.join(output_dir, f"mask_{instance_id}.jpg")
                    cv2.imwrite(mask_filename, mask_3ch)

                    single_cnts = cv2.findContours(
                        single_im_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
                                instance_id,
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
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
                logging.info(f"Time taken for image {idx}: {elapsed:.3f} seconds")

    average_time = total_time / num_images if num_images else 0
    logging.info(f"Average inference time per image: {average_time:.3f} seconds")
