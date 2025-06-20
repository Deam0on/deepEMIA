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

import cv2
import detectron2.data.transforms as T
import imutils
import numpy as np
import os
import time
import cv2
import pandas as pd
import csv
import yaml
from detectron2.data import (DatasetCatalog, MetadataCatalog,
                             build_detection_train_loader)
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode, Visualizer

from src.data.datasets import read_dataset_info, register_datasets
from src.data.models import choose_and_use_model, get_trained_model_paths
from src.utils.logger_utils import system_logger
from src.utils.mask_utils import postprocess_masks, rle_encoding
from src.utils.measurements import calculate_measurements
from src.utils.scalebar_ocr import detect_scale_bar

# Load config once at the start of your program
with open(Path.home() / "uw-com-vision" / "config" / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Resolve paths
SPLIT_DIR = Path(config["paths"]["split_dir"]).expanduser().resolve()
CATEGORY_JSON = Path(config["paths"]["category_json"]).expanduser().resolve()
local_dataset_root = Path(config["paths"]["local_dataset_root"]).expanduser().resolve()


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
    
def is_image_file(filename):
    return filename.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif'))

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


def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def iterative_combo_predictors(predictors, image, iou_threshold=0.7, min_increase=0.25, max_iters=5):
    """
    Run both predictors iteratively, deduplicating after each round,
    until the number of unique masks increases by less than min_increase,
    or if two consecutive iterations find no new masks.
    """

    all_masks = []
    all_scores = []
    all_sources = []
    prev_count = 0
    no_new_mask_iters = 0  # Track consecutive zero-new-mask iterations

    for iteration in range(max_iters):
        new_masks = []
        new_scores = []
        new_sources = []
        for pred_idx, predictor in enumerate(predictors):
            outputs = predictor(image)
            masks = postprocess_masks(
                np.asarray(outputs["instances"].to("cpu")._fields["pred_masks"]),
                outputs["instances"].to("cpu")._fields["scores"].numpy(),
                image,
            )
            scores = outputs["instances"].to("cpu")._fields["scores"].numpy()
            if masks:
                for i, mask in enumerate(masks):
                    new_masks.append(mask)
                    new_scores.append(scores[i])
                    new_sources.append(pred_idx)
        # Combine with previous masks
        all_masks.extend(new_masks)
        all_scores.extend(new_scores)
        all_sources.extend(new_sources)
        # Deduplicate
        unique_masks = []
        unique_scores = []
        unique_sources = []
        for i, mask in enumerate(all_masks):
            if not any(iou(mask, um) > iou_threshold for um in unique_masks):
                unique_masks.append(mask)
                unique_scores.append(all_scores[i])
                unique_sources.append(all_sources[i])
        new_count = len(unique_masks)
        added = new_count - prev_count
        system_logger.info(f"Iteration {iteration + 1}: Added {added} new masks (total: {new_count})")

        # Stop if two consecutive iterations find no new masks
        if added == 0:
            no_new_mask_iters += 1
        else:
            no_new_mask_iters = 0
        if no_new_mask_iters >= 2:
            system_logger.info("Stopping: No new masks found in two consecutive iterations.")
            break

        if prev_count > 0:
            increase = (new_count - prev_count) / max(prev_count, 1)
            if increase < min_increase:
                break
        prev_count = new_count
        all_masks = unique_masks
        all_scores = unique_scores
        all_sources = unique_sources
    return unique_masks, unique_scores, unique_sources


def run_inference(
    dataset_name,
    output_dir,
    visualize=True,
    threshold=0.65,
    draw_id=False,
    dataset_format="json",
    rcnn=50,
    pass_mode="single",
    max_iters=10,  # <-- Add this argument
):
    """
    Runs inference on a dataset and saves the results.

    Parameters:
    - dataset_name (str): Name of the dataset
    - output_dir (str): Directory to save results
    - visualize (bool): Whether to generate visualizations
    - threshold (float): Confidence threshold for predictions
    - rcnn (int): RCNN backbone, 50 or 101

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
    register_datasets(dataset_info, dataset_name, dataset_format=dataset_format)

    # This forces the dataset to be loaded and the metadata to be populated.
    # It's a robust way to ensure the catalog is ready.
    system_logger.info("Forcing metadata population from DatasetCatalog...")
    d = DatasetCatalog.get(f"{dataset_name}_train")
    metadata = MetadataCatalog.get(f"{dataset_name}_train")
    system_logger.info("Metadata populated successfully.")

    if rcnn == "combo":
        predictors = []
        for r in [50, 101]:
            trained_model_paths = get_trained_model_paths(SPLIT_DIR, r)
            predictor, _ = choose_and_use_model(
                trained_model_paths, dataset_name, threshold, metadata, r
            )
            predictors.append(predictor)
    else:
        trained_model_paths = get_trained_model_paths(SPLIT_DIR, rcnn)
        predictor, _ = choose_and_use_model(
            trained_model_paths, dataset_name, threshold, metadata, rcnn
        )
        predictors = [predictor]

    image_folder_path = get_image_folder_path()

    # Path to save outputs
    path = output_dir
    os.makedirs(path, exist_ok=True)
    inpath = image_folder_path
    images_name = [f for f in os.listdir(inpath) if is_image_file(f)]  # <-- changed here

    Img_ID = []
    EncodedPixels = []

    # --- NEW: Track processed images and timing ---
    processed_images = set()
    total_images = len(images_name)
    overall_start_time = time.perf_counter()  # Start timing before first mask

    with open(Path.home() / "uw-com-vision" / "config" / "config.yaml", "r") as f:
        full_config = yaml.safe_load(f)

    # Get the specific ROI config for this dataset, or fall back to the default
    roi_profiles = full_config.get("scale_bar_rois", {})
    roi_config = roi_profiles.get(dataset_name, roi_profiles["default"])
    system_logger.info(
        f"Using scale bar ROI profile for '{dataset_name}': {roi_config}"
    )

    conv = lambda l: " ".join(map(str, l))

    # Store deduplicated masks for each image
    dedup_results = {}

    for idx, name in enumerate(images_name, 1):
        system_logger.info(f"Preparing masks for image {name} ({idx} out of {total_images})")
        image = cv2.imread(os.path.join(inpath, name))
        all_masks = []
        all_scores = []
        all_sources = []

        if rcnn == "combo":
            if pass_mode == "multi":
                unique_masks, unique_scores, unique_sources = iterative_combo_predictors(
                    predictors, image, iou_threshold=0.7, min_increase=0.10, max_iters=max_iters
                )
            else:  # single pass (default/original)
                all_masks = []
                all_scores = []
                all_sources = []
                for pred_idx, predictor in enumerate(predictors):
                    outputs = predictor(image)
                    masks = postprocess_masks(
                        np.asarray(outputs["instances"].to("cpu")._fields["pred_masks"]),
                        outputs["instances"].to("cpu")._fields["scores"].numpy(),
                        image,
                    )
                    scores = outputs["instances"].to("cpu")._fields["scores"].numpy()
                    if masks:
                        for i, mask in enumerate(masks):
                            all_masks.append(mask)
                            all_scores.append(scores[i])
                            all_sources.append(pred_idx)
                # Deduplicate as before
                unique_masks = []
                unique_scores = []
                unique_sources = []
                for i, mask in enumerate(all_masks):
                    if not any(iou(mask, um) > 0.7 for um in unique_masks):
                        unique_masks.append(mask)
                        unique_scores.append(all_scores[i])
                        unique_sources.append(all_sources[i])
        else:
            # ...existing single-model logic...
            all_masks = []
            all_scores = []
            all_sources = []
            for pred_idx, predictor in enumerate(predictors):
                outputs = predictor(image)
                masks = postprocess_masks(
                    np.asarray(outputs["instances"].to("cpu")._fields["pred_masks"]),
                    outputs["instances"].to("cpu")._fields["scores"].numpy(),
                    image,
                )
                scores = outputs["instances"].to("cpu")._fields["scores"].numpy()
                if masks:
                    for i, mask in enumerate(masks):
                        all_masks.append(mask)
                        all_scores.append(scores[i])
                        all_sources.append(pred_idx)
            # Deduplicate as before
            unique_masks = []
            unique_scores = []
            unique_sources = []
            for i, mask in enumerate(all_masks):
                if not any(iou(mask, um) > 0.7 for um in unique_masks):
                    unique_masks.append(mask)
                    unique_scores.append(all_scores[i])
                    unique_sources.append(all_sources[i])

        system_logger.info(
            f"After deduplication: {len(unique_masks)} unique masks for image {name} "
            f"(kept: {unique_sources.count(0)} from R50, {unique_sources.count(1)} from R101)"
        )

        # Save for later use
        dedup_results[name] = {
            "masks": unique_masks,
            "scores": unique_scores,
            "sources": unique_sources,
        }

        processed_images.add(name)

        conv = lambda l: " ".join(map(str, l))
        for i, mask in enumerate(unique_masks):
            Img_ID.append(name.rsplit('.', 1)[0])
            EncodedPixels.append(conv(rle_encoding(mask)))

    overall_elapsed = time.perf_counter() - overall_start_time  # End timing after last mask
    average_time = overall_elapsed / total_images if total_images else 0
    system_logger.info(f"Average mask generation and deduplication time per image: {average_time:.3f} seconds")

    # --- Ensure all images were processed ---
    unprocessed = set(images_name) - processed_images
    if unprocessed:
        system_logger.warning(f"The following images were not processed: {unprocessed}")
    else:
        system_logger.info("All images in the INFERENCE folder were processed.")

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
                system_logger.info(f"Inferencing image {idx} out of {num_images}")

                input_path = os.path.join(test_img_path, test_img)
                im = cv2.imread(input_path)

                psum, um_pix = detect_scale_bar(im, roi_config)

                # Use deduplicated masks for this image
                masks = dedup_results.get(test_img, {}).get("masks", [])
                if not masks:
                    continue

                # --- Visualization: Save overlay image with deduplicated masks ---
                if visualize:
                    vis_img = im.copy()
                    color = (0, 255, 0)
                    for i, mask in enumerate(masks):
                        # Create colored overlay for each mask
                        colored_mask = np.zeros_like(vis_img)
                        colored_mask[mask.astype(bool)] = color
                        vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, 0.5, 0)
                        # Optionally, draw contours and label
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 1)
                        # Draw instance ID at centroid
                        M = cv2.moments(mask.astype(np.uint8))
                        if M["m00"] > 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            cv2.putText(
                                vis_img,
                                str(i + 1),
                                (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                1,
                                cv2.LINE_AA,
                            )
                    vis_save_path = os.path.join(
                        local_dataset_root, f"{test_img}_class_{x_pred}_pred.png"
                    )
                    cv2.imwrite(vis_save_path, vis_img)

                for instance_id, mask in enumerate(masks, 1):
                    binary_mask = (mask > 0).astype(np.uint8) * 255
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

                        measurements = calculate_measurements(
                            c,
                            single_im_mask,
                            um_pix=um_pix,
                            pixelsPerMetric=pixelsPerMetric,
                        )

                        csvwriter.writerow(
                            [
                                instance_id,
                                measurements["major_axis_length"],
                                measurements["minor_axis_length"],
                                measurements["eccentricity"],
                                measurements["Length"],
                                measurements["Width"],
                                measurements["CircularED"],
                                measurements["Aspect_Ratio"],
                                measurements["Circularity"],
                                measurements["Chords"],
                                measurements["Feret_diam"],
                                measurements["Roundness"],
                                measurements["Sphericity"],
                                psum,
                                test_img,
                            ]
                        )
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
                system_logger.info(f"Time taken for image {idx}: {elapsed:.3f} seconds")

    average_time = total_time / num_images if num_images else 0
    system_logger.info(f"Average inference time per image: {average_time:.3f} seconds")

