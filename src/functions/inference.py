"""
Inference module for the deepEMIA project.

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
import gc
import os
import time
from pathlib import Path

import cv2
import detectron2.data.transforms as T
import imutils
import numpy as np
import pandas as pd
import torch
import yaml
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
)
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode, Visualizer
from skimage.morphology import disk, erosion, dilation
from scipy.ndimage import binary_fill_holes

from src.data.datasets import read_dataset_info, register_datasets
from src.data.models import choose_and_use_model, get_trained_model_paths
from src.utils.logger_utils import system_logger
from src.utils.mask_utils import postprocess_masks, rle_encoding
from src.utils.measurements import calculate_measurements
from src.utils.scalebar_ocr import detect_scale_bar

# Load config once at the start of your program
with open(Path.home() / "deepEMIA" / "config" / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

measure_contrast_distribution = config.get("measure_contrast_distribution", False)

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
    """
    Checks if a filename corresponds to a supported image format.

    Parameters:
    - filename (str): The filename to check

    Returns:
    - bool: True if the file is a supported image format, False otherwise
    """
    return filename.lower().endswith(
        (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif")
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


def iou(mask1, mask2):
    """
    Calculates the Intersection over Union (IoU) of two binary masks.

    Parameters:
    - mask1 (numpy.ndarray): First binary mask
    - mask2 (numpy.ndarray): Second binary mask

    Returns:
    - float: IoU score between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def iterative_combo_predictors(
    predictors, image, iou_threshold=0.7, min_increase=0.25, max_iters=5
):
    """
    Run both predictors iteratively with updated stopping conditions.
    """
    all_masks = []
    all_scores = []
    all_sources = []
    prev_count = 0
    no_new_mask_iters = 0

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
        system_logger.info(
            f"Iteration {iteration + 1}: Added {added} new masks (total: {new_count})"
        )

        # UPDATED EARLY STOPPING CONDITIONS (same as single predictor)

        # Track consecutive iterations with no new masks
        if added == 0:
            no_new_mask_iters += 1
        else:
            no_new_mask_iters = 0

        # Stop if no new masks for 2 consecutive iterations
        if no_new_mask_iters >= 2:
            system_logger.info(
                "Stopping: No new masks found in two consecutive iterations."
            )
            break

        # Check stopping conditions only if we have enough masks
        if new_count >= 10:  # Must have at least 10 masks
            if iteration >= 2:  # Allow at least 3 iterations
                # Calculate required increase (25% of current total)
                required_increase = max(1, int(prev_count * min_increase))

                if added < required_increase:
                    system_logger.info(
                        f"Stopping: Added {added} masks < required {required_increase} "
                        f"({min_increase:.0%} of {prev_count} existing masks). "
                        f"Total masks: {new_count}"
                    )
                    break
        else:
            # Continue if we don't have enough masks yet
            system_logger.info(
                f"Continuing: Only {new_count} masks (need at least 10 before considering early stop)"
            )

        prev_count = new_count
        all_masks = unique_masks
        all_scores = unique_scores
        all_sources = unique_sources

    system_logger.info(
        f"Combo inference completed with {len(unique_masks)} total masks"
    )
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
    max_iters=10,
    use_class_specific=None,
):
    """
    Runs inference on a dataset and saves the results.
    Includes memory optimization to prevent OOM errors.
    """
    dataset_info = read_dataset_info(CATEGORY_JSON)
    register_datasets(dataset_info, dataset_name, dataset_format=dataset_format)

    # Force metadata population
    system_logger.info("Forcing metadata population from DatasetCatalog...")
    d = DatasetCatalog.get(f"{dataset_name}_train")
    metadata = MetadataCatalog.get(f"{dataset_name}_train")
    system_logger.info("Metadata populated successfully.")

    # Memory optimization: Clear unnecessary data
    del d
    gc.collect()

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
    images_name = [f for f in os.listdir(inpath) if is_image_file(f)]

    # Memory optimization: Process in smaller batches
    batch_size = min(10, len(images_name))
    system_logger.info(
        f"Processing {len(images_name)} images in batches of {batch_size}"
    )

    Img_ID = []
    EncodedPixels = []

    # Track processed images and timing
    processed_images = set()
    total_images = len(images_name)
    overall_start_time = time.perf_counter()

    with open(Path.home() / "deepEMIA" / "config" / "config.yaml", "r") as f:
        full_config = yaml.safe_load(f)

    # Get the specific ROI config for this dataset
    roi_profiles = full_config.get("scale_bar_rois", {})
    roi_config = roi_profiles.get(dataset_name, roi_profiles["default"])
    system_logger.info(
        f"Using scale bar ROI profile for '{dataset_name}': {roi_config}"
    )

    conv = lambda l: " ".join(map(str, l))

    # Store deduplicated masks AND class predictions for each image
    dedup_results = {}

    # Process images in batches to manage memory
    for batch_start in range(0, len(images_name), batch_size):
        batch_end = min(batch_start + batch_size, len(images_name))
        batch_names = images_name[batch_start:batch_end]

        system_logger.info(
            f"Processing batch {batch_start//batch_size + 1}/{(len(images_name) + batch_size - 1)//batch_size}: images {batch_start + 1}-{batch_end}"
        )

        for idx_in_batch, name in enumerate(batch_names):
            idx = batch_start + idx_in_batch + 1
            system_logger.info(
                f"Preparing masks for image {name} ({idx} out of {total_images})"
            )

            # Memory optimization: Load image and clear previous image data
            image_path = os.path.join(inpath, name)
            image = cv2.imread(image_path)

            if image is None:
                system_logger.warning(f"Could not load image: {image_path}")
                continue

            # Load configuration for inference approach
            if use_class_specific is not None:
                # Use parameter passed to function (from CLI)
                use_class_specific_inference = use_class_specific
            else:
                # Use config file setting
                inference_settings = config.get("inference_settings", {})
                use_class_specific_inference = inference_settings.get(
                    "use_class_specific_inference", False
                )

            if use_class_specific_inference:
                # NEW: Class-specific inference approach
                system_logger.info(f"Running class-specific inference for image {name}")

                # Get number of classes from metadata
                num_classes = len(metadata.thing_classes)

                all_unique_masks = []
                all_unique_scores = []
                all_unique_classes = []

                # Process each class separately
                for target_class in range(num_classes):
                    class_name = metadata.thing_classes[target_class]
                    system_logger.info(
                        f"Processing class {target_class} ({class_name})..."
                    )

                    # Class-specific parameters
                    if target_class == 0:  # Large particles
                        confidence_thresh = 0.5
                        iou_thresh = 0.7
                    elif target_class == 1:  # Small particles
                        confidence_thresh = 0.3  # Lower threshold for small particles
                        iou_thresh = 0.5  # More lenient overlap
                    else:
                        confidence_thresh = 0.4
                        iou_thresh = 0.6

                    if pass_mode == "multi" and target_class == 1:
                        # Multi-scale for small particles only
                        class_masks, class_scores, class_classes = (
                            run_multiscale_class_inference(
                                predictors[0],
                                image,
                                target_class,
                                confidence_thresh,
                                max_iters,
                            )
                        )
                    else:
                        # Single pass for this class
                        class_masks, class_scores, class_classes = (
                            run_class_specific_inference(
                                predictors[0],
                                image,
                                target_class,
                                confidence_thresh,
                                iou_thresh,
                            )
                        )

                    system_logger.info(
                        f"Class {target_class}: Found {len(class_masks)} instances"
                    )

                    # Add to combined results
                    all_unique_masks.extend(class_masks)
                    all_unique_scores.extend(class_scores)
                    all_unique_classes.extend(class_classes)

                # Final cross-class deduplication (optional, more lenient)
                final_masks = []
                final_scores = []
                final_classes = []

                for i, (mask, score, cls) in enumerate(
                    zip(all_unique_masks, all_unique_scores, all_unique_classes)
                ):
                    # Only check against same class or very high overlap with different class
                    is_duplicate = False
                    for j, (existing_mask, existing_cls) in enumerate(
                        zip(final_masks, final_classes)
                    ):
                        overlap = iou(mask, existing_mask)

                        if cls == existing_cls:
                            # Same class: normal threshold
                            threshold_val = 0.7
                        else:
                            # Different class: only remove if very high overlap
                            threshold_val = 0.9

                        if overlap > threshold_val:
                            # Keep the one with higher confidence
                            if score > final_scores[j]:
                                final_masks[j] = mask
                                final_scores[j] = score
                                final_classes[j] = cls
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        final_masks.append(mask)
                        final_scores.append(score)
                        final_classes.append(cls)

                unique_masks = final_masks
                unique_scores = final_scores
                unique_classes = final_classes
                unique_sources = [0] * len(unique_masks)  # All from same source

            else:
                # ORIGINAL: Traditional inference approach
                if rcnn == "combo":
                    if pass_mode == "multi":
                        unique_masks, unique_scores, unique_sources = (
                            iterative_combo_predictors(
                                predictors,
                                image,
                                iou_threshold=0.7,
                                min_increase=0.10,
                                max_iters=max_iters,
                            )
                        )
                        # For combo mode, we need to get class predictions for each mask
                        unique_classes = []
                        # Use the first predictor to get class information
                        temp_outputs = predictors[0](image)
                        temp_classes = (
                            temp_outputs["instances"]
                            .to("cpu")
                            ._fields["pred_classes"]
                            .numpy()
                        )

                        # Assign classes to masks (simplified - assumes same order)
                        for i in range(len(unique_masks)):
                            class_idx = temp_classes[i] if i < len(temp_classes) else 0
                            unique_classes.append(class_idx)

                        del temp_outputs
                        gc.collect()

                    else:  # single pass
                        all_masks = []
                        all_scores = []
                        all_sources = []
                        all_classes = []

                        for pred_idx, predictor in enumerate(predictors):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            outputs = predictor(image)
                            masks = postprocess_masks(
                                np.asarray(
                                    outputs["instances"].to("cpu")._fields["pred_masks"]
                                ),
                                outputs["instances"]
                                .to("cpu")
                                ._fields["scores"]
                                .numpy(),
                                image,
                            )
                            scores = (
                                outputs["instances"].to("cpu")._fields["scores"].numpy()
                            )
                            classes = (
                                outputs["instances"]
                                .to("cpu")
                                ._fields["pred_classes"]
                                .numpy()
                            )

                            del outputs
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            if masks:
                                for i, mask in enumerate(masks):
                                    all_masks.append(mask)
                                    all_scores.append(scores[i])
                                    all_sources.append(pred_idx)
                                    all_classes.append(classes[i])

                            del masks, scores, classes
                            gc.collect()
                        # Deduplicate while preserving class information
                        unique_masks = []
                        unique_scores = []
                        unique_sources = []
                        unique_classes = []

                        for i, mask in enumerate(all_masks):
                            if not any(iou(mask, um) > 0.7 for um in unique_masks):
                                unique_masks.append(mask)
                                unique_scores.append(all_scores[i])
                                unique_sources.append(all_sources[i])
                                unique_classes.append(all_classes[i])

                        del all_masks, all_scores, all_sources, all_classes
                        gc.collect()
                else:
                    # Single model logic with class prediction
                    if pass_mode == "multi":
                        system_logger.info(
                            f"Running iterative single model (R{rcnn}) inference for {max_iters} iterations"
                        )
                        unique_masks, unique_scores, unique_sources, unique_classes = (
                            iterative_single_predictor_with_classes(
                                predictors[0],  # Single predictor
                                image,
                                iou_threshold=0.7,
                                min_increase=0.10,
                                max_iters=max_iters,
                            )
                        )
                    else:  # single pass
                        system_logger.info(f"Running single-pass R{rcnn} inference")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        outputs = predictors[0](image)
                        masks = postprocess_masks(
                            np.asarray(
                                outputs["instances"].to("cpu")._fields["pred_masks"]
                            ),
                            outputs["instances"].to("cpu")._fields["scores"].numpy(),
                            image,
                        )
                        scores = (
                            outputs["instances"].to("cpu")._fields["scores"].numpy()
                        )
                        classes = (
                            outputs["instances"]
                            .to("cpu")
                            ._fields["pred_classes"]
                            .numpy()
                        )

                        del outputs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        unique_masks = masks if masks else []
                        unique_scores = scores.tolist() if len(scores) > 0 else []
                        unique_sources = [0] * len(unique_masks)
                        unique_classes = classes.tolist() if len(classes) > 0 else []

                        del masks, scores, classes
                        gc.collect()

            # FIXED: Log results with class distribution for ALL classes
            class_counts = {}
            for cls in unique_classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1

            # Log all detected classes
            class_summary = ", ".join(
                [f"class {cls}: {count}" for cls, count in sorted(class_counts.items())]
            )
            if class_summary:
                system_logger.info(
                    f"After processing: {len(unique_masks)} unique masks for image {name} ({class_summary})"
                )
            else:
                system_logger.info(
                    f"After processing: {len(unique_masks)} unique masks for image {name} (no classes detected)"
                )

            # Save for later use - now including classes
            dedup_results[name] = {
                "masks": unique_masks,
                "scores": unique_scores,
                "sources": unique_sources,
                "classes": unique_classes,
            }

            processed_images.add(name)

            # Memory optimization: Encode masks immediately and clear image data
            for i, mask in enumerate(unique_masks):
                Img_ID.append(name.rsplit(".", 1)[0])
                EncodedPixels.append(conv(rle_encoding(mask)))

            # Memory optimization: Clear image and mask data after processing
            del image, unique_masks, unique_scores, unique_sources, unique_classes
            gc.collect()

    overall_elapsed = time.perf_counter() - overall_start_time
    average_time = overall_elapsed / total_images if total_images else 0
    system_logger.info(
        f"Average mask generation and deduplication time per image: {average_time:.3f} seconds"
    )

    # Ensure all images were processed
    unprocessed = set(images_name) - processed_images
    if unprocessed:
        system_logger.warning(f"The following images were not processed: {unprocessed}")
    else:
        system_logger.info("All images in the INFERENCE folder were processed.")

    # Save RLE results
    df = pd.DataFrame({"ImageId": Img_ID, "EncodedPixels": EncodedPixels})
    df.to_csv(os.path.join(path, "R50_flip_results.csv"), index=False, sep=",")

    # Memory optimization: Clear large dataframes
    del df, Img_ID, EncodedPixels
    gc.collect()

    # MODIFIED: Single measurements file with class information
    system_logger.info("Starting measurements phase...")

    csv_filename = os.path.join(output_dir, "measurements_results.csv")
    test_img_path = image_folder_path

    # Define colors for different classes (BGR format for OpenCV)
    class_colors = [
        (0, 255, 0),  # Green for class 0
        (255, 0, 0),  # Blue for class 1
        (0, 0, 255),  # Red for class 2
        (255, 255, 0),  # Cyan for class 3
        (255, 0, 255),  # Magenta for class 4
        (0, 255, 255),  # Yellow for class 5
        (128, 0, 128),  # Purple for class 6
        (255, 165, 0),  # Orange for class 7
    ]

    with open(csv_filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)

        # ADDED: Class column to CSV header
        csvwriter.writerow(
            [
                "Instance_ID",
                "Class",
                "Class_Name",
                "Major axis length",
                "Minor axis length",
                "Eccentricity",
                "C. Length",
                "C. Width",
                "Circular eq. diameter",
                "Aspect ratio",
                "Circularity",
                "Chord length",
                "Ferret diameter",
                "Roundness",
                "Sphericity",
                "Contrast d10",
                "Contrast d50",
                "Contrast d90",
                "Detected scale bar",
                "File name",
            ]
        )

        image_list = [f for f in os.listdir(test_img_path) if is_image_file(f)]
        num_images = len(image_list)
        total_time = 0

        # Process images in smaller batches for measurements
        measurement_batch_size = min(5, len(image_list))

        for batch_start in range(0, len(image_list), measurement_batch_size):
            batch_end = min(batch_start + measurement_batch_size, len(image_list))
            batch_images = image_list[batch_start:batch_end]

            system_logger.info(
                f"Processing measurements batch {batch_start//measurement_batch_size + 1}/{(len(image_list) + measurement_batch_size - 1)//measurement_batch_size}"
            )

            for idx_in_batch, test_img in enumerate(batch_images):
                idx = batch_start + idx_in_batch + 1
                start_time = time.perf_counter()
                system_logger.info(
                    f"Processing measurements for image {idx} out of {num_images}: {test_img}"
                )

                input_path = os.path.join(test_img_path, test_img)
                im = cv2.imread(input_path)

                if im is None:
                    system_logger.warning(
                        f"Could not load image for measurements: {input_path}"
                    )
                    continue

                psum, um_pix = detect_scale_bar(im, roi_config)

                # Use deduplicated masks and classes for this image
                image_data = dedup_results.get(test_img, {})
                masks = image_data.get("masks", [])
                classes = image_data.get("classes", [])

                if not masks:
                    system_logger.info(
                        f"No masks found for image {test_img}, skipping measurements"
                    )
                    continue

                system_logger.info(
                    f"Processing {len(masks)} masks for image {test_img}"
                )

                # Track measurements statistics
                measurements_written = 0
                masks_filtered = 0
                class_measurements = {}

                # MODIFIED: Single visualization per image with color-coded classes
                if visualize:
                    vis_img = im.copy()

                    for i, (mask, cls) in enumerate(zip(masks, classes)):
                        # Use class-specific color
                        color = class_colors[cls % len(class_colors)]

                        # Create colored overlay for each mask
                        colored_mask = np.zeros_like(vis_img)
                        colored_mask[mask.astype(bool)] = color
                        vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, 0.5, 0)

                        # Draw contours
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        cv2.drawContours(vis_img, contours, -1, color, 2)

                        # Draw instance ID and class at centroid
                        M = cv2.moments(mask.astype(np.uint8))
                        if M["m00"] > 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])

                            # Get class name
                            class_name = (
                                metadata.thing_classes[cls]
                                if cls < len(metadata.thing_classes)
                                else f"class_{cls}"
                            )

                            # Draw instance ID
                            cv2.putText(
                                vis_img,
                                f"{i + 1}",
                                (cX, cY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),  # White text
                                2,
                                cv2.LINE_AA,
                            )

                            # Draw class name
                            cv2.putText(
                                vis_img,
                                class_name,
                                (cX, cY + 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (255, 255, 255),  # White text
                                1,
                                cv2.LINE_AA,
                            )

                        del colored_mask

                    # Single PNG per image
                    vis_save_path = os.path.join(
                        output_dir, f"{test_img}_predictions.png"
                    )
                    cv2.imwrite(vis_save_path, vis_img)
                    del vis_img
                    gc.collect()

                # Process each mask for measurements
                for instance_id, (mask, cls) in enumerate(zip(masks, classes), 1):
                    binary_mask = (mask > 0).astype(np.uint8) * 255
                    single_im_mask = binary_mask.copy()

                    # Save individual mask image with class info
                    mask_3ch = np.stack([single_im_mask] * 3, axis=-1)
                    class_name = (
                        metadata.thing_classes[cls]
                        if cls < len(metadata.thing_classes)
                        else f"class_{cls}"
                    )
                    mask_filename = os.path.join(
                        output_dir, f"{test_img}_mask_{instance_id}_{class_name}.jpg"
                    )
                    cv2.imwrite(mask_filename, mask_3ch)

                    single_cnts = cv2.findContours(
                        single_im_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    single_cnts = imutils.grab_contours(single_cnts)

                    # Track measurements for this mask
                    mask_measurements = 0
                    total_contours = len(single_cnts)  # Add this line

                    for c in single_cnts:
                        pixelsPerMetric = 1
                        contour_area = cv2.contourArea(c)

                        # ADAPTIVE: Scale thresholds based on image size
                        image_area = im.shape[0] * im.shape[1]
                        base_threshold = image_area * 0.000005  # 0.0005% of image area

                        # if cls == 0:  # First class
                        #     min_area = max(25, base_threshold * 2)
                        # else:  # Other classes
                        min_area = max(5, base_threshold * 0.05)

                        if contour_area < min_area:
                            system_logger.info(
                                f"Skipping contour in mask {instance_id} (class {cls}): area {contour_area:.1f} < {min_area:.1f}"
                            )
                            continue

                        measurements = calculate_measurements(
                            c,
                            single_im_mask,
                            um_pix=um_pix,
                            pixelsPerMetric=pixelsPerMetric,
                            original_image=im,
                            measure_contrast_distribution=measure_contrast_distribution,
                        )

                        # Get class name for CSV
                        class_name = (
                            metadata.thing_classes[cls]
                            if cls < len(metadata.thing_classes)
                            else f"class_{cls}"
                        )

                        # Include class information in CSV
                        csvwriter.writerow(
                            [
                                f"{test_img}_{instance_id}",
                                cls,  # Class number
                                class_name,  # Class name
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
                                measurements["contrast_d10"],
                                measurements["contrast_d50"],
                                measurements["contrast_d90"],
                                psum,
                                test_img,
                            ]
                        )

                        mask_measurements += 1
                        measurements_written += 1

                        # Track by class
                        if cls not in class_measurements:
                            class_measurements[cls] = 0
                        class_measurements[cls] += 1

                    # Add this logging block to show why masks are filtered
                    if mask_measurements == 0:
                        masks_filtered += 1
                        system_logger.info(
                            f"Mask {instance_id} (class {cls}) filtered out: all {total_contours} contours too small (min_area: {min_area:.1f})"
                        )
                    else:
                        system_logger.debug(
                            f"Mask {instance_id} (class {cls}): {mask_measurements}/{total_contours} contours kept"
                        )

                    # Memory optimization: Clear mask data
                    del binary_mask, single_im_mask, mask_3ch, single_cnts
                    gc.collect()

                # ADDED: Report measurement statistics
                system_logger.info(
                    f"Measurements written: {measurements_written}/{len(masks)} masks processed"
                )
                system_logger.info(f"Masks filtered out: {masks_filtered}")

                if class_measurements:
                    class_summary = ", ".join(
                        [
                            f"class {cls}: {count}"
                            for cls, count in sorted(class_measurements.items())
                        ]
                    )
                    system_logger.info(f"Final measurements by class: {class_summary}")
                else:
                    system_logger.warning("No measurements written for any class!")

                # Memory optimization: Clear image data
                del im

                elapsed = time.perf_counter() - start_time
                total_time += elapsed
                system_logger.info(f"Time taken for image {idx}: {elapsed:.3f} seconds")

            # Memory optimization: Force cleanup after each batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final memory cleanup
    del dedup_results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    average_time = total_time / num_images if num_images else 0
    system_logger.info("MEASUREMENTS PHASE COMPLETED")
    system_logger.info(
        f"Average measurement time per image: {average_time:.3f} seconds"
    )
    system_logger.info(f"Results saved to: {csv_filename}")

    # Create a color legend file
    legend_path = os.path.join(output_dir, "class_color_legend.txt")
    with open(legend_path, "w") as f:
        f.write("Class Color Legend:\n")
        f.write("==================\n")
        for i, class_name in enumerate(metadata.thing_classes):
            color_bgr = class_colors[i % len(class_colors)]
            color_rgb = (
                color_bgr[2],
                color_bgr[1],
                color_bgr[0],
            )  # Convert BGR to RGB for display
            f.write(f"Class {i} ({class_name}): RGB{color_rgb}\n")

    system_logger.info(f"Class color legend saved to: {legend_path}")


def iterative_single_predictor_with_classes(
    predictor, image, iou_threshold=0.7, min_increase=0.25, max_iters=5
):
    """
    Run a single predictor iteratively while preserving class information.

    Updated stopping conditions:
    - Must have at least 10 total masks before considering stopping
    - Must add at least 25% of current total OR continue if under 10 masks
    - Stop after 2 consecutive iterations with no new masks
    """
    all_masks = []
    all_scores = []
    all_sources = []
    all_classes = []
    prev_count = 0
    no_new_mask_iters = 0

    for iteration in range(max_iters):
        # Memory optimization: Clear GPU cache before each prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run prediction
        outputs = predictor(image)
        masks = postprocess_masks(
            np.asarray(outputs["instances"].to("cpu")._fields["pred_masks"]),
            outputs["instances"].to("cpu")._fields["scores"].numpy(),
            image,
        )
        scores = outputs["instances"].to("cpu")._fields["scores"].numpy()
        classes = outputs["instances"].to("cpu")._fields["pred_classes"].numpy()

        # Memory optimization: Clear GPU memory
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Add new masks from this iteration
        if masks:
            for i, mask in enumerate(masks):
                all_masks.append(mask)
                all_scores.append(scores[i])
                all_sources.append(0)  # Source 0 for single predictor
                all_classes.append(classes[i])  # Preserve class information

        # Memory optimization: Clear intermediate variables
        del masks, scores, classes
        gc.collect()

        # Deduplicate all masks while preserving class info
        unique_masks = []
        unique_scores = []
        unique_sources = []
        unique_classes = []

        for i, mask in enumerate(all_masks):
            if not any(iou(mask, um) > iou_threshold for um in unique_masks):
                unique_masks.append(mask)
                unique_scores.append(all_scores[i])
                unique_sources.append(all_sources[i])
                unique_classes.append(all_classes[i])  # Preserve class

        new_count = len(unique_masks)
        added = new_count - prev_count

        # Log class distribution in this iteration
        if added > 0:
            new_classes = unique_classes[-added:]  # Classes of newly added masks
            class_counts = {}
            for cls in new_classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
            class_summary = ", ".join(
                [f"class {cls}: {count}" for cls, count in sorted(class_counts.items())]
            )
            system_logger.info(
                f"Iteration {iteration + 1}: Added {added} new masks (total: {new_count}) - New: {class_summary}"
            )
        else:
            system_logger.info(
                f"Iteration {iteration + 1}: Added {added} new masks (total: {new_count})"
            )

        # UPDATED EARLY STOPPING CONDITIONS

        # 1. Track consecutive iterations with no new masks
        if added == 0:
            no_new_mask_iters += 1
        else:
            no_new_mask_iters = 0

        # 2. Stop if no new masks for 2 consecutive iterations
        if no_new_mask_iters >= 2:
            system_logger.info(
                "Stopping: No new masks found in two consecutive iterations."
            )
            break

        # 3. YOUR REQUIREMENTS: Check stopping conditions only if we have enough masks
        if new_count >= 10:  # Only consider stopping if we have at least 10 masks
            if iteration >= 2:  # Allow at least 3 iterations
                # Calculate required increase (25% of current total)
                required_increase = max(1, int(prev_count * min_increase))

                if added < required_increase:
                    system_logger.info(
                        f"Stopping: Added {added} masks < required {required_increase} "
                        f"({min_increase:.0%} of {prev_count} existing masks). "
                        f"Total masks: {new_count}"
                    )
                    break
        else:
            # Continue if we don't have enough masks yet
            system_logger.info(
                f"Continuing: Only {new_count} masks (need at least 10 before considering early stop)"
            )

        prev_count = new_count
        all_masks = unique_masks
        all_scores = unique_scores
        all_sources = unique_sources
        all_classes = unique_classes

    # Final logging
    system_logger.info(
        f"Iterative inference completed with {len(unique_masks)} total masks"
    )

    return unique_masks, unique_scores, unique_sources, unique_classes


def run_class_specific_inference(
    predictor, image, target_class, confidence_threshold=0.3, iou_threshold=0.7
):
    """
    Run inference targeting a specific class with class-optimized parameters.

    Parameters:
    - predictor: Model predictor
    - image: Input image
    - target_class: Class to focus on (0, 1, etc.)
    - confidence_threshold: Confidence threshold for this class
    - iou_threshold: IoU threshold for this class

    Returns:
    - tuple: (masks, scores, classes) for the target class only
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    outputs = predictor(image)

    # Get all predictions
    pred_masks = outputs["instances"].to("cpu")._fields["pred_masks"].numpy()
    pred_scores = outputs["instances"].to("cpu")._fields["scores"].numpy()
    pred_classes = outputs["instances"].to("cpu")._fields["pred_classes"].numpy()

    # Filter for target class only
    class_mask = pred_classes == target_class
    class_pred_masks = pred_masks[class_mask]
    class_pred_scores = pred_scores[class_mask]
    class_pred_classes = pred_classes[class_mask]

    # Apply class-specific confidence threshold
    confidence_mask = class_pred_scores >= confidence_threshold
    filtered_masks = class_pred_masks[confidence_mask]
    filtered_scores = class_pred_scores[confidence_mask]
    filtered_classes = class_pred_classes[confidence_mask]

    # Memory cleanup
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(filtered_masks) == 0:
        return [], [], []

    # Class-specific postprocessing
    if target_class == 0:  # Large particles
        min_size = 25
        processed_masks = postprocess_masks(
            filtered_masks, filtered_scores, image, min_crys_size=min_size
        )
    else:  # Small particles (class 1+)
        min_size = 5  # Much smaller minimum
        processed_masks = postprocess_masks_small_particles(
            filtered_masks, filtered_scores, image, min_crys_size=min_size
        )

    # Class-specific deduplication
    unique_masks = []
    unique_scores = []
    unique_classes = []

    if processed_masks:
        for i, mask in enumerate(processed_masks):
            # More lenient IoU for small particles
            current_iou_threshold = 0.5 if target_class > 0 else iou_threshold

            if not any(iou(mask, um) > current_iou_threshold for um in unique_masks):
                unique_masks.append(mask)
                unique_scores.append(filtered_scores[i])
                unique_classes.append(target_class)

    return unique_masks, unique_scores, unique_classes


def postprocess_masks_small_particles(ori_mask, ori_score, image, min_crys_size=5):
    """
    Gentler postprocessing specifically for small particles.
    """
    if len(ori_mask) == 0:
        return []

    # Much lower score threshold for small particles
    score_threshold = 0.2
    keep_indices = ori_score >= score_threshold

    if not np.any(keep_indices):
        return []

    filtered_masks = ori_mask[keep_indices]

    processed_masks = []
    for mask in filtered_masks:
        # Fill holes but preserve small features
        mask = binary_fill_holes(mask).astype(np.uint8)

        # Very gentle morphological operations
        small_kernel = disk(1)  # Smallest possible kernel
        mask = erosion(mask, small_kernel)
        mask = dilation(mask, small_kernel)

        # Keep if above minimum size
        if np.sum(mask) >= min_crys_size:
            processed_masks.append(mask)

    return processed_masks


def run_multiscale_class_inference(
    predictor, image, target_class, confidence_threshold=0.3, max_iters=5
):
    """
    Multi-scale inference specifically for small particles (class 1).
    """
    scales = [0.8, 1.0, 1.2, 1.5]  # Multiple scales to catch small particles
    all_masks = []
    all_scores = []
    all_classes = []

    for scale in scales:
        system_logger.info(f"Scale {scale}: Processing class {target_class}")

        # Resize image
        if scale != 1.0:
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_image = cv2.resize(image, (new_w, new_h))
        else:
            scaled_image = image

        # Run class-specific inference
        scale_masks, scale_scores, scale_classes = run_class_specific_inference(
            predictor, scaled_image, target_class, confidence_threshold
        )

        # Scale masks back to original size
        if scale != 1.0 and scale_masks:
            original_size_masks = []
            for mask in scale_masks:
                resized_mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                original_size_masks.append(resized_mask.astype(bool))
            scale_masks = original_size_masks

        all_masks.extend(scale_masks)
        all_scores.extend(scale_scores)
        all_classes.extend(scale_classes)

        system_logger.info(f"Scale {scale}: Found {len(scale_masks)} instances")

    # Deduplicate across scales
    unique_masks = []
    unique_scores = []
    unique_classes = []

    # Sort by confidence
    sorted_indices = np.argsort(all_scores)[::-1]

    for idx in sorted_indices:
        mask = all_masks[idx]
        score = all_scores[idx]
        cls = all_classes[idx]

        # Check for duplicates (lenient for small particles)
        is_duplicate = any(iou(mask, um) > 0.4 for um in unique_masks)

        if not is_duplicate:
            unique_masks.append(mask)
            unique_scores.append(score)
            unique_classes.append(cls)

    return unique_masks, unique_scores, unique_classes
