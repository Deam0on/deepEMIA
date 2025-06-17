import re
from math import sqrt
from pathlib import Path

import cv2
import easyocr
import numpy as np
import yaml

from src.utils.logger_utils import system_logger


class ScaleBarDetectionError(Exception):
    pass


def detect_scale_bar(
    image, roi_config, intensity_threshold=200, proximity_threshold=50
):
    """
    Detects scale bars in SEM images using OCR and Hough line detection.

    Parameters:
    - image (numpy.ndarray): Input image
    - roi_config (dict): ROI configuration with keys x_start_factor, y_start_factor, width_factor, height_factor

    Returns:
    - tuple: (scale_bar_length_str, microns_per_pixel)
        - scale_bar_length_str (str): The detected scale bar length as a string (e.g., "500")
        - microns_per_pixel (float): The conversion factor from pixels to microns
    """
    # --- Load thresholds from config if available ---
    config_path = Path.home() / "uw-com-vision" / "config" / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)
            scalebar_thresholds = full_config.get("scalebar_thresholds", {})
            # Only override if not explicitly passed in function call
            if "intensity" in scalebar_thresholds and intensity_threshold == 200:
                intensity_threshold = scalebar_thresholds["intensity"]
            if "proximity" in scalebar_thresholds and proximity_threshold == 50:
                proximity_threshold = scalebar_thresholds["proximity"]
        except Exception as e:
            system_logger.warning(f"Could not load thresholds from config.yaml: {e}")

    if not isinstance(image, np.ndarray):
        system_logger.error("Input image is not a numpy array.")
        raise ScaleBarDetectionError("Input image is not a numpy array.")
    for key in ["x_start_factor", "y_start_factor", "width_factor", "height_factor"]:
        if key not in roi_config:
            system_logger.error(f"ROI config missing key: {key}")
            raise ScaleBarDetectionError(f"ROI config missing key: {key}")

    h, w = image.shape[:2]
    x_start = int(w * roi_config["x_start_factor"])
    y_start = int(h * roi_config["y_start_factor"])
    x_end = int(x_start + w * roi_config["width_factor"])
    y_end = int(y_start + h * roi_config["height_factor"])

    system_logger.info(
        f"ROI for scale bar OCR: x={x_start}:{x_end}, y={y_start}:{y_end}"
    )
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

    roi = image[y_start:y_end, x_start:x_end].copy()
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    try:
        reader = easyocr.Reader(["en"], verbose=False)
        result = reader.readtext(gray_roi, detail=1, paragraph=False)
    except Exception as e:
        system_logger.error(f"EasyOCR failed: {e}")
        result = []

    text_box_center = None
    psum = "0"

    if result:
        system_logger.info(f"OCR detected text: {[r[1] for r in result]}")
        for detection in result:
            bbox, text, _ = detection
            text_clean = re.sub("[^0-9]", "", text)
            if text_clean:
                pxum_r = text
                psum = text_clean
                x_min = int(min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]))
                y_min = int(min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]))
                x_max = int(max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]))
                y_max = int(max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]))
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
        system_logger.warning("No text detected by OCR in scale bar ROI.")

    edges = cv2.Canny(gray_roi, 50, 150, apertureSize=3)
    lines_list = []
    scale_len = 0
    um_pix = 1

    longest_line = None
    max_length = 0

    if text_box_center:
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=100,
            minLineLength=50,
            maxLineGap=5,
        )
        if lines is not None:
            for points in lines:
                x1, y1, x2, y2 = points[0]
                line_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                dist_to_text = sqrt(
                    (line_center[0] - text_box_center[0]) ** 2
                    + (line_center[1] - text_box_center[1]) ** 2
                )
                if dist_to_text < proximity_threshold:
                    length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    # Intensity check: mean intensity along the line should be high (white bar)
                    line_mask = np.zeros_like(gray_roi, dtype=np.uint8)
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
                    mean_intensity = cv2.mean(gray_roi, mask=line_mask)[0]
                    if mean_intensity > intensity_threshold:
                        if length > max_length:
                            max_length = length
                            longest_line = (x1, y1, x2, y2)

    if longest_line:
        x1, y1, x2, y2 = longest_line
        x1_full = x1 + x_start
        x2_full = x2 + x_start
        y1_full = y1 + y_start
        y2_full = y2 + y_start
        cv2.line(image, (x1_full, y1_full), (x2_full, y2_full), (0, 255, 0), 2)
        scale_len = max_length
        um_pix = float(psum) / scale_len if scale_len > 0 else 1.0
        system_logger.info(
            f"Detected scale bar: {psum} units, {scale_len:.2f} pixels, {um_pix:.4f} units/pixel"
        )
    else:
        um_pix = 1
        psum = "0"
        system_logger.warning("No scale bar line detected near OCR text.")

    return psum, um_pix
