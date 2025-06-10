import re
from math import sqrt

import cv2
import easyocr
import numpy as np


def detect_scale_bar(image, roi_config):
    """
    Detects scale bars in SEM images.

    Parameters:
    - image (numpy.ndarray): Input image

    Returns:
    - tuple: (scale_bar_length, scale_bar_unit)
    """
    h, w = image.shape[:2]
    # Define proportional region where the scale bar and text are located
    # x_start = int(w * 0.667)
    # y_start = int(h * 0.866)
    # x_end = w
    # y_end = int(y_start + h * 0.067)

    x_start = int(w * roi_config["x_start_factor"])
    y_start = int(h * roi_config["y_start_factor"])
    x_end = int(x_start + w * roi_config["width_factor"])
    y_end = int(y_start + h * roi_config["height_factor"])

    # Draw detection ROI border in bright red
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

    roi = image[y_start:y_end, x_start:x_end].copy()
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    reader = easyocr.Reader(["en"], verbose=False)
    result = reader.readtext(gray_roi, detail=1, paragraph=False)

    text_box_center = None
    psum = "0"

    if result:
        # Extract the first recognized text that looks like a scale (e.g., "500nm")
        for detection in result:
            bbox, text, _ = detection
            text_clean = re.sub("[^0-9]", "", text)  # Extract numeric part
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

    # Use Canny edge detection
    edges = cv2.Canny(gray_roi, 50, 150, apertureSize=3)

    lines_list = []
    scale_len = 0
    um_pix = 1

    if text_box_center:
        # Focus on lines near the detected text box
        proximity_threshold = 50  # Distance from text box to search for lines
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
        cv2.line(image, (x1_full, y1_full), (x2_full, y2_full), (0, 255, 0), 2)

        scale_len = max_length
        um_pix = float(psum) / scale_len if scale_len > 0 else 1.0
    else:
        um_pix = 1
        psum = "0"

    return psum, um_pix
