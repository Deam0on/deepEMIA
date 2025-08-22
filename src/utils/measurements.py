"""
Measurement Utilities

This module provides utility functions for geometric and color measurements,
including midpoint calculation, color conversions, wavelength estimation, and
arrow detection in images.
"""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from imutils import perspective
from scipy.spatial import distance as dist

from src.utils.logger_utils import system_logger


def midpoint(ptA: tuple, ptB: tuple) -> tuple:
    """
    Calculates the midpoint between two points.

    Parameters:
    - ptA (tuple): First point coordinates (x, y)
    - ptB (tuple): Second point coordinates (x, y)

    Returns:
    - tuple: Midpoint coordinates (x, y)
    """
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def rgb_to_hsv(r: int, g: int, b: int) -> tuple:
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


def hue_to_wavelength(hue: float) -> float:
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


def rgb_to_wavelength(r: int, g: int, b: int) -> float:
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


def detect_arrows(image: np.ndarray) -> List[Tuple[float, float]]:
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


def calculate_measurements(
    c: np.ndarray,
    single_im_mask: np.ndarray,
    um_pix: float = 1.0,
    pixelsPerMetric: float = 1.0,
    original_image: Optional[np.ndarray] = None,
    measure_contrast_distribution: bool = False,
) -> Dict[str, Union[float, None]]:
    """
    Calculates geometric measurements for a given contour and mask.

    Parameters:
    - c: Contour (numpy array)
    - single_im_mask: Binary mask (numpy array)
    - um_pix: Microns per pixel (float)
    - pixelsPerMetric: Scaling factor (float)

    Returns:
    - dict: All calculated measurements
    """
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    orig = single_im_mask.copy()
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
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
        if dimPerimeter != 0
        else 0
    )
    Circularity = (
        4 * np.pi * (dimArea / (dimPerimeter) ** 2) * um_pix if dimPerimeter != 0 else 0
    )
    Feret_diam = diaFeret * um_pix

    # Ellipse fit
    if len(c) >= 5:
        ellipse = cv2.fitEllipse(c)
        (x, y), (major_axis, minor_axis), angle = ellipse

        if major_axis > minor_axis:
            a = major_axis / 2.0
            b = minor_axis / 2.0
        else:
            a = minor_axis / 2.0
            b = major_axis / 2.0
        eccentricity = np.sqrt(1 - (b**2 / a**2)) if a != 0 else 0

        major_axis_length = major_axis / pixelsPerMetric * um_pix
        minor_axis_length = minor_axis / pixelsPerMetric * um_pix
    else:
        eccentricity = 0
        major_axis_length = 0
        minor_axis_length = 0

    contrast_d10 = contrast_d50 = contrast_d90 = None
    if measure_contrast_distribution and original_image is not None:
        # Mask the original image to get pixel intensities inside the particle
        if len(original_image.shape) == 3:
            # If image is color, convert to grayscale for contrast analysis
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = original_image.copy()
        particle_pixels = gray[single_im_mask > 0]
        if len(particle_pixels) > 0:
            # Compute histogram (0-255)
            hist, bin_edges = np.histogram(
                particle_pixels, bins=256, range=(0, 255), density=True
            )
            cdf = np.cumsum(hist)
            cdf /= cdf[-1]  # Normalize to 1

            # Find d10, d50, d90 (intensity values at 10%, 50%, 90% cumulative area)
            contrast_d10 = np.interp(0.10, cdf, bin_edges[:-1])
            contrast_d50 = np.interp(0.50, cdf, bin_edges[:-1])
            contrast_d90 = np.interp(0.90, cdf, bin_edges[:-1])

    return {
        "major_axis_length": major_axis_length,
        "minor_axis_length": minor_axis_length,
        "eccentricity": eccentricity,
        "Length": Length,
        "Width": Width,
        "CircularED": CircularED,
        "Aspect_Ratio": Aspect_Ratio,
        "Circularity": Circularity,
        "Chords": Chords,
        "Feret_diam": Feret_diam,
        "Roundness": Roundness,
        "Sphericity": Sphericity,
        "contrast_d10": contrast_d10,
        "contrast_d50": contrast_d50,
        "contrast_d90": contrast_d90,
    }
