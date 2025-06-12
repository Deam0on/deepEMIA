"""
Measurement Utilities

This module provides utility functions for geometric and color measurements,
including midpoint calculation, color conversions, wavelength estimation, and
arrow detection in images.
"""

import cv2
import numpy as np
import logging


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


def detect_arrows(image: np.ndarray) -> list:
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
