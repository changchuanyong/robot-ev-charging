"""Additional measured dark apertures for manual-review suggestions only."""
from __future__ import annotations

import math

import cv2
import numpy as np


def promote_dark_apertures(candidates, crop, contours):
    """Verify already fitted Canny centers against raw-image dark-hole evidence."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    promoted = []
    for item in candidates:
        if item.contour_source != "canny" or item.aperture_center is not None or not item.fit_valid:
            continue
        contrast = contours.dark_boundary_contrast(gray, item)
        if contrast >= contours.DARK_CONTRAST_OFFSET:
            item.aperture_center = item.fitted_center
            item.aperture_diameter_px = math.sqrt(np.prod(item.ellipse_axes))
            item.review_only_aperture = True
            promoted.append({"center": list(item.aperture_center), "contrast": contrast,
                             "contour_index": item.contour_index})
    return promoted
