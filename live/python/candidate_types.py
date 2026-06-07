from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CandidateContour:
    contour: np.ndarray
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    fill_ratio: float
    prior_value: float
    inside_prior: bool
    score: float

    layout_name: str = ""
    layout_group: str = ""
    layout_u: float = 0.0
    layout_v: float = 0.0
    layout_dist: float = 999.0
    layout_score: float = 0.0
    layout_accept_dist: float = 999.0
    is_main_anchor: bool = False
    is_top_backup: bool = False

    fitted_center: tuple[float, float] | None = None
    fit_method: str = "none"
    ellipse_axes: tuple[float, float] | None = None
    ellipse_angle: float | None = None


@dataclass
class LayoutModel:
    origin: tuple[float, float]
    ux: tuple[float, float]
    uy: tuple[float, float]
    scale: float
    left_anchor: CandidateContour
    right_anchor: CandidateContour
