from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ProjectedPitchGeometry:
    valid: bool
    polygon_area_pixels: float
    frame_area_ratio: float
    minimum_edge_length_pixels: float
    errors: tuple[str, ...]


def validate_projected_pitch_geometry(
    image_corners: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> ProjectedPitchGeometry:
    """Weiger gekruiste, ingeklapte of numeriek ongeldige veldprojecties."""
    corners = np.asarray(image_corners, dtype=np.float64)
    errors: list[str] = []
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        return ProjectedPitchGeometry(
            valid=False,
            polygon_area_pixels=0.0,
            frame_area_ratio=0.0,
            minimum_edge_length_pixels=0.0,
            errors=("Veldhoeken zijn niet vier eindige xy-punten.",),
        )

    contour = corners.astype(np.float32).reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(contour)))
    frame_area = max(float(frame_width * frame_height), 1.0)
    area_ratio = area / frame_area
    edges = np.roll(corners, -1, axis=0) - corners
    minimum_edge = float(np.min(np.linalg.norm(edges, axis=1)))

    if not cv2.isContourConvex(contour):
        errors.append(
            "Geprojecteerde veldranden kruisen of vormen geen convexe vierhoek."
        )
    if area_ratio < 0.02:
        errors.append(
            "Geprojecteerd veld beslaat minder dan 2% van het frameoppervlak."
        )
    if minimum_edge < 20.0:
        errors.append("Minimaal één geprojecteerde veldrand is korter dan 20 px.")

    return ProjectedPitchGeometry(
        valid=not errors,
        polygon_area_pixels=area,
        frame_area_ratio=area_ratio,
        minimum_edge_length_pixels=minimum_edge,
        errors=tuple(errors),
    )
