from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ZoomSegmentIntrinsics:
    start_seconds: float
    end_seconds: float
    reference_seconds: float
    focal_length_px: float
    principal_point: tuple[float, float]
    horizontal_fov_degrees: float
    evidence: str


def select_widest_zoom_segment(
    segments: tuple[ZoomSegmentIntrinsics, ...],
) -> ZoomSegmentIntrinsics:
    """Select the stable segment with the widest field of view.

    For equal frame sizes and a fixed principal point, the smallest focal
    length is the most zoomed-out view.  FOV is used as a deterministic
    secondary check.
    """
    if not segments:
        raise ValueError("Minimaal één zoomsegment vereist.")
    if any(item.focal_length_px <= 0.0 for item in segments):
        raise ValueError("Zoomsegmenten vereisen een positieve brandpuntsafstand.")
    return min(
        segments,
        key=lambda item: (
            item.focal_length_px,
            -item.horizontal_fov_degrees,
            item.start_seconds,
        ),
    )


def focal_from_orthogonal_vanishing_points(
    first: tuple[float, float],
    second: tuple[float, float],
    principal_point: tuple[float, float],
) -> float:
    first_value = np.asarray(first, dtype=np.float64)
    second_value = np.asarray(second, dtype=np.float64)
    center = np.asarray(principal_point, dtype=np.float64)
    focal_squared = -float((first_value - center) @ (second_value - center))
    if focal_squared <= 0.0:
        raise ValueError("Orthogonale verdwijnpunten leveren geen fysieke brandpuntsafstand.")
    return float(np.sqrt(focal_squared))


def horizontal_fov(focal_length_px: float, frame_width_px: int) -> float:
    if focal_length_px <= 0.0 or frame_width_px <= 0:
        raise ValueError("Brandpuntsafstand en beeldbreedte moeten positief zijn.")
    return float(np.degrees(2.0 * np.arctan(frame_width_px / (2.0 * focal_length_px))))
