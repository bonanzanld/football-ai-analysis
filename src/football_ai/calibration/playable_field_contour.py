from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import PitchDetectionProfile
from football_ai.calibration.playable_boundary_semantics import (
    PlayableBoundaryBinding,
    PlayableBoundaryRole,
)


@dataclass(frozen=True, slots=True)
class PlayableFieldCorner:
    corner_id: str
    ground_point_m: tuple[float, float]

    def to_dict(self) -> dict:
        return {"corner_id": self.corner_id, "ground_point_m": list(self.ground_point_m)}


@dataclass(frozen=True, slots=True)
class PlayableFieldContour:
    match_format: str
    pitch_length_m: float
    pitch_width_m: float
    goal_width_m: float
    goal_height_m: float
    corners: tuple[PlayableFieldCorner, ...]
    boundary_bindings: tuple[PlayableBoundaryBinding, ...]

    def __post_init__(self) -> None:
        roles = {item.role for item in self.boundary_bindings}
        required = {PlayableBoundaryRole.END_LINE_A, PlayableBoundaryRole.END_LINE_B}
        if not required <= roles or not all(
            item.confirmed for item in self.boundary_bindings if item.role in required
        ):
            raise ValueError("Beide 8v8-achterlijnen moeten expliciet bevestigd zijn.")

    @property
    def polygon_ground_m(self) -> np.ndarray:
        return np.asarray([item.ground_point_m for item in self.corners], dtype=np.float64)

    def project(self, ground_to_image: np.ndarray) -> np.ndarray:
        points = self.polygon_ground_m
        homogeneous = np.column_stack((points, np.ones(len(points))))
        projected = (np.asarray(ground_to_image, dtype=np.float64) @ homogeneous.T).T
        return projected[:, :2] / projected[:, 2:3]

    def projected_boundaries(
        self, ground_to_image: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Return one closed ring: adjacent boundaries reuse exact corner arrays."""
        corners = tuple(self.project(ground_to_image))
        return tuple((corners[index], corners[(index + 1) % 4]) for index in range(4))

    def to_dict(self) -> dict:
        return {
            "match_format": self.match_format,
            "pitch_length_m": self.pitch_length_m,
            "pitch_width_m": self.pitch_width_m,
            "goal_width_m": self.goal_width_m,
            "goal_height_m": self.goal_height_m,
            "corners": [item.to_dict() for item in self.corners],
            "boundary_bindings": [item.to_dict() for item in self.boundary_bindings],
        }


@dataclass(frozen=True, slots=True)
class PlayableContourGeometryQuality:
    valid: bool
    area_m2: float
    expected_area_m2: float
    edge_lengths_m: tuple[float, float, float, float]
    maximum_dimension_error_m: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "area_m2": self.area_m2,
            "expected_area_m2": self.expected_area_m2,
            "edge_lengths_m": list(self.edge_lengths_m),
            "maximum_dimension_error_m": self.maximum_dimension_error_m,
            "reasons": list(self.reasons),
        }


def create_playable_field_contour(
    profile: PitchDetectionProfile,
    boundary_bindings: tuple[PlayableBoundaryBinding, ...],
) -> PlayableFieldContour:
    length, width = profile.pitch_length_m, profile.pitch_width_m
    corners = (
        PlayableFieldCorner("a_rear", (0.0, 0.0)),
        PlayableFieldCorner("b_rear", (length, 0.0)),
        PlayableFieldCorner("b_front", (length, width)),
        PlayableFieldCorner("a_front", (0.0, width)),
    )
    return PlayableFieldContour(
        profile.match_format.value,
        length,
        width,
        profile.goal_width_m,
        profile.goal_height_m,
        corners,
        boundary_bindings,
    )


def validate_playable_contour_geometry(
    polygon_ground_m: np.ndarray,
    expected_length_m: float,
    expected_width_m: float,
    dimension_tolerance_m: float = 3.0,
) -> PlayableContourGeometryQuality:
    polygon = np.asarray(polygon_ground_m, dtype=np.float64)
    reasons = []
    if polygon.shape != (4, 2) or not np.all(np.isfinite(polygon)):
        return PlayableContourGeometryQuality(False, 0.0, expected_length_m * expected_width_m, (), float("inf"), ("Vier eindige grondhoeken vereist.",))
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        reasons.append("Grondcontour is niet convex of bevat kruisende grenzen.")
    area = abs(float(cv2.contourArea(contour)))
    edges = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
    expected = np.asarray((expected_length_m, expected_width_m, expected_length_m, expected_width_m))
    errors = np.abs(edges - expected)
    maximum_error = float(np.max(errors))
    if maximum_error > dimension_tolerance_m:
        reasons.append(f"Maximale afmetingsfout {maximum_error:.1f}m is groter dan {dimension_tolerance_m:.1f}m.")
    expected_area = expected_length_m * expected_width_m
    if not 0.85 * expected_area <= area <= 1.15 * expected_area:
        reasons.append("Contouroppervlak wijkt meer dan 15% af van het 8v8-referentieveld.")
    return PlayableContourGeometryQuality(
        not reasons,
        area,
        expected_area,
        tuple(map(float, edges)),
        maximum_error,
        tuple(reasons),
    )
