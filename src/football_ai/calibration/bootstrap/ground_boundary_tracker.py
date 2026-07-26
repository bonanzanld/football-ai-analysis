from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from football_ai.calibration.bootstrap.local_mask_tracker import LocalMaskTracker
from football_ai.calibration.bootstrap.visible_field_mask import (
    FieldBoundaryGeometry,
    polygon_from_field_boundaries,
)


@dataclass(frozen=True, slots=True)
class GroundBoundaryTrackingResult:
    polygon: np.ndarray
    geometry: FieldBoundaryGeometry
    reliable: bool
    inlier_ratio: float


class GroundBoundaryTracker:
    """Track finite ground-line anchors and rebuild the visible field each frame."""

    def __init__(
        self,
        frame: np.ndarray,
        geometry: FieldBoundaryGeometry,
        *,
        include_backline: bool,
    ) -> None:
        self.frame_size = (frame.shape[1], frame.shape[0])
        self.geometry = geometry
        self.include_backline = include_backline
        polygon = polygon_from_field_boundaries(
            geometry, self.frame_size, include_backline=include_backline
        )
        self.motion = LocalMaskTracker(
            frame,
            polygon,
            feature_polygon=polygon,
            transform_model="homography",
        )

    def update(self, frame: np.ndarray) -> GroundBoundaryTrackingResult:
        motion = self.motion.update(frame)
        if motion.reliable:
            self.geometry = transform_field_geometry(self.geometry, motion.transform)
        polygon = polygon_from_field_boundaries(
            self.geometry, self.frame_size, include_backline=self.include_backline
        )
        return GroundBoundaryTrackingResult(
            polygon=polygon,
            geometry=self.geometry,
            reliable=motion.reliable,
            inlier_ratio=motion.inlier_ratio,
        )


def transform_field_geometry(
    geometry: FieldBoundaryGeometry,
    matrix: np.ndarray,
) -> FieldBoundaryGeometry:
    points = np.vstack(
        (geometry.backline, geometry.rear_sideline, geometry.front_sideline, geometry.interior[None, :])
    ).astype(np.float32)
    transformed = cv2.perspectiveTransform(points[None, :, :], matrix)[0].astype(np.float64)
    return FieldBoundaryGeometry(
        backline=transformed[0:2],
        rear_sideline=transformed[2:4],
        front_sideline=transformed[4:6],
        interior=transformed[6],
    )
