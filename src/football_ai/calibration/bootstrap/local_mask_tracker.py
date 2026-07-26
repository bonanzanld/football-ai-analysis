from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class MaskTrackingResult:
    polygon: np.ndarray
    reliable: bool
    tracked_points: int
    inlier_ratio: float
    transform: np.ndarray


class LocalMaskTracker:
    """Follow a calibrated image polygon during a short continuous camera move."""

    def __init__(
        self,
        frame: np.ndarray,
        polygon: np.ndarray,
        *,
        feature_polygon: np.ndarray | None = None,
        transform_model: str = "affine",
    ) -> None:
        self.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.polygon = np.asarray(polygon, dtype=np.float64).copy()
        self.feature_polygon = np.asarray(
            feature_polygon if feature_polygon is not None else polygon,
            dtype=np.float64,
        ).copy()
        if transform_model not in {"affine", "homography"}:
            raise ValueError("transform_model moet 'affine' of 'homography' zijn.")
        self.transform_model = transform_model

    def update(self, frame: np.ndarray) -> MaskTrackingResult:
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feature_mask = np.zeros(self.previous_gray.shape, dtype=np.uint8)
        visible_points = np.round(self.feature_polygon).astype(np.int32)
        if len(visible_points) >= 3:
            cv2.fillPoly(feature_mask, [visible_points], 255)
        points = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=1200,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
            mask=feature_mask,
        )
        if points is None or len(points) < 30:
            return self._fallback(current_gray, 0)
        tracked, status, _errors = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            current_gray,
            points,
            None,
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if tracked is None or status is None:
            return self._fallback(current_gray, 0)
        valid = status.ravel().astype(bool)
        source = points[:, 0, :][valid]
        target = tracked[:, 0, :][valid]
        if len(source) < 30:
            return self._fallback(current_gray, len(source))
        if self.transform_model == "homography":
            matrix, inliers = cv2.findHomography(
                source,
                target,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.5,
                maxIters=3000,
                confidence=0.995,
            )
        else:
            matrix, inliers = cv2.estimateAffinePartial2D(
                source,
                target,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.5,
                maxIters=3000,
                confidence=0.995,
            )
        if matrix is None or inliers is None:
            return self._fallback(current_gray, len(source))
        inlier_ratio = float(np.mean(inliers))
        scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
        translation = float(np.hypot(matrix[0, 2], matrix[1, 2]))
        perspective = 0.0 if matrix.shape == (2, 3) else float(np.hypot(matrix[2, 0], matrix[2, 1]))
        reliable = (
            int(np.count_nonzero(inliers)) >= 25
            and inlier_ratio >= 0.45
            and 0.90 <= scale <= 1.10
            and translation <= 80.0
            and perspective <= 0.002
        )
        if reliable:
            self.polygon = _transform_points(self.polygon, matrix)
            self.feature_polygon = _transform_points(self.feature_polygon, matrix)
        self.previous_gray = current_gray
        applied = matrix if reliable else _identity_transform(self.transform_model)
        return MaskTrackingResult(self.polygon.copy(), reliable, len(source), inlier_ratio, applied)

    def _fallback(self, current_gray: np.ndarray, tracked_points: int) -> MaskTrackingResult:
        self.previous_gray = current_gray
        return MaskTrackingResult(
            self.polygon.copy(), False, tracked_points, 0.0, _identity_transform(self.transform_model)
        )


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if matrix.shape == (2, 3):
        homogeneous = np.column_stack((values, np.ones(len(values))))
        return (matrix @ homogeneous.T).T
    return cv2.perspectiveTransform(values.astype(np.float32)[None, :, :], matrix)[0].astype(np.float64)


def _identity_transform(model: str) -> np.ndarray:
    return np.eye(3, dtype=np.float64) if model == "homography" else np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
