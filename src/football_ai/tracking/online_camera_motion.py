"""Lichtgewicht compensatie voor camerabeweging tijdens online tracking."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraMotionDiagnostics:
    accepted: bool
    tracked_points: int = 0
    inliers: int = 0
    median_error: float | None = None


class OnlineCameraMotion:
    """Schat per frame een transformatie naar het eerste camerabeeld.

    Alleen dominante, kleine camerabewegingen worden geaccepteerd. Spelers en de
    bal zijn RANSAC-outliers en bepalen daardoor normaal gesproken niet de camera.
    """

    def __init__(self, target_width: int = 640) -> None:
        self.target_width = target_width
        self._previous_gray: np.ndarray | None = None
        self._previous_to_reference = np.eye(3, dtype=np.float64)
        self._full_to_small = np.eye(3, dtype=np.float64)
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.last_diagnostics = CameraMotionDiagnostics(accepted=False)

    def update(self, frame: np.ndarray) -> np.ndarray:
        gray, full_to_small = self._prepare(frame)
        if self._previous_gray is None:
            self._previous_gray = gray
            self._full_to_small = full_to_small
            self.last_diagnostics = CameraMotionDiagnostics(accepted=True)
            return self._previous_to_reference.copy()

        previous_points = self._features(self._previous_gray)
        if previous_points is None or len(previous_points) < 24:
            return self._reject(gray, full_to_small, 0)

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            gray,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if current_points is None or status is None:
            return self._reject(gray, full_to_small, 0)

        valid = status.reshape(-1).astype(bool)
        previous_valid = previous_points.reshape(-1, 2)[valid]
        current_valid = current_points.reshape(-1, 2)[valid]
        tracked = len(previous_valid)
        if tracked < 24:
            return self._reject(gray, full_to_small, tracked)

        previous_to_current, mask = cv2.estimateAffinePartial2D(
            previous_valid,
            current_valid,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )
        if previous_to_current is None or mask is None:
            return self._reject(gray, full_to_small, tracked)

        inlier_mask = mask.reshape(-1).astype(bool)
        inliers = int(np.count_nonzero(inlier_mask))
        inlier_ratio = inliers / max(tracked, 1)
        predicted = cv2.transform(
            previous_valid.reshape(1, -1, 2), previous_to_current
        ).reshape(-1, 2)
        errors = np.linalg.norm(predicted - current_valid, axis=1)
        median_error = float(np.median(errors[inlier_mask])) if inliers else math.inf

        a, b = float(previous_to_current[0, 0]), float(previous_to_current[0, 1])
        scale = math.hypot(a, b)
        rotation = abs(math.degrees(math.atan2(b, a)))
        translation = math.hypot(
            float(previous_to_current[0, 2]), float(previous_to_current[1, 2])
        )
        diagonal = math.hypot(gray.shape[1], gray.shape[0])
        accepted = (
            inliers >= 14
            and inlier_ratio >= 0.45
            and median_error <= 2.5
            and abs(scale - 1.0) <= 0.04
            and rotation <= 3.5
            and translation <= 0.15 * diagonal
        )
        if not accepted:
            return self._reject(
                gray, full_to_small, tracked, inliers, median_error
            )

        affine3 = np.vstack([previous_to_current, [0.0, 0.0, 1.0]])
        current_to_previous_small = np.linalg.inv(affine3)
        small_to_full = np.linalg.inv(full_to_small)
        current_to_previous_full = (
            small_to_full @ current_to_previous_small @ self._full_to_small
        )
        self._previous_to_reference = (
            self._previous_to_reference @ current_to_previous_full
        )
        self._previous_gray = gray
        self._full_to_small = full_to_small
        self.accepted_updates += 1
        self.last_diagnostics = CameraMotionDiagnostics(
            True, tracked, inliers, median_error
        )
        return self._previous_to_reference.copy()

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = frame.shape[:2]
        scale = min(1.0, self.target_width / max(width, 1))
        small = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return gray, np.array(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @staticmethod
    def _features(gray: np.ndarray) -> np.ndarray | None:
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        mask[: round(gray.shape[0] * 0.12)] = 0
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=700,
            qualityLevel=0.01,
            minDistance=7,
            mask=mask,
            blockSize=7,
        )

    def _reject(
        self,
        gray: np.ndarray,
        full_to_small: np.ndarray,
        tracked: int,
        inliers: int = 0,
        median_error: float | None = None,
    ) -> np.ndarray:
        self.rejected_updates += 1
        self.last_diagnostics = CameraMotionDiagnostics(
            False, tracked, inliers, median_error
        )
        return self._previous_to_reference.copy()


def transform_point(
    point: tuple[float, float], matrix: np.ndarray
) -> tuple[float, float]:
    vector = matrix @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    return float(vector[0] / vector[2]), float(vector[1] / vector[2])


def transform_box(
    box: tuple[float, float, float, float], matrix: np.ndarray
) -> tuple[float, float, float, float]:
    corners = (
        (box[0], box[1]),
        (box[2], box[1]),
        (box[2], box[3]),
        (box[0], box[3]),
    )
    transformed = [transform_point(point, matrix) for point in corners]
    xs = [point[0] for point in transformed]
    ys = [point[1] for point in transformed]
    return min(xs), min(ys), max(xs), max(ys)
