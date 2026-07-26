from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True, slots=True)
class LensIntrinsics:
    frame_size: tuple[int, int]
    focal_length_px: float
    principal_point: tuple[float, float]
    radial_distortion: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        width, height = self.frame_size
        if width <= 0 or height <= 0 or self.focal_length_px <= 0.0:
            raise ValueError("Lensparameters vereisen positieve frame- en brandpuntsafmetingen.")
        if not np.all(np.isfinite((*self.principal_point, *self.radial_distortion))):
            raise ValueError("Lensparameters moeten eindig zijn.")

    @property
    def camera_matrix(self) -> np.ndarray:
        cx, cy = self.principal_point
        focal = self.focal_length_px
        return np.asarray(((focal, 0.0, cx), (0.0, focal, cy), (0.0, 0.0, 1.0)), dtype=np.float64)

    @property
    def distortion_coefficients(self) -> np.ndarray:
        k1, k2 = self.radial_distortion
        return np.asarray((k1, k2, 0.0, 0.0, 0.0), dtype=np.float64)

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        image_points = np.asarray(points, dtype=np.float64)
        if image_points.ndim != 2 or image_points.shape[1] != 2:
            raise ValueError("Beeldpunten moeten een Nx2-array vormen.")
        corrected = cv2.undistortPoints(
            image_points.reshape(-1, 1, 2),
            self.camera_matrix,
            self.distortion_coefficients,
            P=self.camera_matrix,
        )
        return corrected.reshape(-1, 2)


@dataclass(frozen=True, slots=True)
class StraightLineObservation:
    points: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("Een rechte-lijnobservatie vereist minimaal drie beeldpunten.")
        if not np.all(np.isfinite(points)):
            raise ValueError("Lijnpunten moeten eindig zijn.")
        object.__setattr__(self, "points", points)


@dataclass(frozen=True, slots=True)
class LensDistortionEstimate:
    intrinsics: LensIntrinsics
    rms_straightness_px: float
    maximum_straightness_px: float
    line_count: int


def estimate_radial_distortion_from_lines(
    frame_size: tuple[int, int],
    observations: tuple[StraightLineObservation, ...],
    *,
    initial_focal_length_px: float | None = None,
) -> LensDistortionEstimate:
    """Estimate radial distortion from physically straight image markings."""
    if len(observations) < 2:
        raise ValueError("Lensschatting vereist minimaal twee rechte lijnen.")
    width, height = frame_size
    focal = float(initial_focal_length_px or max(width, height))
    center = (width / 2.0, height / 2.0)

    def corrected(parameters: np.ndarray, points: np.ndarray) -> np.ndarray:
        model = LensIntrinsics(frame_size, focal, center, (float(parameters[0]), float(parameters[1])))
        return model.undistort_points(points)

    def signed_line_errors(points: np.ndarray) -> np.ndarray:
        centroid = np.mean(points, axis=0)
        _u, _s, vh = np.linalg.svd(points - centroid, full_matrices=False)
        return (points - centroid) @ vh[-1]

    def residual(parameters: np.ndarray) -> np.ndarray:
        parts = [signed_line_errors(corrected(parameters, item.points)) for item in observations]
        parts.append(np.asarray((parameters[1] * 0.25,), dtype=np.float64))
        return np.concatenate(parts)

    optimum = least_squares(
        residual,
        np.zeros(2, dtype=np.float64),
        bounds=((-1.0, -1.0), (1.0, 1.0)),
        loss="soft_l1",
        f_scale=1.5,
        max_nfev=600,
    )
    intrinsics = LensIntrinsics(frame_size, focal, center, tuple(float(v) for v in optimum.x))
    errors = np.concatenate(
        [np.abs(signed_line_errors(intrinsics.undistort_points(item.points))) for item in observations]
    )
    return LensDistortionEstimate(
        intrinsics,
        float(np.sqrt(np.mean(np.square(errors)))),
        float(np.max(errors)),
        len(observations),
    )
