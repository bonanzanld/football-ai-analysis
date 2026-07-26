from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from football_ai.calibration.reference_3d import FootballFieldReference3D, Point3D
from football_ai.calibration.reference_observation import CameraViewObservations


@dataclass(frozen=True, slots=True)
class CameraProjection3D:
    """Projective camera mapping metric world points to image pixels."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("Een cameraprojectie moet een eindige 3x4-matrix zijn.")
        object.__setattr__(self, "matrix", matrix.copy())

    def project(self, point: Point3D | tuple[float, float, float]) -> tuple[float, float]:
        coordinates = point.as_tuple() if isinstance(point, Point3D) else point
        homogeneous = self.matrix @ np.asarray((*coordinates, 1.0), dtype=np.float64)
        if abs(float(homogeneous[2])) < 1e-12:
            raise ValueError("3D-punt projecteert naar oneindig.")
        return float(homogeneous[0] / homogeneous[2]), float(homogeneous[1] / homogeneous[2])

    def ground_homography(self) -> np.ndarray:
        """Return world ground (x, y, 1) -> image (u, v, 1)."""
        homography = self.matrix[:, (0, 1, 3)]
        if abs(float(np.linalg.det(homography))) < 1e-12:
            raise ValueError("De cameraprojectie bepaalt geen omkeerbaar grondvlak.")
        return homography.copy()

    def image_to_ground(self, image_point: tuple[float, float]) -> tuple[float, float]:
        inverse = np.linalg.inv(self.ground_homography())
        ground = inverse @ np.asarray((*image_point, 1.0), dtype=np.float64)
        if abs(float(ground[2])) < 1e-12:
            raise ValueError("Beeldpunt snijdt het grondvlak op oneindig.")
        return float(ground[0] / ground[2]), float(ground[1] / ground[2])


@dataclass(frozen=True, slots=True)
class CameraProjectionEstimate:
    projection: CameraProjection3D
    point_errors_px: tuple[float, ...]
    rms_error_px: float
    maximum_error_px: float


def estimate_camera_projection_dlt(
    reference: FootballFieldReference3D,
    view: CameraViewObservations,
) -> CameraProjectionEstimate:
    """Estimate a full projective camera without assuming camera intrinsics."""
    view.validate(reference)
    if not view.supports_3d_pose(reference):
        raise ValueError(
            "Minimaal vier verspreide grondpunten, twee hoogtepunten en zes observaties vereist."
        )
    world = np.asarray(
        [reference.landmark(item.landmark_id).point.as_tuple() for item in view.observations],
        dtype=np.float64,
    )
    image = np.asarray([item.image_point for item in view.observations], dtype=np.float64)
    world_normalized, world_transform = _normalize_points_3d(world)
    image_normalized, image_transform = _normalize_points_2d(image)
    rows: list[np.ndarray] = []
    for point_3d, point_2d in zip(world_normalized, image_normalized):
        x, y, z = point_3d
        u, v = point_2d
        homogeneous = np.asarray((x, y, z, 1.0), dtype=np.float64)
        zeros = np.zeros(4, dtype=np.float64)
        rows.append(np.concatenate((homogeneous, zeros, -u * homogeneous)))
        rows.append(np.concatenate((zeros, homogeneous, -v * homogeneous)))
    design = np.vstack(rows)
    if np.linalg.matrix_rank(design) < 11:
        raise ValueError("De 3D-observaties bepalen geen unieke cameraprojectie.")
    _u, _singular_values, vh = np.linalg.svd(design)
    normalized_projection = vh[-1].reshape(3, 4)
    projection = np.linalg.inv(image_transform) @ normalized_projection @ world_transform
    scale = float(np.linalg.norm(projection[2, :3]))
    if scale < 1e-12:
        raise ValueError("De geschatte cameraprojectie is numeriek ingestort.")
    projection /= scale
    camera = CameraProjection3D(projection)
    errors = tuple(
        float(np.linalg.norm(np.asarray(camera.project(tuple(point))) - observed))
        for point, observed in zip(world, image)
    )
    rms = float(np.sqrt(np.mean(np.square(errors))))
    return CameraProjectionEstimate(camera, errors, rms, max(errors))


def _normalize_points_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    scale = np.sqrt(2.0) / max(float(np.mean(distances)), 1e-12)
    transform = np.asarray(
        ((scale, 0.0, -scale * center[0]), (0.0, scale, -scale * center[1]), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :2], transform


def _normalize_points_3d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    scale = np.sqrt(3.0) / max(float(np.mean(distances)), 1e-12)
    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = transform[1, 1] = transform[2, 2] = scale
    transform[:3, 3] = -scale * center
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :3], transform
