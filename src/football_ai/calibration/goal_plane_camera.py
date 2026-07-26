from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from football_ai.calibration.camera_projection_3d import (
    CameraProjection3D,
    CameraProjectionEstimate,
)
from football_ai.calibration.reference_3d import FootballFieldReference3D
from football_ai.calibration.reference_observation import CameraViewObservations


@dataclass(frozen=True, slots=True)
class GoalPlaneCameraConfig:
    horizontal_fov_degrees: float = 78.0
    minimum_fov_degrees: float = 35.0
    maximum_fov_degrees: float = 120.0
    use_plane_focal_estimate: bool = True


def estimate_camera_from_goal_plane(
    reference: FootballFieldReference3D,
    view: CameraViewObservations,
    frame_size: tuple[int, int],
    config: GoalPlaneCameraConfig = GoalPlaneCameraConfig(),
    ground_direction_vanishing_point: tuple[float, float] | None = None,
) -> CameraProjectionEstimate:
    """Estimate a camera from one known vertical goal/backline plane.

    The principal point is assumed to be the image centre and pixels are square.
    Focal length is estimated from the plane when stable, with a configurable
    field-of-view prior as a bounded fallback.
    """
    view.validate(reference)
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("Frame-afmetingen moeten positief zijn.")
    world = np.asarray(
        [reference.landmark(item.landmark_id).point.as_tuple() for item in view.observations],
        dtype=np.float64,
    )
    image = np.asarray([item.image_point for item in view.observations], dtype=np.float64)
    if len(world) < 4:
        raise ValueError("Minimaal vier hoeken op het bekende doelvlak vereist.")
    x_spread = float(np.ptp(world[:, 0]))
    if x_spread > 1e-6 or float(np.ptp(world[:, 1])) < 1e-6 or float(np.ptp(world[:, 2])) < 1e-6:
        raise ValueError("Doelvlakpunten moeten één verticale rechthoek met breedte en hoogte vormen.")

    plane = world[:, (1, 2)]
    homography, _mask = cv2.findHomography(plane, image, method=0)
    if homography is None or not np.all(np.isfinite(homography)):
        raise ValueError("Het doelvlak bepaalt geen bruikbare beeldprojectie.")
    focal = (
        _estimate_focal_length(homography, width, height, config)
        if config.use_plane_focal_estimate
        else width / (2.0 * np.tan(np.deg2rad(config.horizontal_fov_degrees) / 2.0))
    )
    camera_matrix = np.asarray(
        ((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    solutions = cv2.solvePnPGeneric(
        world,
        image,
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not solutions[0] or not solutions[1]:
        raise ValueError("Camerapositie kon niet uit het doelvlak worden bepaald.")
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for rotation_vector, translation in zip(solutions[1], solutions[2]):
        rotation, _jacobian = cv2.Rodrigues(rotation_vector)
        depths = (rotation @ world.T + translation).T[:, 2]
        camera_center = (-rotation.T @ translation).reshape(3)
        if np.min(depths) <= 0.0 or not 0.5 <= float(camera_center[2]) <= 30.0:
            continue
        projected, _jacobian = cv2.projectPoints(world, rotation_vector, translation, camera_matrix, None)
        projected = projected.reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum(np.square(projected - image), axis=1))))
        direction_penalty = 0.0
        if ground_direction_vanishing_point is not None:
            vanishing = camera_matrix @ rotation[:, 0]
            if abs(float(vanishing[2])) < 1e-9:
                continue
            vanishing = vanishing[:2] / vanishing[2]
            direction_penalty = 0.02 * float(
                np.linalg.norm(vanishing - np.asarray(ground_direction_vanishing_point))
            )
        candidates.append((rms + direction_penalty, rotation_vector, translation))
    if not candidates:
        raise ValueError("Geen fysiek geldige camerapositie voor het doelvlak gevonden.")
    _initial_rms, rotation_vector, translation = min(candidates, key=lambda item: item[0])
    rotation_vector, translation = cv2.solvePnPRefineLM(
        world,
        image,
        camera_matrix,
        None,
        rotation_vector,
        translation,
    )
    rotation, _jacobian = cv2.Rodrigues(rotation_vector)
    projection_matrix = camera_matrix @ np.hstack((rotation, translation))
    projection = CameraProjection3D(projection_matrix)
    depths = (rotation @ world.T + translation).T[:, 2]
    if np.min(depths) <= 0.0:
        raise ValueError("De geschatte camera kijkt van het veld af.")
    camera_center = (-rotation.T @ translation).reshape(3)
    if not 0.5 <= float(camera_center[2]) <= 30.0:
        raise ValueError(
            f"Geschatte camerahoogte is fysiek onlogisch ({camera_center[2]:.1f} m)."
        )
    errors = tuple(
        float(np.linalg.norm(np.asarray(projection.project(tuple(point))) - observed))
        for point, observed in zip(world, image)
    )
    rms = float(np.sqrt(np.mean(np.square(errors))))
    return CameraProjectionEstimate(projection, errors, rms, max(errors))


def _estimate_focal_length(
    homography: np.ndarray,
    width: int,
    height: int,
    config: GoalPlaneCameraConfig,
) -> float:
    centred = np.asarray(
        ((1.0, 0.0, -width / 2.0), (0.0, 1.0, -height / 2.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    ) @ homography
    first, second = centred[:, 0], centred[:, 1]
    coefficients = np.asarray(
        (
            first[0] * second[0] + first[1] * second[1],
            first[0] ** 2 + first[1] ** 2 - second[0] ** 2 - second[1] ** 2,
        ),
        dtype=np.float64,
    )
    targets = np.asarray(
        (-first[2] * second[2], -(first[2] ** 2 - second[2] ** 2)),
        dtype=np.float64,
    )
    denominator = float(coefficients @ coefficients)
    inverse_focal_squared = float(coefficients @ targets / denominator) if denominator > 1e-18 else -1.0
    prior = width / (2.0 * np.tan(np.deg2rad(config.horizontal_fov_degrees) / 2.0))
    focal = np.sqrt(1.0 / inverse_focal_squared) if inverse_focal_squared > 0.0 else prior
    minimum = width / (2.0 * np.tan(np.deg2rad(config.maximum_fov_degrees) / 2.0))
    maximum = width / (2.0 * np.tan(np.deg2rad(config.minimum_fov_degrees) / 2.0))
    if not np.isfinite(focal) or not minimum <= focal <= maximum:
        return float(prior)
    return float(focal)
