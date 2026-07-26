from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from football_ai.calibration.camera_projection_3d import CameraProjection3D
from football_ai.calibration.reference_3d import FootballFieldReference3D
from football_ai.calibration.reference_observation import CameraViewObservations


@dataclass(frozen=True, slots=True)
class FixedCameraLineConstraint:
    view_index: int
    world_axis: int
    image_line: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.world_axis not in (0, 1):
            raise ValueError("Een grondrichting moet wereldas 0 of 1 gebruiken.")
        line = np.asarray(self.image_line, dtype=np.float64)
        if line.shape != (3,) or not np.all(np.isfinite(line)):
            raise ValueError("Een lijnvoorwaarde vereist drie eindige lijncoëfficiënten.")
        normal = float(np.linalg.norm(line[:2]))
        if normal < 1e-9:
            raise ValueError("Een lijnvoorwaarde heeft geen geldige richting.")
        object.__setattr__(self, "image_line", line / normal)


@dataclass(frozen=True, slots=True)
class FixedCameraPointConstraint:
    view_index: int
    world_point: tuple[float, float, float]
    image_point: tuple[float, float]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0.0 or not np.all(np.isfinite((*self.world_point, *self.image_point))):
            raise ValueError("Een gewogen camerapuntvoorwaarde moet eindig en positief zijn.")


@dataclass(frozen=True, slots=True)
class FixedCameraViewPose:
    frame_number: int
    focal_length_px: float
    rotation_vector: np.ndarray
    projection: CameraProjection3D
    rms_error_px: float
    maximum_error_px: float
    principal_point: tuple[float, float]


@dataclass(frozen=True, slots=True)
class FixedCameraPoseEstimate:
    camera_center: np.ndarray
    views: tuple[FixedCameraViewPose, ...]
    rms_error_px: float
    maximum_error_px: float
    pitch_length_m: float
    pitch_width_m: float


def estimate_fixed_camera_poses(
    reference: FootballFieldReference3D,
    views: tuple[CameraViewObservations, ...],
    frame_size: tuple[int, int],
    line_constraints: tuple[FixedCameraLineConstraint, ...] = (),
    point_constraints: tuple[FixedCameraPointConstraint, ...] = (),
    focal_length_prior_px: float | None = None,
    focal_prior_weight: float = 0.0,
    shared_focal_weight: float = 0.0,
    camera_height_prior_m: float | None = None,
    camera_height_weight: float = 0.0,
    camera_center_prior_xy: tuple[float, float] | None = None,
    camera_center_weight: float = 0.0,
    minimum_camera_y: float | None = None,
    pitch_dimension_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    pitch_dimension_prior_weight: float = 0.0,
    camera_outside_clearance_m: float | None = None,
    camera_outside_weight: float = 0.0,
    camera_x_bounds: tuple[float, float] | None = None,
    estimate_principal_point: bool = False,
    principal_point_prior_weight: float = 0.0,
    principal_point_max_shift_ratio: float = 0.2,
) -> FixedCameraPoseEstimate:
    """Jointly estimate fixed camera location with pan/tilt/zoom per view."""
    if focal_length_prior_px is not None and focal_length_prior_px <= 0.0:
        raise ValueError("De brandpuntsprior moet positief zijn.")
    if focal_prior_weight < 0.0 or shared_focal_weight < 0.0:
        raise ValueError("Brandpuntsgewichten mogen niet negatief zijn.")
    if camera_height_prior_m is not None and camera_height_prior_m <= 0.0:
        raise ValueError("De camerahoogteprior moet positief zijn.")
    if camera_height_weight < 0.0:
        raise ValueError("Het camerahoogtegewicht mag niet negatief zijn.")
    if camera_center_prior_xy is not None and not np.all(np.isfinite(camera_center_prior_xy)):
        raise ValueError("De camerapositieprior moet eindig zijn.")
    if camera_center_weight < 0.0:
        raise ValueError("Het camerapositiegewicht mag niet negatief zijn.")
    if minimum_camera_y is not None and not np.isfinite(minimum_camera_y):
        raise ValueError("De minimale camera-y moet eindig zijn.")
    if pitch_dimension_prior_weight < 0.0 or camera_outside_weight < 0.0:
        raise ValueError("Veld- en buitenveldgewichten mogen niet negatief zijn.")
    if camera_outside_clearance_m is not None and camera_outside_clearance_m < 0.0:
        raise ValueError("De minimale afstand buiten het veld mag niet negatief zijn.")
    if camera_x_bounds is not None and not (
        np.all(np.isfinite(camera_x_bounds)) and camera_x_bounds[0] < camera_x_bounds[1]
    ):
        raise ValueError("De grenzen voor camera-x moeten eindig en oplopend zijn.")
    if principal_point_prior_weight < 0.0:
        raise ValueError("Het gewicht voor het optische beeldcentrum mag niet negatief zijn.")
    if not 0.0 < principal_point_max_shift_ratio < 0.5:
        raise ValueError("De maximale beeldcentrumverschuiving moet tussen 0 en 0,5 liggen.")
    if pitch_dimension_bounds is not None:
        (minimum_length, maximum_length), (minimum_width, maximum_width) = pitch_dimension_bounds
        if not (0.0 < minimum_length <= reference.pitch_length_m <= maximum_length):
            raise ValueError("De nominale veldlengte moet binnen de schattingsgrenzen liggen.")
        if not (0.0 < minimum_width <= reference.pitch_width_m <= maximum_width):
            raise ValueError("De nominale veldbreedte moet binnen de schattingsgrenzen liggen.")
    if len(views) < 2:
        raise ValueError("Een vast cameramodel vereist minimaal twee referentiebeelden.")
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("Frame-afmetingen moeten positief zijn.")
    prepared = []
    initial_centers = []
    initial_rotations = []
    initial_focals = []
    for view in views:
        view.validate(reference)
        world = np.asarray(
            [reference.landmark(item.landmark_id).point.as_tuple() for item in view.observations],
            dtype=np.float64,
        )
        image = np.asarray([item.image_point for item in view.observations], dtype=np.float64)
        if len(world) < 4:
            raise ValueError("Ieder posebeeld vereist minimaal vier 3D-naar-2D-observaties.")
        focal, rotation_vector, translation = _initial_pose(world, image, width, height)
        rotation, _ = cv2.Rodrigues(rotation_vector)
        center = (-rotation.T @ translation).reshape(3)
        prepared.append((view, world, image))
        initial_centers.append(center)
        initial_rotations.append(rotation_vector.reshape(3))
        initial_focals.append(focal)
    center = np.median(np.asarray(initial_centers), axis=0)
    center[2] = float(np.clip(center[2], 1.0, 25.0))
    if camera_center_prior_xy is not None:
        # Goal-only PnP estimates are nearly planar and can initialise on the
        # mirrored side of the pitch. Start the joint solve at the supplied
        # physical setup prior; the prior remains soft during optimisation.
        center[:2] = np.asarray(camera_center_prior_xy, dtype=np.float64)
    if camera_height_prior_m is not None:
        center[2] = camera_height_prior_m
    values = [*center]
    pose_stride = 6 if estimate_principal_point else 4
    for rotation, focal in zip(initial_rotations, initial_focals):
        values.extend(rotation)
        values.append(float(np.log(focal)))
        if estimate_principal_point:
            values.extend((width / 2.0, height / 2.0))
    if pitch_dimension_bounds is not None:
        values.extend((reference.pitch_length_m, reference.pitch_width_m))
    initial = np.asarray(values, dtype=np.float64)
    lower = np.full_like(initial, -np.inf)
    upper = np.full_like(initial, np.inf)
    lower[:3] = (-250.0, -250.0, 0.5)
    upper[:3] = (250.0, 250.0, 30.0)
    if camera_x_bounds is not None:
        lower[0], upper[0] = camera_x_bounds
        initial[0] = float(np.clip(initial[0], lower[0] + 0.1, upper[0] - 0.1))
    if minimum_camera_y is not None:
        lower[1] = minimum_camera_y
        initial[1] = max(initial[1], minimum_camera_y + 0.5)
    for index in range(len(views)):
        offset = 3 + index * pose_stride
        lower[offset + 3] = np.log(250.0)
        upper[offset + 3] = np.log(6000.0)
        if estimate_principal_point:
            lower[offset + 4] = width * (0.5 - principal_point_max_shift_ratio)
            upper[offset + 4] = width * (0.5 + principal_point_max_shift_ratio)
            lower[offset + 5] = height * (0.5 - principal_point_max_shift_ratio)
            upper[offset + 5] = height * (0.5 + principal_point_max_shift_ratio)
    if pitch_dimension_bounds is not None:
        lower[-2:] = (pitch_dimension_bounds[0][0], pitch_dimension_bounds[1][0])
        upper[-2:] = (pitch_dimension_bounds[0][1], pitch_dimension_bounds[1][1])

    def unpack(parameters: np.ndarray):
        camera_center = parameters[:3]
        pitch_length = (
            float(parameters[-2]) if pitch_dimension_bounds is not None else reference.pitch_length_m
        )
        pitch_width = (
            float(parameters[-1]) if pitch_dimension_bounds is not None else reference.pitch_width_m
        )
        poses = []
        for index in range(len(views)):
            offset = 3 + index * pose_stride
            rotation_vector = parameters[offset:offset + 3]
            focal = float(np.exp(parameters[offset + 3]))
            principal_x = float(parameters[offset + 4]) if estimate_principal_point else width / 2.0
            principal_y = float(parameters[offset + 5]) if estimate_principal_point else height / 2.0
            rotation, _ = cv2.Rodrigues(rotation_vector)
            translation = -rotation @ camera_center.reshape(3, 1)
            camera_matrix = np.asarray(
                ((focal, 0.0, principal_x), (0.0, focal, principal_y), (0.0, 0.0, 1.0)),
                dtype=np.float64,
            )
            poses.append((rotation_vector, rotation, translation, camera_matrix))
        return camera_center, poses, pitch_length, pitch_width

    def residual(parameters: np.ndarray, include_lines: bool = True) -> np.ndarray:
        camera_center, poses, pitch_length, pitch_width = unpack(parameters)
        parts = []
        for (view, _world, image), (rotation_vector, _rotation, translation, camera_matrix) in zip(prepared, poses):
            world = _metric_world_points(reference, view, pitch_length, pitch_width)
            projected, _ = cv2.projectPoints(world, rotation_vector, translation, camera_matrix, None)
            parts.append((projected.reshape(-1, 2) - image).reshape(-1))
        for constraint in line_constraints if include_lines else ():
            if not 0 <= constraint.view_index < len(poses):
                raise ValueError("Lijnvoorwaarde verwijst naar een onbekend referentiebeeld.")
            _rvec, rotation, _translation, camera_matrix = poses[constraint.view_index]
            direction = np.zeros(3, dtype=np.float64)
            direction[constraint.world_axis] = 1.0
            vanishing = camera_matrix @ (rotation @ direction)
            if abs(float(vanishing[2])) < 1e-9:
                parts.append(np.asarray((500.0,)))
            else:
                point = vanishing / vanishing[2]
                parts.append(
                    np.asarray((float(constraint.image_line @ point) * np.sqrt(constraint.weight),))
                )
        for constraint in point_constraints:
            if not 0 <= constraint.view_index < len(poses):
                raise ValueError("Puntvoorwaarde verwijst naar een onbekend referentiebeeld.")
            rotation_vector, _rotation, translation, camera_matrix = poses[constraint.view_index]
            projected, _ = cv2.projectPoints(
                np.asarray(
                    (
                        _metric_constraint_point(
                            reference, constraint.world_point, pitch_length, pitch_width
                        ),
                    ),
                    dtype=np.float64,
                ),
                rotation_vector,
                translation,
                camera_matrix,
                None,
            )
            parts.append(
                (projected.reshape(2) - np.asarray(constraint.image_point, dtype=np.float64))
                * np.sqrt(constraint.weight)
            )
        focal_values = np.asarray(
            [float(camera_matrix[0, 0]) for _rvec, _rotation, _translation, camera_matrix in poses]
        )
        if focal_length_prior_px is not None and focal_prior_weight > 0.0:
            parts.append(
                np.log(focal_values / focal_length_prior_px) * np.sqrt(focal_prior_weight)
            )
        if len(focal_values) > 1 and shared_focal_weight > 0.0:
            common = float(np.exp(np.mean(np.log(focal_values))))
            parts.append(np.log(focal_values / common) * np.sqrt(shared_focal_weight))
        if estimate_principal_point and principal_point_prior_weight > 0.0:
            principal_points = np.asarray(
                [camera_matrix[:2, 2] for _rvec, _rotation, _translation, camera_matrix in poses]
            )
            scale = np.asarray((width, height), dtype=np.float64)
            parts.append(
                ((principal_points - scale / 2.0) / scale).reshape(-1)
                * np.sqrt(principal_point_prior_weight)
            )
        if camera_height_prior_m is not None and camera_height_weight > 0.0:
            parts.append(
                np.asarray(
                    ((camera_center[2] - camera_height_prior_m) * np.sqrt(camera_height_weight),)
                )
            )
        if camera_center_prior_xy is not None and camera_center_weight > 0.0:
            parts.append(
                (camera_center[:2] - np.asarray(camera_center_prior_xy, dtype=np.float64))
                * np.sqrt(camera_center_weight)
            )
        if pitch_dimension_bounds is not None and pitch_dimension_prior_weight > 0.0:
            parts.append(
                np.asarray(
                    (
                        (pitch_length - reference.pitch_length_m)
                        / reference.pitch_length_m
                        * np.sqrt(pitch_dimension_prior_weight),
                        (pitch_width - reference.pitch_width_m)
                        / reference.pitch_width_m
                        * np.sqrt(pitch_dimension_prior_weight),
                    )
                )
            )
        if camera_outside_clearance_m is not None and camera_outside_weight > 0.0:
            shortage = pitch_width + camera_outside_clearance_m - camera_center[1]
            parts.append(np.asarray((max(0.0, shortage) * np.sqrt(camera_outside_weight),)))
        # A fixed sports camera must remain above the ground; this weak prior only
        # stabilises nearly planar goal observations and does not prescribe a brand.
        parts.append(np.asarray((max(0.0, 1.5 - camera_center[2]) * 20.0,)))
        return np.concatenate(parts)

    point_optimum = least_squares(
        lambda parameters: residual(parameters, False),
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=3.0,
        max_nfev=1200,
    )
    optimum = least_squares(
        residual,
        point_optimum.x,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=3.0,
        max_nfev=1200,
    )
    camera_center, poses, pitch_length, pitch_width = unpack(optimum.x)
    view_results = []
    all_errors = []
    for (view, _world, image), (rotation_vector, rotation, translation, camera_matrix) in zip(prepared, poses):
        world = _metric_world_points(reference, view, pitch_length, pitch_width)
        projected, _ = cv2.projectPoints(world, rotation_vector, translation, camera_matrix, None)
        errors = np.linalg.norm(projected.reshape(-1, 2) - image, axis=1)
        projection = CameraProjection3D(camera_matrix @ np.hstack((rotation, translation)))
        all_errors.extend(errors.tolist())
        view_results.append(
            FixedCameraViewPose(
                view.frame_number,
                float(camera_matrix[0, 0]),
                rotation_vector.copy(),
                projection,
                float(np.sqrt(np.mean(np.square(errors)))),
                float(np.max(errors)),
                (float(camera_matrix[0, 2]), float(camera_matrix[1, 2])),
            )
        )
    errors = np.asarray(all_errors, dtype=np.float64)
    return FixedCameraPoseEstimate(
        camera_center.copy(),
        tuple(view_results),
        float(np.sqrt(np.mean(np.square(errors)))),
        float(np.max(errors)),
        pitch_length,
        pitch_width,
    )


def _metric_world_points(
    reference: FootballFieldReference3D,
    view: CameraViewObservations,
    pitch_length_m: float,
    pitch_width_m: float,
) -> np.ndarray:
    """Rebuild metric landmarks while keeping the physical goal at 5 x 2 metres."""
    half_goal = reference.goal_width_m / 2.0
    points = []
    for observation in view.observations:
        landmark = reference.landmark(observation.landmark_id)
        identifier = landmark.landmark_id
        if identifier.startswith("goal_a_"):
            x = 0.0
        elif identifier.startswith("goal_b_"):
            x = pitch_length_m
        else:
            x = landmark.point.x / reference.pitch_length_m * pitch_length_m
        if "_rear_" in identifier and identifier.startswith("goal_"):
            y = pitch_width_m / 2.0 - half_goal
        elif "_front_" in identifier and identifier.startswith("goal_"):
            y = pitch_width_m / 2.0 + half_goal
        else:
            y = landmark.point.y / reference.pitch_width_m * pitch_width_m
        points.append((x, y, landmark.point.z))
    return np.asarray(points, dtype=np.float64)


def _metric_constraint_point(
    reference: FootballFieldReference3D,
    point: tuple[float, float, float],
    pitch_length_m: float,
    pitch_width_m: float,
) -> tuple[float, float, float]:
    """Move generic ground constraints with the estimated field dimensions."""
    return (
        point[0] / reference.pitch_length_m * pitch_length_m,
        point[1] / reference.pitch_width_m * pitch_width_m,
        point[2],
    )


def _initial_pose(
    world: np.ndarray,
    image: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    focal = float(width / (2.0 * np.tan(np.deg2rad(70.0) / 2.0)))
    camera_matrix = np.asarray(
        ((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    success, rotation_vector, translation = cv2.solvePnP(
        world,
        image,
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        raise ValueError("Geen initiële camerapose gevonden.")
    rotation_vector, translation = cv2.solvePnPRefineLM(
        world, image, camera_matrix, None, rotation_vector, translation
    )
    return focal, rotation_vector, translation
