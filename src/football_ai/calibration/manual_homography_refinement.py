from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True, slots=True)
class ManualHomographyRefinement:
    homography: np.ndarray
    rms_point_error_px: float
    maximum_point_error_px: float
    direction_assignment: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ManualLineHomographyRefinement:
    homography: np.ndarray
    rms_point_error_px: float
    maximum_point_error_px: float
    rms_line_error_px: float
    line_axis_assignment: tuple[int, ...]


def refine_ground_homography_with_vanishing_points(
    initial_homography: np.ndarray,
    ground_points: np.ndarray,
    image_points: np.ndarray,
    vanishing_points: tuple[tuple[float, float], tuple[float, float]],
) -> ManualHomographyRefinement:
    initial = np.asarray(initial_homography, dtype=np.float64)
    ground = np.asarray(ground_points, dtype=np.float64)
    image = np.asarray(image_points, dtype=np.float64)
    if initial.shape != (3, 3) or ground.shape[0] < 3 or ground.shape != image.shape:
        raise ValueError("Homographyverfijning vereist een 3x3-startmodel en minimaal drie 2D-puntparen.")
    if ground.shape[1] != 2:
        raise ValueError("Grond- en beeldpunten moeten tweedimensionaal zijn.")
    targets = tuple(np.asarray(item, dtype=np.float64) for item in vanishing_points)

    def pack(matrix: np.ndarray) -> np.ndarray:
        matrix = matrix / matrix[2, 2]
        return matrix.reshape(-1)[:8]

    def unpack(values: np.ndarray) -> np.ndarray:
        return np.append(values, 1.0).reshape(3, 3)

    def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
        homogeneous = np.column_stack((points, np.ones(len(points))))
        values = (matrix @ homogeneous.T).T
        return values[:, :2] / values[:, 2:3]

    best = None
    for assignment in ((0, 1), (1, 0)):
        def residual(values: np.ndarray) -> np.ndarray:
            matrix = unpack(values)
            point_errors = (project(matrix, ground) - image).reshape(-1) * 3.0
            direction_errors = []
            for axis, target_index in enumerate(assignment):
                vanishing = matrix[:, axis]
                if abs(float(vanishing[2])) < 1e-9:
                    direction_errors.extend((500.0, 500.0))
                else:
                    direction_errors.extend((vanishing[:2] / vanishing[2] - targets[target_index]).tolist())
            regularization = (values - pack(initial)) * 1e-4
            return np.concatenate((point_errors, np.asarray(direction_errors), regularization))

        optimum = least_squares(
            residual,
            pack(initial),
            loss="soft_l1",
            f_scale=3.0,
            max_nfev=500,
        )
        matrix = unpack(optimum.x)
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            continue
        errors = np.linalg.norm(project(matrix, ground) - image, axis=1)
        score = float(np.sqrt(np.mean(np.square(residual(optimum.x)))))
        candidate = (score, matrix, errors, assignment)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("Handmatige verdwijnpunten leveren geen omkeerbare grondhomography.")
    _score, matrix, errors, assignment = best
    return ManualHomographyRefinement(
        matrix / matrix[2, 2],
        float(np.sqrt(np.mean(np.square(errors)))),
        float(np.max(errors)),
        assignment,
    )


def refine_ground_homography_with_lines(
    initial_homography: np.ndarray,
    ground_points: np.ndarray,
    image_points: np.ndarray,
    image_lines: tuple[np.ndarray, ...],
) -> ManualLineHomographyRefinement:
    """Refine a metric ground projection from sparse points and unlabeled line families.

    Every image line constrains one of the two orthogonal ground-plane vanishing
    points.  Three lines are therefore useful: two can establish one direction,
    while the third constrains the other direction together with the metric points.
    """
    initial = np.asarray(initial_homography, dtype=np.float64)
    ground = np.asarray(ground_points, dtype=np.float64)
    image = np.asarray(image_points, dtype=np.float64)
    lines = tuple(_normalise_line(item) for item in image_lines)
    if initial.shape != (3, 3) or ground.shape[0] < 3 or ground.shape != image.shape:
        raise ValueError("Lijnverfijning vereist een 3x3-startmodel en minimaal drie 2D-puntparen.")
    if len(lines) < 2:
        raise ValueError("Lijnverfijning vereist minimaal twee rechte beeldlijnen.")

    def pack(matrix: np.ndarray) -> np.ndarray:
        matrix = matrix / matrix[2, 2]
        return matrix.reshape(-1)[:8]

    def unpack(values: np.ndarray) -> np.ndarray:
        return np.append(values, 1.0).reshape(3, 3)

    def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
        homogeneous = np.column_stack((points, np.ones(len(points))))
        values = (matrix @ homogeneous.T).T
        return values[:, :2] / values[:, 2:3]

    best = None
    initial_vanishing = tuple(
        initial[:2, axis] / initial[2, axis]
        if abs(float(initial[2, axis])) > 1e-9 else np.asarray((0.0, 0.0))
        for axis in (0, 1)
    )
    initial_horizon = np.cross(
        (*initial_vanishing[0], 1.0), (*initial_vanishing[1], 1.0)
    ).astype(np.float64)
    initial_horizon /= max(float(np.linalg.norm(initial_horizon[:2])), 1e-9)
    axis_spans = np.ptp(ground, axis=0)
    weak_axes = tuple(int(axis) for axis in np.flatnonzero(axis_spans < 1.0))
    scale_prior_ground = []
    ground_center = np.mean(ground, axis=0)
    for axis in weak_axes:
        for distance in (5.0, 10.0):
            point = ground_center.copy()
            point[axis] += distance
            scale_prior_ground.append(point)
    scale_prior_ground = np.asarray(scale_prior_ground, dtype=np.float64)
    scale_prior_image = (
        project(initial, scale_prior_ground)
        if len(scale_prior_ground) else np.empty((0, 2), dtype=np.float64)
    )
    assignments = [
        tuple((mask >> index) & 1 for index in range(len(lines)))
        for mask in range(1 << len(lines))
    ]
    if len(lines) == 3:
        pair_candidates = []
        for first in range(3):
            for second in range(first + 1, 3):
                point = np.cross(lines[first], lines[second])
                if abs(float(point[2])) < 1e-9:
                    continue
                point /= point[2]
                horizon_distance = abs(float(initial_horizon @ point))
                pair_candidates.append((horizon_distance, first, second))
        if pair_candidates:
            _distance, first, second = min(pair_candidates)
            base = tuple(0 if index in (first, second) else 1 for index in range(3))
            assignments = [base, tuple(1 - item for item in base)]
    for assignment in assignments:
        if len(set(assignment)) < 2:
            continue

        def residual(values: np.ndarray) -> np.ndarray:
            matrix = unpack(values)
            point_errors = (project(matrix, ground) - image).reshape(-1) * 3.0
            line_errors = []
            for line, axis in zip(lines, assignment):
                vanishing = matrix[:, axis]
                if abs(float(vanishing[2])) < 1e-9:
                    line_errors.append(500.0)
                else:
                    line_errors.append(float(line @ (vanishing / vanishing[2])) * 2.0)
            vanishing_prior = []
            for axis in (0, 1):
                vanishing = matrix[:, axis]
                if abs(float(vanishing[2])) < 1e-9:
                    vanishing_prior.extend((50.0, 50.0))
                else:
                    vanishing_prior.extend(
                        ((vanishing[:2] / vanishing[2] - initial_vanishing[axis]) * 0.01).tolist()
                    )
            current_points = []
            for axis in (0, 1):
                vanishing = matrix[:, axis]
                current_points.append(vanishing[:2] / vanishing[2])
            current_horizon = np.cross(
                (*current_points[0], 1.0), (*current_points[1], 1.0)
            ).astype(np.float64)
            current_horizon /= max(float(np.linalg.norm(current_horizon[:2])), 1e-9)
            if float(current_horizon[:2] @ initial_horizon[:2]) < 0.0:
                current_horizon *= -1.0
            horizon_prior = (current_horizon[:2] - initial_horizon[:2]) * 80.0
            scale_prior = (
                (project(matrix, scale_prior_ground) - scale_prior_image).reshape(-1) * 0.35
                if len(scale_prior_ground) else np.empty(0, dtype=np.float64)
            )
            regularization = (values - pack(initial)) * 2e-4
            return np.concatenate(
                (
                    point_errors,
                    np.asarray(line_errors),
                    np.asarray(vanishing_prior),
                    horizon_prior,
                    scale_prior,
                    regularization,
                )
            )

        optimum = least_squares(
            residual,
            pack(initial),
            loss="soft_l1",
            f_scale=3.0,
            max_nfev=800,
        )
        matrix = unpack(optimum.x)
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            continue
        point_errors = np.linalg.norm(project(matrix, ground) - image, axis=1)
        line_errors = np.asarray(
            [abs(float(line @ (matrix[:, axis] / matrix[2, axis]))) for line, axis in zip(lines, assignment)],
            dtype=np.float64,
        )
        vanishing_drift = np.mean(
            [
                np.linalg.norm(matrix[:2, axis] / matrix[2, axis] - initial_vanishing[axis])
                for axis in (0, 1)
                if abs(float(matrix[2, axis])) > 1e-9
            ]
        )
        current_vanishing = tuple(matrix[:2, axis] / matrix[2, axis] for axis in (0, 1))
        current_horizon = np.cross(
            (*current_vanishing[0], 1.0), (*current_vanishing[1], 1.0)
        ).astype(np.float64)
        current_horizon /= max(float(np.linalg.norm(current_horizon[:2])), 1e-9)
        horizon_cosine = np.clip(
            abs(float(current_horizon[:2] @ initial_horizon[:2])), 0.0, 1.0
        )
        horizon_angle = float(np.degrees(np.arccos(horizon_cosine)))
        score = (
            float(np.sqrt(np.mean(np.square(point_errors))))
            + float(np.sqrt(np.mean(np.square(line_errors))))
            + 0.01 * float(vanishing_drift)
            + 0.5 * horizon_angle
        )
        candidate = (score, matrix, point_errors, line_errors, assignment)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("De handmatige lijnen leveren geen omkeerbare grondhomography.")
    _score, matrix, point_errors, line_errors, assignment = best
    return ManualLineHomographyRefinement(
        matrix / matrix[2, 2],
        float(np.sqrt(np.mean(np.square(point_errors)))),
        float(np.max(point_errors)),
        float(np.sqrt(np.mean(np.square(line_errors)))),
        assignment,
    )


def _normalise_line(line: np.ndarray) -> np.ndarray:
    value = np.asarray(line, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("Beeldlijn moet uit drie eindige coëfficiënten bestaan.")
    normal = float(np.linalg.norm(value[:2]))
    if normal < 1e-9:
        raise ValueError("Beeldlijn heeft geen geldige richting.")
    return value / normal
