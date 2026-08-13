from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from football_ai.calibration.global_frame_graph import FrameGraphEdge


@dataclass(frozen=True, slots=True)
class AnchoredGoalProjection:
    ground_points: tuple[tuple[float, float], tuple[float, float]]
    top_points: tuple[tuple[float, float], tuple[float, float]]
    model_disagreement_px: float
    valid: bool


def contiguous_goal_windows(
    times_seconds: tuple[float, ...],
    *,
    maximum_gap_seconds: float,
) -> tuple[tuple[float, float, int], ...]:
    """Group accepted samples into deterministic local tracking windows."""
    if maximum_gap_seconds <= 0:
        raise ValueError("Maximum gap must be positive")
    times = sorted(set(float(item) for item in times_seconds))
    if not times:
        return ()
    windows = []
    start = previous = times[0]
    count = 1
    for current in times[1:]:
        if current - previous > maximum_gap_seconds:
            windows.append((start, previous, count))
            start, count = current, 1
        else:
            count += 1
        previous = current
    windows.append((start, previous, count))
    return tuple(windows)


def project_anchored_goal(
    ground_points: tuple[tuple[float, float], tuple[float, float]],
    top_points: tuple[tuple[float, float], tuple[float, float]],
    full_frame_edge: FrameGraphEdge,
    ground_frame_edge: FrameGraphEdge,
    *,
    maximum_model_disagreement_px: float = 12.0,
) -> AnchoredGoalProjection:
    """Propagate a confirmed goal while guarding against parallax/model drift."""
    ground = np.asarray(ground_points, dtype=np.float32)
    top = np.asarray(top_points, dtype=np.float32)
    projected_ground = _project(ground, ground_frame_edge.source_to_target)
    full_ground = _project(ground, full_frame_edge.source_to_target)
    projected_top = _project(top, full_frame_edge.source_to_target)
    disagreement = float(np.max(np.linalg.norm(projected_ground - full_ground, axis=1)))
    valid = bool(
        disagreement <= maximum_model_disagreement_px
        and full_frame_edge.inliers >= 30
        and ground_frame_edge.inliers >= 30
    )
    return AnchoredGoalProjection(
        tuple(tuple(map(float, item)) for item in projected_ground),
        tuple(tuple(map(float, item)) for item in projected_top),
        disagreement,
        valid,
    )


def project_anchored_goal_line(
    ground_points: tuple[tuple[float, float], tuple[float, float]],
    full_frame_edge: FrameGraphEdge,
    ground_frame_edge: FrameGraphEdge,
    *,
    maximum_model_disagreement_px: float = 8.0,
) -> AnchoredGoalProjection:
    """Propagate goal feet when no reliable crossbar annotations exist."""
    ground = np.asarray(ground_points, dtype=np.float32)
    projected_ground = _project(ground, ground_frame_edge.source_to_target)
    full_ground = _project(ground, full_frame_edge.source_to_target)
    disagreement = float(np.max(np.linalg.norm(projected_ground - full_ground, axis=1)))
    valid = bool(
        disagreement <= maximum_model_disagreement_px
        and full_frame_edge.inliers >= 30
        and ground_frame_edge.inliers >= 30
    )
    points = tuple(tuple(map(float, item)) for item in projected_ground)
    return AnchoredGoalProjection(points, (), disagreement, valid)


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(1, -1, 2), matrix).reshape(-1, 2)
