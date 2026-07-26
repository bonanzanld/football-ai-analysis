from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from football_ai.calibration.bootstrap.goal_seed import GoalSeed
from football_ai.calibration.bootstrap.sideline_anchor import SidelineAnchor
from football_ai.calibration.bootstrap.visible_field_mask import (
    FieldBoundaryGeometry,
    build_field_boundary_geometry,
    interpolate_sideline_geometry,
)


@dataclass(frozen=True, slots=True)
class FieldViewAnchor:
    anchor_id: str
    camera_state: int
    view_position: float
    frame_number: int
    time_seconds: float
    rear_line: np.ndarray | None
    front_line: np.ndarray | None
    backline: np.ndarray | None
    interior: np.ndarray

    @property
    def observed_boundary_count(self) -> int:
        return sum(item is not None for item in (self.rear_line, self.front_line, self.backline))


def build_field_anchor_bank(
    goals: tuple[GoalSeed, GoalSeed],
    sidelines: tuple[SidelineAnchor, ...],
    pitch_width_m: float,
    frame_size: tuple[int, int],
) -> tuple[FieldViewAnchor, ...]:
    goal_items = sorted(goals, key=lambda item: item.view_position)
    first_geometry = build_field_boundary_geometry(goal_items[0], pitch_width_m)
    second_geometry = build_field_boundary_geometry(goal_items[-1], pitch_width_m)
    anchors: list[FieldViewAnchor] = [
        _goal_anchor(goal_items[0], first_geometry),
        _goal_anchor(goal_items[-1], second_geometry),
    ]
    position_span = max(goal_items[-1].view_position - goal_items[0].view_position, 1e-9)
    for item in sidelines:
        fraction = (item.view_position - goal_items[0].view_position) / position_span
        predicted = interpolate_sideline_geometry(
            first_geometry, second_geometry, fraction, frame_size
        )
        rear = _parallel_line_through(predicted.rear_sideline, item.rear_point)
        front = _parallel_line_through(predicted.front_sideline, item.front_point)
        anchors.append(
            FieldViewAnchor(
                anchor_id=f"stand-{item.camera_state}",
                camera_state=item.camera_state,
                view_position=item.view_position,
                frame_number=item.frame_number,
                time_seconds=item.time_seconds,
                rear_line=rear,
                front_line=front,
                backline=None,
                interior=_choose_interior(rear, front, frame_size),
            )
        )
    return tuple(sorted(anchors, key=lambda item: item.view_position))


def anchor_visible_polygon(anchor: FieldViewAnchor, frame_size: tuple[int, int]) -> np.ndarray:
    width, height = frame_size
    polygon = np.asarray(
        ((0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)),
        dtype=np.float64,
    )
    for line in (anchor.backline, anchor.rear_line, anchor.front_line):
        if line is None:
            continue
        polygon = _clip_to_line(polygon, line, anchor.interior)
        if len(polygon) < 3:
            return np.empty((0, 2), dtype=np.float64)
    return polygon


def _goal_anchor(seed: GoalSeed, geometry: FieldBoundaryGeometry) -> FieldViewAnchor:
    return FieldViewAnchor(
        anchor_id=f"doel-{seed.goal_id}",
        camera_state=seed.camera_state,
        view_position=seed.view_position,
        frame_number=seed.frame_number,
        time_seconds=seed.time_seconds,
        rear_line=geometry.rear_sideline,
        front_line=geometry.front_sideline,
        backline=geometry.backline,
        interior=geometry.interior,
    )


def _parallel_line_through(
    predicted_line: np.ndarray,
    point: tuple[float, float] | None,
) -> np.ndarray | None:
    if point is None:
        return None
    first, second = np.asarray(predicted_line, dtype=np.float64)
    direction = second - first
    selected = np.asarray(point, dtype=np.float64)
    return np.asarray((selected - direction, selected + direction), dtype=np.float64)


def _choose_interior(
    rear: np.ndarray | None,
    front: np.ndarray | None,
    frame_size: tuple[int, int],
) -> np.ndarray:
    width, height = frame_size
    candidates = [np.asarray((width * 0.5, height * 0.62), dtype=np.float64)]
    for line in (rear, front):
        if line is not None:
            candidates.append(np.mean(line, axis=0))
    return np.mean(candidates, axis=0)


def _clip_to_line(polygon: np.ndarray, line: np.ndarray, interior: np.ndarray) -> np.ndarray:
    first, second = line
    direction = second - first
    def cross(point: np.ndarray) -> float:
        offset = point - first
        return float(direction[0] * offset[1] - direction[1] * offset[0])
    keep_sign = np.sign(cross(interior)) or 1.0
    output: list[np.ndarray] = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        start_value, end_value = keep_sign * cross(start), keep_sign * cross(end)
        start_inside, end_inside = start_value >= 0.0, end_value >= 0.0
        if start_inside:
            output.append(start)
        if start_inside != end_inside:
            amount = start_value / (start_value - end_value)
            output.append(start + amount * (end - start))
    return np.asarray(output, dtype=np.float64)
