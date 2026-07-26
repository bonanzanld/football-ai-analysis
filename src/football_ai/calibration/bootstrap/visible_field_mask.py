from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from football_ai.calibration.bootstrap.goal_seed import GoalSeed, estimate_backline_endpoints
from football_ai.calibration.field_zone import FieldZone


@dataclass(frozen=True, slots=True)
class VisibleFieldMask:
    frame_number: int
    polygon: np.ndarray
    frame_area_ratio: float
    tracking_polygon: np.ndarray

    def contains(self, point: tuple[float, float], margin_pixels: float = 0.0) -> bool:
        return cv2.pointPolygonTest(self.polygon.astype(np.float32), point, True) >= -margin_pixels

    def classify(self, point: tuple[float, float], edge_margin_pixels: float = 12.0) -> FieldZone:
        distance = cv2.pointPolygonTest(self.polygon.astype(np.float32), point, True)
        if distance > edge_margin_pixels:
            return FieldZone.INSIDE
        if distance >= -edge_margin_pixels:
            return FieldZone.EDGE
        return FieldZone.OUTSIDE


@dataclass(frozen=True, slots=True)
class FieldBoundaryGeometry:
    backline: np.ndarray
    rear_sideline: np.ndarray
    front_sideline: np.ndarray
    interior: np.ndarray


def build_field_boundary_geometry(seed: GoalSeed, pitch_width_m: float) -> FieldBoundaryGeometry:
    if seed.rear_sideline_support is None or seed.front_sideline_support is None:
        raise ValueError("Beide zijlijnsteunpunten zijn vereist.")
    rear, front = estimate_backline_endpoints(
        seed.first_ground,
        seed.second_ground,
        seed.goal_width_m,
        pitch_width_m,
        seed.rear_corner,
        seed.front_corner,
    )
    rear = np.asarray(rear, dtype=np.float64)
    front = np.asarray(front, dtype=np.float64)
    rear_support = np.asarray(seed.rear_sideline_support, dtype=np.float64)
    front_support = np.asarray(seed.front_sideline_support, dtype=np.float64)
    return FieldBoundaryGeometry(
        backline=np.asarray((rear, front)),
        rear_sideline=np.asarray((rear, rear_support)),
        front_sideline=np.asarray((front, front_support)),
        interior=(rear_support + front_support) / 2.0,
    )


def polygon_from_field_boundaries(
    geometry: FieldBoundaryGeometry,
    frame_size: tuple[int, int],
    *,
    include_backline: bool,
) -> np.ndarray:
    polygon = _rectangle(0.0, 0.0, frame_size[0] - 1.0, frame_size[1] - 1.0)
    boundaries = [geometry.rear_sideline, geometry.front_sideline]
    if include_backline:
        boundaries.insert(0, geometry.backline)
    for boundary in boundaries:
        polygon = _clip_to_half_plane(polygon, boundary[0], boundary[1], geometry.interior)
        if len(polygon) < 3:
            return np.empty((0, 2), dtype=np.float64)
    return polygon


def interpolate_sideline_geometry(
    first: FieldBoundaryGeometry,
    second: FieldBoundaryGeometry,
    fraction: float,
    frame_size: tuple[int, int],
) -> FieldBoundaryGeometry:
    """Blend the same two fixed pitch sidelines between camera end states."""
    amount = float(np.clip(fraction, 0.0, 1.0))
    rear = _interpolate_line(first.rear_sideline, second.rear_sideline, amount, frame_size)
    front = _interpolate_line(first.front_sideline, second.front_sideline, amount, frame_size)
    interior = (1.0 - amount) * first.interior + amount * second.interior
    return FieldBoundaryGeometry(
        backline=(1.0 - amount) * first.backline + amount * second.backline,
        rear_sideline=rear,
        front_sideline=front,
        interior=interior,
    )


def _interpolate_line(
    first: np.ndarray,
    second: np.ndarray,
    fraction: float,
    frame_size: tuple[int, int],
) -> np.ndarray:
    first_line = np.cross(np.append(first[0], 1.0), np.append(first[1], 1.0))
    second_line = np.cross(np.append(second[0], 1.0), np.append(second[1], 1.0))
    first_line /= max(np.hypot(first_line[0], first_line[1]), 1e-9)
    second_line /= max(np.hypot(second_line[0], second_line[1]), 1e-9)
    if float(np.dot(first_line[:2], second_line[:2])) < 0.0:
        second_line *= -1.0
    line = (1.0 - fraction) * first_line + fraction * second_line
    line /= max(np.hypot(line[0], line[1]), 1e-9)
    a, b, c = line
    width, height = frame_size
    if abs(b) >= abs(a):
        return np.asarray(((0.0, -c / b), (width - 1.0, -(a * (width - 1.0) + c) / b)))
    return np.asarray(((-c / a, 0.0), (-(b * (height - 1.0) + c) / a, height - 1.0)))


def build_visible_field_mask(
    seed: GoalSeed,
    pitch_width_m: float,
    frame_size: tuple[int, int],
    *,
    include_backline: bool = True,
) -> VisibleFieldMask:
    geometry = build_field_boundary_geometry(seed, pitch_width_m)
    rear, front = geometry.backline
    rear_support = geometry.rear_sideline[1]
    front_support = geometry.front_sideline[1]
    interior = geometry.interior
    frame_polygon = _rectangle(0.0, 0.0, frame_size[0] - 1.0, frame_size[1] - 1.0)
    margin = float(max(frame_size) * 12)
    tracking_polygon = _rectangle(-margin, -margin, frame_size[0] + margin, frame_size[1] + margin)
    boundaries = [(rear, rear_support), (front, front_support)]
    if include_backline:
        boundaries.insert(0, (rear, front))
    for first, second in boundaries:
        frame_polygon = _clip_to_half_plane(frame_polygon, first, second, interior)
        tracking_polygon = _clip_to_half_plane(tracking_polygon, first, second, interior)
        if len(frame_polygon) < 3 or len(tracking_polygon) < 3:
            raise ValueError("De gekozen veldlijnen omsluiten geen zichtbaar speelveld.")
    area = abs(float(cv2.contourArea(frame_polygon.astype(np.float32))))
    ratio = area / float(frame_size[0] * frame_size[1])
    if ratio < 0.02:
        raise ValueError("Het zichtbare veldmasker is onrealistisch klein.")
    return VisibleFieldMask(seed.frame_number, frame_polygon, ratio, tracking_polygon)


def clip_polygon_to_frame(polygon: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """Clip a convex tracked field polygon to the current image rectangle."""
    output = np.asarray(polygon, dtype=np.float64)
    width, height = frame_size
    constraints = (
        (np.asarray((0.0, 0.0)), np.asarray((width - 1.0, 0.0)), np.asarray((0.0, 1.0))),
        (np.asarray((width - 1.0, 0.0)), np.asarray((width - 1.0, height - 1.0)), np.asarray((width - 2.0, 0.0))),
        (np.asarray((width - 1.0, height - 1.0)), np.asarray((0.0, height - 1.0)), np.asarray((0.0, height - 2.0))),
        (np.asarray((0.0, height - 1.0)), np.asarray((0.0, 0.0)), np.asarray((1.0, 0.0))),
    )
    for first, second, interior in constraints:
        output = _clip_to_half_plane(output, first, second, interior)
        if len(output) < 3:
            return np.empty((0, 2), dtype=np.float64)
    return output


def _rectangle(left: float, top: float, right: float, bottom: float) -> np.ndarray:
    return np.asarray([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float64)


def _clip_to_half_plane(polygon: np.ndarray, first: np.ndarray, second: np.ndarray, interior: np.ndarray) -> np.ndarray:
    direction = second - first
    def cross(point: np.ndarray) -> float:
        offset = point - first
        return float(direction[0] * offset[1] - direction[1] * offset[0])
    keep_sign = np.sign(cross(interior)) or 1.0
    def signed(point: np.ndarray) -> float:
        return float(keep_sign * cross(point))
    output = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        start_value, end_value = signed(start), signed(end)
        start_inside, end_inside = start_value >= 0.0, end_value >= 0.0
        if start_inside:
            output.append(start)
        if start_inside != end_inside:
            alpha = start_value / (start_value - end_value)
            output.append(start + alpha * (end - start))
    return np.asarray(output, dtype=np.float64)
