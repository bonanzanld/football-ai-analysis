from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ParallelBoundaryQuality:
    valid: bool
    vanishing_point: tuple[float, float]
    boundary_residual_degrees: tuple[float, float]
    maximum_residual_degrees: float

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "vanishing_point": list(self.vanishing_point),
            "boundary_residual_degrees": list(self.boundary_residual_degrees),
            "maximum_residual_degrees": self.maximum_residual_degrees,
        }


def sideline_rays_from_confirmed_endline(
    rear_corner: tuple[float, float],
    front_corner: tuple[float, float],
    direction_vanishing_point: tuple[float, float],
) -> tuple[
    tuple[tuple[float, float], tuple[float, float]],
    tuple[tuple[float, float], tuple[float, float]],
]:
    """Create the only two certain sidelines from one confirmed 8v8 end line.

    The opposite end line is intentionally not inferred: parallel direction
    fixes two projective rays but does not determine the pitch length in image.
    """
    corners = np.asarray((rear_corner, front_corner), dtype=np.float64)
    vanishing = np.asarray(direction_vanishing_point, dtype=np.float64)
    if corners.shape != (2, 2) or vanishing.shape != (2,):
        raise ValueError("Two end-line corners and one vanishing point are required")
    if not np.all(np.isfinite(corners)) or not np.all(np.isfinite(vanishing)):
        raise ValueError("Sideline ray geometry must be finite")
    if np.linalg.norm(corners[0] - corners[1]) < 2.0:
        raise ValueError("Confirmed end-line corners must be distinct")
    if any(np.linalg.norm(corner - vanishing) < 2.0 for corner in corners):
        raise ValueError("Vanishing point must differ from both end-line corners")
    point = tuple(map(float, vanishing))
    return (
        (tuple(map(float, corners[0])), point),
        (tuple(map(float, corners[1])), point),
    )


def sideline_support_deviation_degrees(
    corner: tuple[float, float],
    direction_vanishing_point: tuple[float, float],
    support: tuple[float, float],
    *,
    away_from_vanishing: bool,
) -> float:
    """Measure cone support without allowing it to rotate official geometry."""
    corner_v = np.asarray(corner, dtype=np.float64)
    official = np.asarray(direction_vanishing_point, dtype=np.float64) - corner_v
    if away_from_vanishing:
        official *= -1.0
    observed = np.asarray(support, dtype=np.float64) - corner_v
    if np.linalg.norm(official) < 2.0 or np.linalg.norm(observed) < 25.0:
        raise ValueError("Sideline direction and support must be separated from the corner")
    official_angle = float(np.arctan2(official[1], official[0]))
    observed_angle = float(np.arctan2(observed[1], observed[0]))
    difference = float(np.arctan2(
        np.sin(observed_angle - official_angle),
        np.cos(observed_angle - official_angle),
    ))
    return abs(float(np.degrees(difference)))


def measure_ground_line_angle(
    first_line: tuple[tuple[float, float], tuple[float, float]],
    second_line: tuple[tuple[float, float], tuple[float, float]],
    ground_homography: np.ndarray,
) -> float:
    """Measure the acute angle between two image lines on the metric ground plane."""
    inverse = np.linalg.inv(np.asarray(ground_homography, dtype=np.float64))

    def back_project(point: tuple[float, float]) -> np.ndarray:
        result = inverse @ np.asarray((*point, 1.0), dtype=np.float64)
        if abs(float(result[2])) < 1e-9:
            raise ValueError("Beeldpunt kan niet stabiel naar het grondvlak worden geprojecteerd.")
        return result[:2] / result[2]

    first_direction = back_project(first_line[1]) - back_project(first_line[0])
    second_direction = back_project(second_line[1]) - back_project(second_line[0])
    denominator = float(np.linalg.norm(first_direction) * np.linalg.norm(second_direction))
    if denominator < 1e-9:
        raise ValueError("Grondlijnen moeten twee verschillende punten bevatten.")
    cosine = np.clip(abs(float(first_direction @ second_direction)) / denominator, 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def correct_sidelines_to_ground_perpendicular(
    polygon: np.ndarray,
    anchored_end: str,
    ground_homography: np.ndarray,
) -> np.ndarray:
    """Keep the confirmed end line fixed and rotate both sidelines to metric 90 degrees."""
    corners = np.asarray(polygon, dtype=np.float64).copy()
    homography = np.asarray(ground_homography, dtype=np.float64)
    inverse = np.linalg.inv(homography)
    if anchored_end == "A":
        rear_index, front_index, opposite_rear, opposite_front = 0, 3, 1, 2
    elif anchored_end == "B":
        rear_index, front_index, opposite_rear, opposite_front = 1, 2, 0, 3
    else:
        raise ValueError("anchored_end moet A of B zijn.")

    def back_project(point: np.ndarray) -> np.ndarray:
        result = inverse @ np.asarray((*point, 1.0), dtype=np.float64)
        return result[:2] / result[2]

    def project(point: np.ndarray) -> np.ndarray:
        result = homography @ np.asarray((*point, 1.0), dtype=np.float64)
        return result[:2] / result[2]

    rear_ground = back_project(corners[rear_index])
    front_ground = back_project(corners[front_index])
    backline_direction = front_ground - rear_ground
    length = float(np.linalg.norm(backline_direction))
    if length < 1e-9:
        raise ValueError("Bevestigde achterlijn is te kort voor een haaksheidscorrectie.")
    perpendicular = np.asarray((-backline_direction[1], backline_direction[0])) / length
    candidates = (perpendicular, -perpendicular)
    best = None
    for ground_direction in candidates:
        rear_ray = project(rear_ground + ground_direction) - corners[rear_index]
        front_ray = project(front_ground + ground_direction) - corners[front_index]
        score = float(
            rear_ray @ (corners[opposite_rear] - corners[rear_index])
            + front_ray @ (corners[opposite_front] - corners[front_index])
        )
        if best is None or score > best[0]:
            best = score, rear_ray, front_ray
    assert best is not None
    for anchor_index, moving_index, ray in (
        (rear_index, opposite_rear, best[1]),
        (front_index, opposite_front, best[2]),
    ):
        denominator = float(ray @ ray)
        if denominator < 1e-9:
            raise ValueError("Haakse zijlijnrichting is numeriek instabiel.")
        parameter = float((corners[moving_index] - corners[anchor_index]) @ ray / denominator)
        corners[moving_index] = corners[anchor_index] + parameter * ray
    if not cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2)):
        raise ValueError("Haaksheidscorrectie zou een gekruiste contour maken.")
    return corners


def estimate_vanishing_point_from_lines(
    lines: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
) -> tuple[float, float]:
    if len(lines) < 2:
        raise ValueError("Minimaal twee onafhankelijke beeldlijnen vereist voor een verdwijnpunt.")
    equations = np.asarray([_line_equation(start, end) for start, end in lines])
    _u, _s, vh = np.linalg.svd(equations)
    point = vh[-1]
    if abs(float(point[2])) < 1e-9:
        raise ValueError("Referentielijnen leveren geen eindig verdwijnpunt.")
    point /= point[2]
    return float(point[0]), float(point[1])


def assess_playable_sideline_parallelism(
    polygon: np.ndarray,
    vanishing_point: tuple[float, float],
    threshold_degrees: float = 2.5,
) -> ParallelBoundaryQuality:
    corners = np.asarray(polygon, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError("Speelveldcontour vereist vier beeldhoeken.")
    point = np.asarray(vanishing_point, dtype=np.float64)
    residuals = (
        _direction_residual(corners[0], corners[1], point),
        _direction_residual(corners[3], corners[2], point),
    )
    maximum = max(residuals)
    return ParallelBoundaryQuality(maximum <= threshold_degrees, vanishing_point, residuals, maximum)


def align_playable_sidelines_to_vanishing_point(
    polygon: np.ndarray,
    vanishing_point: tuple[float, float],
    anchored_end: str,
) -> np.ndarray:
    """Minimally move the opposite end so both sidelines share the observed VP."""
    corners = np.asarray(polygon, dtype=np.float64).copy()
    point = np.asarray(vanishing_point, dtype=np.float64)
    if corners.shape != (4, 2) or anchored_end not in {"A", "B"}:
        raise ValueError("Vier hoeken en anchored_end A of B vereist.")
    pairs = ((0, 1), (3, 2)) if anchored_end == "A" else ((1, 0), (2, 3))
    for anchor_index, moving_index in pairs:
        anchor = corners[anchor_index]
        direction = point - anchor
        denominator = float(direction @ direction)
        if denominator < 1e-9:
            raise ValueError("Verdwijnpunt valt samen met een verankerde veldhoek.")
        parameter = float((corners[moving_index] - anchor) @ direction / denominator)
        corners[moving_index] = anchor + parameter * direction
    if not cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2)):
        raise ValueError("Paralleliteitscorrectie zou een gekruiste speelveldcontour maken.")
    return corners


def align_playable_sidelines_to_support_points(
    polygon: np.ndarray,
    anchored_end: str,
    rear_support: tuple[float, float],
    front_support: tuple[float, float],
) -> np.ndarray:
    """Force both sidelines through their clicked cone/line support in one goal view."""
    corners = np.asarray(polygon, dtype=np.float64).copy()
    if corners.shape != (4, 2) or anchored_end not in {"A", "B"}:
        raise ValueError("Vier hoeken en anchored_end A of B vereist.")
    constraints = (
        ((0, 1), np.asarray(rear_support, dtype=np.float64)),
        ((3, 2), np.asarray(front_support, dtype=np.float64)),
    ) if anchored_end == "A" else (
        ((1, 0), np.asarray(rear_support, dtype=np.float64)),
        ((2, 3), np.asarray(front_support, dtype=np.float64)),
    )
    for (anchor_index, moving_index), support in constraints:
        anchor = corners[anchor_index]
        direction = support - anchor
        denominator = float(direction @ direction)
        if denominator < 25.0:
            raise ValueError("Zijlijnsteun ligt te dicht bij de bijbehorende hoek.")
        parameter = float((corners[moving_index] - anchor) @ direction / denominator)
        corners[moving_index] = anchor + parameter * direction
    if not cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2)):
        raise ValueError("Hoek- en hoedjesconstraints zouden een gekruiste speelveldcontour maken.")
    return corners


def rebuild_from_endline_goal_area_and_far_support(
    polygon: np.ndarray,
    anchored_end: str,
    goal_posts: tuple[tuple[float, float], tuple[float, float]],
    goal_area_line: tuple[tuple[float, float], tuple[float, float]],
    far_support: tuple[float, float],
) -> tuple[np.ndarray, tuple[float, float]]:
    """Rebuild one image rectangle from the shared end line and the white 5.5m line."""
    corners = np.asarray(polygon, dtype=np.float64).copy()
    if corners.shape != (4, 2) or anchored_end not in {"A", "B"}:
        raise ValueError("Vier hoeken en anchored_end A of B vereist.")
    rear_index, front_index = (0, 3) if anchored_end == "A" else (1, 2)
    opposite_rear, opposite_front = (1, 2) if anchored_end == "A" else (0, 3)
    endline = _line_equation(*goal_posts)
    near_line = _line_equation(*goal_area_line)
    near_corner = _intersection(endline, near_line)
    far_line = _line_equation(tuple(corners[rear_index]), far_support)
    far_corner = _intersection(endline, far_line)
    vanishing = _intersection(near_line, far_line)
    corners[rear_index] = far_corner
    corners[front_index] = near_corner
    for anchor_index, moving_index in (
        (rear_index, opposite_rear),
        (front_index, opposite_front),
    ):
        anchor = corners[anchor_index]
        direction = vanishing - anchor
        denominator = float(direction @ direction)
        if denominator < 1e-9:
            raise ValueError("Zijlijnverdwijnpunt valt samen met een veldhoek.")
        parameter = float((corners[moving_index] - anchor) @ direction / denominator)
        corners[moving_index] = anchor + parameter * direction
    if not cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2)):
        raise ValueError("5,5-meterlijn en hoedjessteun leveren geen convexe 8v8-contour.")
    return corners, tuple(map(float, vanishing))


def rebuild_from_confirmed_backline_and_ground_horizon(
    polygon: np.ndarray,
    anchored_end: str,
    backline_corners: tuple[tuple[float, float], tuple[float, float]],
    near_support_line: tuple[tuple[float, float], tuple[float, float]],
    ground_homography: np.ndarray,
    parallel_reference_line: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Anchor both end-line corners and derive the sideline VP on the ground horizon."""
    corners = np.asarray(polygon, dtype=np.float64).copy()
    homography = np.asarray(ground_homography, dtype=np.float64)
    if corners.shape != (4, 2) or homography.shape != (3, 3):
        raise ValueError("Vier beeldhoeken en een 3x3 grondhomography vereist.")
    if anchored_end == "A":
        rear_index, front_index, opposite_rear, opposite_front = 0, 3, 1, 2
    elif anchored_end == "B":
        rear_index, front_index, opposite_rear, opposite_front = 1, 2, 0, 3
    else:
        raise ValueError("anchored_end moet A of B zijn.")

    rear_corner = np.asarray(backline_corners[0], dtype=np.float64)
    front_corner = np.asarray(backline_corners[1], dtype=np.float64)
    observed_start = np.asarray(near_support_line[0], dtype=np.float64)
    observed_end = np.asarray(near_support_line[1], dtype=np.float64)
    observed_direction = observed_end - observed_start
    if float(np.linalg.norm(observed_direction)) < 40.0:
        raise ValueError("De gemiddelde 5,5m-/hoedjeslijn is te kort.")

    ground_line_at_infinity = np.asarray((0.0, 0.0, 1.0))
    image_horizon = np.linalg.inv(homography).T @ ground_line_at_infinity
    near_line = _line_equation(
        tuple(front_corner),
        tuple(front_corner + observed_direction),
    )
    if parallel_reference_line is None:
        vanishing = _intersection(near_line, image_horizon)
    else:
        reference_line = _line_equation(*parallel_reference_line)
        vanishing = _intersection(near_line, reference_line)

    corners[rear_index] = rear_corner
    corners[front_index] = front_corner
    for anchor_index, moving_index in (
        (rear_index, opposite_rear),
        (front_index, opposite_front),
    ):
        anchor = corners[anchor_index]
        direction = vanishing - anchor
        denominator = float(direction @ direction)
        if denominator < 1e-9:
            raise ValueError("Zijlijnverdwijnpunt valt samen met een veldhoek.")
        parameter = float((corners[moving_index] - anchor) @ direction / denominator)
        corners[moving_index] = anchor + parameter * direction
    if not cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2)):
        raise ValueError("Bevestigde achterlijn en grondhorizon leveren geen convexe contour.")
    return corners, tuple(map(float, vanishing))


def detect_long_white_right_reference(
    frame: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Detect the long, near-vertical 11v11 marking in the right image region."""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Een BGR-kleurenframe is vereist.")
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 155), (180, 95, 255))
    white[: int(0.28 * height)] = 0
    white[:, : int(0.68 * width)] = 0
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    detected = cv2.HoughLinesP(
        white,
        1,
        np.pi / 180.0,
        45,
        minLineLength=max(int(0.14 * height), 40),
        maxLineGap=28,
    )
    if detected is None:
        return None
    candidates = []
    for x1, y1, x2, y2 in np.asarray(detected).reshape(-1, 4):
        direction = np.asarray((x2 - x1, y2 - y1), dtype=np.float64)
        length = float(np.linalg.norm(direction))
        angle = abs(float(np.degrees(np.arctan2(direction[1], direction[0]))))
        angle = min(angle, 180.0 - angle)
        if angle >= 60.0:
            candidates.append((length, (float(x1), float(y1)), (float(x2), float(y2))))
    if not candidates:
        return None
    _length, start, end = max(candidates, key=lambda item: item[0])
    return start, end


def _line_equation(start: tuple[float, float], end: tuple[float, float]) -> np.ndarray:
    line = np.cross(np.asarray((*start, 1.0)), np.asarray((*end, 1.0)))
    return line / max(float(np.linalg.norm(line[:2])), 1e-12)


def _intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    point = np.cross(first, second)
    if abs(float(point[2])) < 1e-9:
        raise ValueError("Benodigde beeldlijnen zijn parallel of numeriek instabiel.")
    return point[:2] / point[2]


def _direction_residual(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    midpoint = (start + end) / 2.0
    direction = end - start
    toward = point - midpoint
    denominator = float(np.linalg.norm(direction) * np.linalg.norm(toward))
    if denominator < 1e-9:
        return 90.0
    cosine = np.clip(abs(float(direction @ toward)) / denominator, 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))
