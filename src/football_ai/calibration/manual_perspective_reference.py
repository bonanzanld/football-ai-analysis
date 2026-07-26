from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from itertools import combinations

import cv2
import numpy as np


class PerspectiveDirection(str, Enum):
    BETWEEN_GOALS = "between_goals"
    ALONG_END_LINES = "along_end_lines"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ManualReferenceLine:
    direction: PerspectiveDirection
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError("Een handmatige referentielijn vereist minimaal drie punten.")
        values = np.asarray(self.points, dtype=np.float64)
        if values.shape[1:] != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("Referentielijn bevat ongeldige beeldpunten.")

    def equation(self) -> np.ndarray:
        points = np.asarray(self.points, dtype=np.float64)
        center = np.mean(points, axis=0)
        _u, _s, vh = np.linalg.svd(points - center)
        direction = vh[0]
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        line = np.asarray((normal[0], normal[1], -float(normal @ center)))
        return line / max(float(np.linalg.norm(line[:2])), 1e-12)

    def endpoints(self, width: int, height: int) -> tuple[tuple[float, float], tuple[float, float]]:
        line = self.equation()
        candidates = []
        if abs(float(line[1])) > 1e-9:
            for x in (0.0, float(width - 1)):
                y = -(line[0] * x + line[2]) / line[1]
                if -height <= y <= 2 * height:
                    candidates.append((x, float(y)))
        if abs(float(line[0])) > 1e-9:
            for y in (0.0, float(height - 1)):
                x = -(line[1] * y + line[2]) / line[0]
                if -width <= x <= 2 * width:
                    candidates.append((float(x), y))
        if len(candidates) < 2:
            raise ValueError("Referentielijn kan niet in het beeld worden getekend.")
        return candidates[0], candidates[-1]

    def to_dict(self) -> dict:
        return {"direction": self.direction.value, "points": [list(item) for item in self.points]}

    @classmethod
    def from_dict(cls, data: dict) -> "ManualReferenceLine":
        return cls(
            PerspectiveDirection(data["direction"]),
            tuple(tuple(map(float, item)) for item in data["points"]),
        )


@dataclass(frozen=True, slots=True)
class ManualPerspectiveView:
    label: str
    frame_number: int
    time_seconds: float
    lines: tuple[ManualReferenceLine, ...]

    @property
    def perspective_complete(self) -> bool:
        return all(
            sum(item.direction is direction for item in self.lines) >= 2
            for direction in (PerspectiveDirection.BETWEEN_GOALS, PerspectiveDirection.ALONG_END_LINES)
        )

    def vanishing_point(self, direction: PerspectiveDirection) -> tuple[float, float]:
        equations = [item.equation() for item in self.lines if item.direction is direction]
        if len(equations) < 2:
            raise ValueError(f"Minimaal twee lijnen vereist voor {direction.value}.")
        rows = np.asarray(equations, dtype=np.float64)
        _u, _s, vh = np.linalg.svd(rows)
        point = vh[-1]
        if abs(float(point[2])) < 1e-9:
            raise ValueError("De gekozen lijnen leveren een verdwijnpunt op oneindig.")
        point /= point[2]
        return float(point[0]), float(point[1])

    def horizon(self) -> tuple[float, float, float]:
        first = (*self.vanishing_point(PerspectiveDirection.BETWEEN_GOALS), 1.0)
        second = (*self.vanishing_point(PerspectiveDirection.ALONG_END_LINES), 1.0)
        line = np.cross(first, second).astype(np.float64)
        line /= max(float(np.linalg.norm(line[:2])), 1e-12)
        return tuple(map(float, line))

    def to_dict(self) -> dict:
        data = {
            "label": self.label,
            "frame_number": self.frame_number,
            "time_seconds": self.time_seconds,
            "lines": [item.to_dict() for item in self.lines],
            "perspective_complete": self.perspective_complete,
            "vanishing_points": None,
            "horizon": None,
        }
        if self.perspective_complete:
            data["vanishing_points"] = {
                item.value: list(self.vanishing_point(item)) for item in PerspectiveDirection
                if item is not PerspectiveDirection.UNKNOWN
            }
            data["horizon"] = list(self.horizon())
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ManualPerspectiveView":
        return cls(
            str(data["label"]),
            int(data["frame_number"]),
            float(data["time_seconds"]),
            tuple(ManualReferenceLine.from_dict(item) for item in data["lines"]),
        )


@dataclass(frozen=True, slots=True)
class ManualPerspectiveReference:
    video_name: str
    views: tuple[ManualPerspectiveView, ...]

    def __post_init__(self) -> None:
        if tuple(item.label for item in self.views) != ("left_goal", "center", "right_goal"):
            raise ValueError("Perspectiefreferentie vereist links, midden en rechts in die volgorde.")

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "video_name": self.video_name,
            "views": [item.to_dict() for item in self.views],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ManualPerspectiveReference":
        return cls(
            str(data["video_name"]),
            tuple(ManualPerspectiveView.from_dict(item) for item in data["views"]),
        )


def save_manual_perspective_reference(reference: ManualPerspectiveReference, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_manual_perspective_reference(path: Path) -> ManualPerspectiveReference:
    return ManualPerspectiveReference.from_dict(json.loads(path.read_text(encoding="utf-8")))


def implied_focal_length(view: ManualPerspectiveView, frame_size: tuple[int, int]) -> float:
    width, height = frame_size
    first = np.asarray(view.vanishing_point(PerspectiveDirection.BETWEEN_GOALS))
    second = np.asarray(view.vanishing_point(PerspectiveDirection.ALONG_END_LINES))
    center = np.asarray((width / 2.0, height / 2.0), dtype=np.float64)
    focal_squared = -float((first - center) @ (second - center))
    if focal_squared <= 0.0:
        raise ValueError("De twee gekozen lijnfamilies leveren geen fysiek mogelijke 90-graden-camera op.")
    return float(np.sqrt(focal_squared))


def automatically_classify_line_directions(
    lines: tuple[ManualReferenceLine, ...],
    frame_size: tuple[int, int],
) -> tuple[ManualReferenceLine, ...]:
    """Split arbitrary straight pitch lines into two perspective families."""
    if len(lines) < 4:
        raise ValueError("Minimaal vier verschillende rechte lijnen vereist.")
    width, height = frame_size
    center = np.asarray((width / 2.0, height / 2.0), dtype=np.float64)
    best = None
    indices = tuple(range(len(lines)))
    for size in range(2, len(lines) - 1):
        for first_group in combinations(indices[1:], size - 1):
            first_indices = (0, *first_group)
            second_indices = tuple(item for item in indices if item not in first_indices)
            if len(second_indices) < 2:
                continue
            first_point = _fit_vanishing_point(tuple(lines[item] for item in first_indices))
            second_point = _fit_vanishing_point(tuple(lines[item] for item in second_indices))
            if first_point is None or second_point is None:
                continue
            focal_squared = -float((first_point - center) @ (second_point - center))
            if focal_squared <= (0.12 * width) ** 2:
                continue
            focal = float(np.sqrt(focal_squared))
            if not 0.15 * width <= focal <= 6.0 * width:
                continue
            residuals = [
                _line_angle_error(lines[item], first_point) for item in first_indices
            ] + [
                _line_angle_error(lines[item], second_point) for item in second_indices
            ]
            score = float(np.sqrt(np.mean(np.square(residuals))))
            balance_penalty = 0.05 * abs(len(first_indices) - len(second_indices))
            candidate = (score + balance_penalty, first_indices, second_indices)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None or best[0] > 4.0:
        raise ValueError(
            "De gekozen lijnen vormen niet betrouwbaar twee parallelle richtingsgroepen. "
            "Kies meer verspreide rechte kalklijnen."
        )
    first_indices, second_indices = set(best[1]), set(best[2])
    return tuple(
        ManualReferenceLine(
            PerspectiveDirection.BETWEEN_GOALS if index in first_indices else PerspectiveDirection.ALONG_END_LINES,
            line.points,
        )
        for index, line in enumerate(lines)
    )


def _fit_vanishing_point(lines: tuple[ManualReferenceLine, ...]) -> np.ndarray | None:
    equations = np.asarray([item.equation() for item in lines], dtype=np.float64)
    _u, _s, vh = np.linalg.svd(equations)
    point = vh[-1]
    if abs(float(point[2])) < 1e-9:
        return None
    return point[:2] / point[2]


def _line_angle_error(line: ManualReferenceLine, vanishing_point: np.ndarray) -> float:
    points = np.asarray(line.points, dtype=np.float64)
    direction = points[-1] - points[0]
    toward = vanishing_point - np.mean(points, axis=0)
    denominator = float(np.linalg.norm(direction) * np.linalg.norm(toward))
    if denominator < 1e-9:
        return 90.0
    cosine = np.clip(abs(float(direction @ toward)) / denominator, 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def assess_three_view_consistency(
    views: tuple[ManualPerspectiveView, ...],
    frame_size: tuple[int, int],
    maximum_focal_ratio: float = 2.0,
) -> tuple[bool, tuple[float, ...], str]:
    complete_views = tuple(view for view in views if view.perspective_complete)
    if len(complete_views) < 2:
        return False, (), "PARTIAL: minder dan twee camerabeelden bepalen zelfstandig beide richtingen."
    focals = tuple(implied_focal_length(view, frame_size) for view in complete_views)
    ratio = max(focals) / max(min(focals), 1e-9)
    if ratio > maximum_focal_ratio:
        return False, focals, f"Brandpuntschattingen verschillen te sterk ({ratio:.1f}x; maximaal {maximum_focal_ratio:.1f}x)."
    return True, focals, "Drie perspectiefbeelden zijn onderling fysiek consistent."


def assess_global_readiness(
    views: tuple[ManualPerspectiveView, ...],
) -> tuple[bool, str]:
    if len(views) != 3 or any(len(view.lines) < 2 for view in views):
        return False, "Ieder van de drie camerabeelden vereist minimaal twee lijnen."
    complete = sum(view.perspective_complete for view in views)
    total_lines = sum(len(view.lines) for view in views)
    if complete >= 1 and total_lines >= 8:
        return True, (
            "Minimaal één beeld bepaalt beide richtingen en de overige beelden leveren "
            "voldoende lijnsteun voor de globale camerakoppeling."
        )
    return False, "Nog geen volledig richtingsbeeld of onvoldoende gezamenlijke lijnsteun."


def draw_manual_perspective_view(frame: np.ndarray, view: ManualPerspectiveView) -> np.ndarray:
    preview = frame.copy()
    colors = {
        PerspectiveDirection.BETWEEN_GOALS: (0, 255, 255),
        PerspectiveDirection.ALONG_END_LINES: (255, 255, 0),
        PerspectiveDirection.UNKNOWN: (0, 165, 255),
    }
    for line in view.lines:
        start, end = line.endpoints(frame.shape[1], frame.shape[0])
        cv2.line(preview, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), colors[line.direction], 3, cv2.LINE_AA)
        for point in line.points:
            cv2.circle(preview, tuple(np.round(point).astype(int)), 6, (255, 0, 255), -1, cv2.LINE_AA)
    if view.perspective_complete:
        for direction in (PerspectiveDirection.BETWEEN_GOALS, PerspectiveDirection.ALONG_END_LINES):
            point = np.asarray(view.vanishing_point(direction))
            if 0 <= point[0] < frame.shape[1] and 0 <= point[1] < frame.shape[0]:
                cv2.circle(preview, tuple(np.round(point).astype(int)), 10, colors[direction], 3, cv2.LINE_AA)
        horizon = view.horizon()
        if abs(horizon[1]) > 1e-9:
            p1 = (0, int(round(-horizon[2] / horizon[1])))
            p2 = (frame.shape[1] - 1, int(round(-(horizon[0] * (frame.shape[1] - 1) + horizon[2]) / horizon[1])))
            cv2.line(preview, p1, p2, (0, 128, 255), 3, cv2.LINE_AA)
    return preview
