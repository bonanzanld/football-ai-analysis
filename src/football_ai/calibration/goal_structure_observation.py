from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class GoalStructureLine:
    name: str
    points: tuple[tuple[float, float], ...]
    equation: tuple[float, float, float]
    rms_error_px: float
    maximum_error_px: float

    @classmethod
    def fit(cls, name: str, points: tuple[tuple[float, float], ...]) -> "GoalStructureLine":
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 5:
            raise ValueError("Een doelstructuurlijn vereist minimaal vijf beeldpunten.")
        center = np.median(values, axis=0)
        _u, _s, vh = np.linalg.svd(values - center, full_matrices=False)
        direction = vh[0]
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        offsets = (values - center) @ normal
        # Refit once after rejecting a clearly accidental click.
        scale = max(float(np.median(np.abs(offsets))) * 1.4826, 1.0)
        keep = np.abs(offsets) <= max(4.0 * scale, 4.0)
        if int(np.count_nonzero(keep)) >= 4 and not np.all(keep):
            center = np.mean(values[keep], axis=0)
            _u, _s, vh = np.linalg.svd(values[keep] - center, full_matrices=False)
            direction = vh[0]
            normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        equation = np.asarray((normal[0], normal[1], -float(normal @ center)))
        equation /= max(float(np.linalg.norm(equation[:2])), 1e-12)
        errors = np.abs(np.column_stack((values, np.ones(len(values)))) @ equation)
        return cls(
            name,
            tuple(tuple(map(float, item)) for item in values),
            tuple(map(float, equation)),
            float(np.sqrt(np.mean(np.square(errors)))),
            float(np.max(errors)),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "points": [list(item) for item in self.points],
            "equation": list(self.equation),
            "rms_error_px": self.rms_error_px,
            "maximum_error_px": self.maximum_error_px,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoalStructureLine":
        return cls(
            str(data["name"]),
            tuple(tuple(map(float, item)) for item in data["points"]),
            tuple(map(float, data["equation"])),
            float(data["rms_error_px"]),
            float(data["maximum_error_px"]),
        )


@dataclass(frozen=True, slots=True)
class GoalStructureObservation:
    goal_id: str
    frame_number: int
    time_seconds: float
    lines: tuple[GoalStructureLine, ...]

    def __post_init__(self) -> None:
        expected = ("far_post", "crossbar", "near_post", "goal_line")
        if tuple(item.name for item in self.lines) != expected:
            raise ValueError("Doelstructuur vereist verste paal, lat, dichtstbijzijnde paal en doellijn.")

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "frame_number": self.frame_number,
            "time_seconds": self.time_seconds,
            "lines": [item.to_dict() for item in self.lines],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoalStructureObservation":
        return cls(
            str(data["goal_id"]),
            int(data["frame_number"]),
            float(data["time_seconds"]),
            tuple(GoalStructureLine.from_dict(item) for item in data["lines"]),
        )

    def corners(self) -> dict[str, tuple[float, float]]:
        lines = {item.name: np.asarray(item.equation, dtype=np.float64) for item in self.lines}

        def intersection(first: str, second: str) -> tuple[float, float]:
            point = np.cross(lines[first], lines[second])
            if abs(float(point[2])) < 1e-9:
                raise ValueError(f"Doellijnen {first} en {second} snijden niet betrouwbaar.")
            return float(point[0] / point[2]), float(point[1] / point[2])

        return {
            "far_bottom": intersection("far_post", "goal_line"),
            "far_top": intersection("far_post", "crossbar"),
            "near_top": intersection("near_post", "crossbar"),
            "near_bottom": intersection("near_post", "goal_line"),
        }


def save_goal_structure_observations(
    observations: tuple[GoalStructureObservation, ...], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": 1, "goals": [item.to_dict() for item in observations]},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_goal_structure_observations(path: Path) -> tuple[GoalStructureObservation, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(GoalStructureObservation.from_dict(item) for item in data["goals"])
