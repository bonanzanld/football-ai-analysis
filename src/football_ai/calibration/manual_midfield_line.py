from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ManualMidfieldLine:
    """A manually observed 11v11 halfway line in one video frame."""

    video_name: str
    frame_number: int
    time_seconds: float
    points: tuple[tuple[float, float], ...]
    equation: tuple[float, float, float]
    rms_error_px: float
    maximum_error_px: float
    rear_sideline_point: tuple[float, float] | None = None
    front_sideline_point: tuple[float, float] | None = None
    rear_sideline_frame_number: int | None = None
    front_sideline_frame_number: int | None = None

    @classmethod
    def fit(
        cls,
        video_name: str,
        frame_number: int,
        time_seconds: float,
        points: tuple[tuple[float, float], ...],
        rear_sideline_point: tuple[float, float] | None = None,
        front_sideline_point: tuple[float, float] | None = None,
        rear_sideline_frame_number: int | None = None,
        front_sideline_frame_number: int | None = None,
    ) -> "ManualMidfieldLine":
        values = np.asarray(points, dtype=np.float64)
        if values.shape != (5, 2) or not np.all(np.isfinite(values)):
            raise ValueError("De 11v11-middenlijn vereist precies vijf geldige beeldpunten.")

        center = np.median(values, axis=0)
        _u, _s, vh = np.linalg.svd(values - center, full_matrices=False)
        direction = vh[0]
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        offsets = (values - center) @ normal

        # Chalk, cones and mouse clicks have a real width. Reject only an obvious
        # accidental click and keep the normal placement tolerance in the average.
        scale = max(float(np.median(np.abs(offsets))) * 1.4826, 1.0)
        keep = np.abs(offsets) <= max(4.0 * scale, 6.0)
        if int(np.count_nonzero(keep)) >= 4 and not np.all(keep):
            center = np.mean(values[keep], axis=0)
            _u, _s, vh = np.linalg.svd(values[keep] - center, full_matrices=False)
            direction = vh[0]
            normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)

        equation = np.asarray((normal[0], normal[1], -float(normal @ center)))
        equation /= max(float(np.linalg.norm(equation[:2])), 1e-12)
        errors = np.abs(np.column_stack((values, np.ones(len(values)))) @ equation)
        return cls(
            video_name,
            int(frame_number),
            float(time_seconds),
            tuple(tuple(map(float, point)) for point in values),
            tuple(map(float, equation)),
            float(np.sqrt(np.mean(np.square(errors)))),
            float(np.max(errors)),
            rear_sideline_point,
            front_sideline_point,
            rear_sideline_frame_number,
            front_sideline_frame_number,
        )

    def endpoints(self, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        a, b, c = self.equation
        candidates: list[tuple[float, float]] = []
        if abs(b) > 1e-9:
            for x in (0.0, float(width - 1)):
                y = -(a * x + c) / b
                if 0.0 <= y < height:
                    candidates.append((x, y))
        if abs(a) > 1e-9:
            for y in (0.0, float(height - 1)):
                x = -(b * y + c) / a
                if 0.0 <= x < width:
                    candidates.append((x, y))
        unique = []
        for point in candidates:
            if not any(np.linalg.norm(np.asarray(point) - other) < 1.0 for other in unique):
                unique.append(np.asarray(point))
        if len(unique) < 2:
            raise ValueError("De middenlijn snijdt het zichtbare beeld niet op twee plaatsen.")
        return tuple(np.rint(unique[0]).astype(int)), tuple(np.rint(unique[1]).astype(int))

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "video_name": self.video_name,
            "frame_number": self.frame_number,
            "time_seconds": self.time_seconds,
            "semantic_line": "11v11_midfield_line",
            "direction_role": "parallel_to_8v8_sidelines",
            "points": [list(point) for point in self.points],
            "equation": list(self.equation),
            "rms_error_px": self.rms_error_px,
            "maximum_error_px": self.maximum_error_px,
            "rear_sideline_point": (
                None if self.rear_sideline_point is None else list(self.rear_sideline_point)
            ),
            "front_sideline_point": (
                None if self.front_sideline_point is None else list(self.front_sideline_point)
            ),
            "rear_sideline_frame_number": self.rear_sideline_frame_number,
            "front_sideline_frame_number": self.front_sideline_frame_number,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ManualMidfieldLine":
        return cls(
            str(data["video_name"]),
            int(data["frame_number"]),
            float(data["time_seconds"]),
            tuple(tuple(map(float, point)) for point in data["points"]),
            tuple(map(float, data["equation"])),
            float(data["rms_error_px"]),
            float(data["maximum_error_px"]),
            (
                None
                if data.get("rear_sideline_point") is None
                else tuple(map(float, data["rear_sideline_point"]))
            ),
            (
                None
                if data.get("front_sideline_point") is None
                else tuple(map(float, data["front_sideline_point"]))
            ),
            (
                None
                if data.get("rear_sideline_frame_number") is None
                else int(data["rear_sideline_frame_number"])
            ),
            (
                None
                if data.get("front_sideline_frame_number") is None
                else int(data["front_sideline_frame_number"])
            ),
        )


def save_manual_midfield_line(observation: ManualMidfieldLine, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(observation.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_manual_midfield_line(path: Path) -> ManualMidfieldLine:
    return ManualMidfieldLine.from_dict(json.loads(path.read_text(encoding="utf-8")))
