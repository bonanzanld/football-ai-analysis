from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class RegisteredGroundFrame:
    frame_number: int
    time_seconds: float
    ground_to_image: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.ground_to_image, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("Geregistreerd grondframe vereist een eindige 3x3-homography.")
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            raise ValueError("Geregistreerde grondhomography is niet omkeerbaar.")
        object.__setattr__(self, "ground_to_image", matrix / matrix[2, 2])

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "time_seconds": self.time_seconds,
            "ground_to_image": self.ground_to_image.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RegisteredGroundFrame":
        return cls(
            int(data["frame_number"]),
            float(data["time_seconds"]),
            np.asarray(data["ground_to_image"], dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class GlobalGroundRegistration:
    video_name: str
    match_format: str
    reference_anchor_id: str
    frames: tuple[RegisteredGroundFrame, ...]
    connected_ratio: float
    line_observations: int
    solved_for_playable_field: bool
    reason: str

    def nearest(self, time_seconds: float) -> RegisteredGroundFrame:
        if not self.frames:
            raise ValueError("Globale grondregistratie bevat geen frames.")
        return min(self.frames, key=lambda item: abs(item.time_seconds - time_seconds))

    def ground_to_image_at(self, time_seconds: float) -> np.ndarray:
        """Interpolate the globally registered ground plane between keyframes."""
        ordered = tuple(sorted(self.frames, key=lambda item: item.time_seconds))
        if not ordered:
            raise ValueError("Globale grondregistratie bevat geen frames.")
        if time_seconds <= ordered[0].time_seconds:
            return ordered[0].ground_to_image.copy()
        if time_seconds >= ordered[-1].time_seconds:
            return ordered[-1].ground_to_image.copy()
        for first, second in zip(ordered, ordered[1:]):
            if first.time_seconds <= time_seconds <= second.time_seconds:
                span = second.time_seconds - first.time_seconds
                alpha = (time_seconds - first.time_seconds) / max(span, 1e-9)
                first_h = first.ground_to_image / first.ground_to_image[2, 2]
                second_h = second.ground_to_image / second.ground_to_image[2, 2]
                matrix = (1.0 - alpha) * first_h + alpha * second_h
                if abs(float(matrix[2, 2])) < 1e-9 or abs(float(np.linalg.det(matrix))) < 1e-12:
                    return first_h.copy() if alpha < 0.5 else second_h.copy()
                return matrix / matrix[2, 2]
        return ordered[-1].ground_to_image.copy()

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "video_name": self.video_name,
            "match_format": self.match_format,
            "reference_anchor_id": self.reference_anchor_id,
            "connected_ratio": self.connected_ratio,
            "line_observations": self.line_observations,
            "solved_for_playable_field": self.solved_for_playable_field,
            "reason": self.reason,
            "frames": [item.to_dict() for item in self.frames],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalGroundRegistration":
        return cls(
            str(data["video_name"]),
            str(data["match_format"]),
            str(data["reference_anchor_id"]),
            tuple(RegisteredGroundFrame.from_dict(item) for item in data["frames"]),
            float(data["connected_ratio"]),
            int(data["line_observations"]),
            bool(data["solved_for_playable_field"]),
            str(data["reason"]),
        )


def save_global_ground_registration(registration: GlobalGroundRegistration, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registration.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_global_ground_registration(path: Path) -> GlobalGroundRegistration:
    return GlobalGroundRegistration.from_dict(json.loads(path.read_text(encoding="utf-8")))
