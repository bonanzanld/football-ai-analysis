from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from football_ai.calibration.quality_report import CalibrationQualityReport

from .field_model import PitchProfile


@dataclass(frozen=True)
class CalibrationKeyframe:
    frame_number: int
    time_seconds: float
    image_to_pitch_matrix: np.ndarray
    pitch_to_image_matrix: np.ndarray
    image_corners: np.ndarray
    point_count: int
    line_point_count: int
    line_rms_error_pixels: float | None
    geometry_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "image_to_pitch_matrix",
            np.asarray(self.image_to_pitch_matrix, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "pitch_to_image_matrix",
            np.asarray(self.pitch_to_image_matrix, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "image_corners",
            np.asarray(self.image_corners, dtype=np.float32),
        )

    @property
    def is_valid(self) -> bool:
        return not self.geometry_errors

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "time_seconds": self.time_seconds,
            "image_to_pitch_matrix": self.image_to_pitch_matrix.tolist(),
            "pitch_to_image_matrix": self.pitch_to_image_matrix.tolist(),
            "image_corners": self.image_corners.tolist(),
            "point_count": self.point_count,
            "line_point_count": self.line_point_count,
            "line_rms_error_pixels": self.line_rms_error_pixels,
            "geometry_errors": list(self.geometry_errors),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationKeyframe":
        return cls(
            frame_number=int(data["frame_number"]),
            time_seconds=float(data["time_seconds"]),
            image_to_pitch_matrix=np.asarray(
                data["image_to_pitch_matrix"], dtype=np.float64
            ),
            pitch_to_image_matrix=np.asarray(
                data["pitch_to_image_matrix"], dtype=np.float64
            ),
            image_corners=np.asarray(data["image_corners"], dtype=np.float32),
            point_count=int(data["point_count"]),
            line_point_count=int(data["line_point_count"]),
            line_rms_error_pixels=(
                float(data["line_rms_error_pixels"])
                if data.get("line_rms_error_pixels") is not None
                else None
            ),
            geometry_errors=tuple(data.get("geometry_errors", [])),
        )


@dataclass
class PitchCalibration:
    """
    Bevat alle informatie die nodig is om beeldcoördinaten
    om te zetten naar veldcoördinaten (en andersom).

    Dit is nadrukkelijk GEEN beschrijving van het speelveld,
    maar een opgeslagen kalibratieresultaat.
    """

    profile: PitchProfile

    image_corners: np.ndarray

    image_to_pitch_matrix: np.ndarray
    pitch_to_image_matrix: np.ndarray

    source_video: str
    source_frame_number: int
    source_time_seconds: float

    frame_width: int
    frame_height: int
    quality: CalibrationQualityReport | None = None
    keyframes: tuple[CalibrationKeyframe, ...] = ()

    def __post_init__(self) -> None:
        self.image_corners = np.asarray(
            self.image_corners,
            dtype=np.float32,
        )

        self.image_to_pitch_matrix = np.asarray(
            self.image_to_pitch_matrix,
            dtype=np.float64,
        )

        self.pitch_to_image_matrix = np.asarray(
            self.pitch_to_image_matrix,
            dtype=np.float64,
        )

        if self.image_corners.shape != (4, 2):
            raise ValueError(
                "image_corners moet exact vier xy-punten bevatten."
            )

        if self.image_to_pitch_matrix.shape != (3, 3):
            raise ValueError(
                "image_to_pitch_matrix moet een 3x3-matrix zijn."
            )

        if self.pitch_to_image_matrix.shape != (3, 3):
            raise ValueError(
                "pitch_to_image_matrix moet een 3x3-matrix zijn."
            )
        self.keyframes = tuple(
            sorted(self.keyframes, key=lambda item: item.frame_number)
        )

    def to_dict(self) -> dict:
        data = {
            "profile": self.profile.to_dict(),
            "image_corners": self.image_corners.tolist(),
            "image_to_pitch_matrix": (
                self.image_to_pitch_matrix.tolist()
            ),
            "pitch_to_image_matrix": (
                self.pitch_to_image_matrix.tolist()
            ),
            "source_video": self.source_video,
            "source_frame_number": self.source_frame_number,
            "source_time_seconds": self.source_time_seconds,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }
        if self.quality is not None:
            data["quality"] = self.quality.to_dict()
        if self.keyframes:
            data["keyframes"] = [
                keyframe.to_dict() for keyframe in self.keyframes
            ]
        return data

    def save(self, path: Path) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.to_dict(),
                file,
                indent=2,
                ensure_ascii=False,
            )

    @property
    def is_usable(self) -> bool:
        return self.quality is not None and self.quality.is_usable

    def require_usable(self) -> None:
        if self.quality is None or self.quality.assessment is None:
            raise RuntimeError(
                "Kalibratie heeft geen bruikbaarheidsbeoordeling. "
                "Voer de kalibratie opnieuw uit."
            )
        if self.keyframes and not all(
            keyframe.is_valid for keyframe in self.keyframes
        ):
            raise RuntimeError(
                "Kalibratiesequentie bevat geometrisch ongeldige keyframes."
            )
        if not self.quality.is_usable:
            assessment = self.quality.assessment
            reasons = "\n".join(
                f"- {reason}" for reason in assessment.failures
            )
            raise RuntimeError(
                "Kalibratie heeft status FAIL en mag niet worden gebruikt."
                + (f"\n{reasons}" if reasons else "")
            )

    def image_to_pitch_for_frame(self, frame_number: int) -> np.ndarray:
        if not self.keyframes:
            return self.image_to_pitch_matrix
        return min(
            self.keyframes,
            key=lambda item: abs(item.frame_number - frame_number),
        ).image_to_pitch_matrix

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> "PitchCalibration":
        if not path.exists():
            raise FileNotFoundError(
                f"Kalibratiebestand niet gevonden: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        return cls(
            profile=PitchProfile.from_dict(
                data["profile"]
            ),
            image_corners=np.array(
                data["image_corners"],
                dtype=np.float32,
            ),
            image_to_pitch_matrix=np.array(
                data["image_to_pitch_matrix"],
                dtype=np.float64,
            ),
            pitch_to_image_matrix=np.array(
                data["pitch_to_image_matrix"],
                dtype=np.float64,
            ),
            source_video=data["source_video"],
            source_frame_number=int(
                data["source_frame_number"]
            ),
            source_time_seconds=float(
                data["source_time_seconds"]
            ),
            frame_width=int(
                data["frame_width"]
            ),
            frame_height=int(
                data["frame_height"]
            ),
            quality=(
                CalibrationQualityReport.from_dict(data["quality"])
                if data.get("quality") is not None
                else None
            ),
            keyframes=tuple(
                CalibrationKeyframe.from_dict(item)
                for item in data.get("keyframes", [])
            ),
        )
