from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from football_ai.calibration.quality_report import CalibrationQualityReport

from .field_model import PitchProfile


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
        )
