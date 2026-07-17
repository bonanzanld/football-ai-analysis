from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class PitchType(str, Enum):
    QUARTER = "quarter"
    HALF = "half"
    FULL = "full"
    CUSTOM = "custom"


@dataclass(frozen=True)
class PitchProfile:
    """
    Beschrijving van het werkelijke speelveld in meters.

    Coordinate system:
    - x: breedterichting
    - y: lengterichting

    Hoekpunten:
    - (0, 0)
    - (width_m, 0)
    - (width_m, length_m)
    - (0, length_m)
    """

    name: str
    pitch_type: PitchType
    length_m: float
    width_m: float
    goal_width_m: float
    goal_height_m: float

    length_tolerance_m: float = 2.0
    width_tolerance_m: float = 2.0

    back_lines_reliable: bool = False
    side_lines_reliable: bool = False
    goals_reliable: bool = True
    cones_expected: bool = False

    def __post_init__(self) -> None:
        if self.length_m <= 0:
            raise ValueError("length_m moet groter zijn dan 0.")

        if self.width_m <= 0:
            raise ValueError("width_m moet groter zijn dan 0.")

        if self.goal_width_m <= 0:
            raise ValueError("goal_width_m moet groter zijn dan 0.")

        if self.goal_height_m <= 0:
            raise ValueError("goal_height_m moet groter zijn dan 0.")

    @property
    def world_corners(self) -> np.ndarray:
        """
        Veldhoeken in vaste volgorde:

        0: links op achterlijn A
        1: rechts op achterlijn A
        2: rechts op achterlijn B
        3: links op achterlijn B
        """
        return np.array(
            [
                [0.0, 0.0],
                [self.width_m, 0.0],
                [self.width_m, self.length_m],
                [0.0, self.length_m],
            ],
            dtype=np.float32,
        )

    @property
    def goal_a_posts(self) -> np.ndarray:
        left_x = (self.width_m - self.goal_width_m) / 2.0
        right_x = left_x + self.goal_width_m

        return np.array(
            [
                [left_x, 0.0],
                [right_x, 0.0],
            ],
            dtype=np.float32,
        )

    @property
    def goal_b_posts(self) -> np.ndarray:
        left_x = (self.width_m - self.goal_width_m) / 2.0
        right_x = left_x + self.goal_width_m

        return np.array(
            [
                [left_x, self.length_m],
                [right_x, self.length_m],
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pitch_type"] = self.pitch_type.value
        return data

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> PitchProfile:
        converted = dict(data)
        converted["pitch_type"] = PitchType(
            converted["pitch_type"]
        )

        return cls(**converted)


@dataclass
class PitchCalibration:
    profile: PitchProfile
    image_corners: np.ndarray
    image_to_pitch_matrix: np.ndarray
    pitch_to_image_matrix: np.ndarray
    source_video: str
    source_frame_number: int
    source_time_seconds: float
    frame_width: int
    frame_height: int

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

    def to_dict(self) -> dict[str, Any]:
        return {
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
    ) -> PitchCalibration:
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
        )


def create_half_pitch_profile() -> PitchProfile:
    return PitchProfile(
        name="Half veld 8-tegen-8",
        pitch_type=PitchType.HALF,
        length_m=64.0,
        width_m=42.5,
        goal_width_m=5.0,
        goal_height_m=2.0,
        length_tolerance_m=2.5,
        width_tolerance_m=2.5,
        back_lines_reliable=True,
        side_lines_reliable=False,
        goals_reliable=True,
        cones_expected=True,
    )


def create_full_pitch_profile() -> PitchProfile:
    return PitchProfile(
        name="Volledig voetbalveld",
        pitch_type=PitchType.FULL,
        length_m=105.0,
        width_m=68.0,
        goal_width_m=7.32,
        goal_height_m=2.44,
        length_tolerance_m=5.0,
        width_tolerance_m=4.0,
        back_lines_reliable=True,
        side_lines_reliable=True,
        goals_reliable=True,
        cones_expected=False,
    )


def create_quarter_pitch_profile(
    length_m: float = 42.5,
    width_m: float = 32.0,
    goal_width_m: float = 5.0,
    goal_height_m: float = 2.0,
) -> PitchProfile:
    """
    Een kwartveld kan per vereniging of spelvorm verschillen.

    Daarom zijn alle afmetingen hier bewust aanpasbaar.
    """
    return PitchProfile(
        name="Kwart veld",
        pitch_type=PitchType.QUARTER,
        length_m=length_m,
        width_m=width_m,
        goal_width_m=goal_width_m,
        goal_height_m=goal_height_m,
        length_tolerance_m=3.0,
        width_tolerance_m=3.0,
        back_lines_reliable=False,
        side_lines_reliable=False,
        goals_reliable=True,
        cones_expected=True,
    )