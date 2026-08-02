from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
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
    def from_dict(cls, data: dict[str, Any]) -> "PitchProfile":
        converted = dict(data)
        converted["pitch_type"] = PitchType(converted["pitch_type"])
        return cls(**converted)


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