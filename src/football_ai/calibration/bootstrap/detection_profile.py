from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MatchFormat(str, Enum):
    SIX_V_SIX = "6v6"
    EIGHT_V_EIGHT = "8v8"
    ELEVEN_V_ELEVEN = "11v11"


@dataclass(frozen=True, slots=True)
class PitchDetectionProfile:
    match_format: MatchFormat
    name: str
    pitch_length_m: float
    pitch_width_m: float
    goal_width_m: float
    goal_height_m: float
    white_line_evidence_weight: float
    boundary_marker_evidence_weight: float
    goal_evidence_weight: float
    notes: tuple[str, ...]
    boundary_layout_tolerance_m: float = 0.0
    minimum_pitch_length_m: float | None = None
    maximum_pitch_length_m: float | None = None
    minimum_pitch_width_m: float | None = None
    maximum_pitch_width_m: float | None = None

    def __post_init__(self) -> None:
        minimum_length = self.minimum_pitch_length_m or self.pitch_length_m
        maximum_length = self.maximum_pitch_length_m or self.pitch_length_m
        minimum_width = self.minimum_pitch_width_m or self.pitch_width_m
        maximum_width = self.maximum_pitch_width_m or self.pitch_width_m
        object.__setattr__(self, "minimum_pitch_length_m", minimum_length)
        object.__setattr__(self, "maximum_pitch_length_m", maximum_length)
        object.__setattr__(self, "minimum_pitch_width_m", minimum_width)
        object.__setattr__(self, "maximum_pitch_width_m", maximum_width)
        if self.boundary_layout_tolerance_m < 0.0:
            raise ValueError("De praktische uitzetmarge mag niet negatief zijn")
        if not (
            0 < minimum_length
            <= self.pitch_length_m
            <= maximum_length
        ):
            raise ValueError("Nominale veldlengte moet binnen de toegestane bandbreedte liggen")
        if not (
            0 < minimum_width
            <= self.pitch_width_m
            <= maximum_width
        ):
            raise ValueError("Nominale veldbreedte moet binnen de toegestane bandbreedte liggen")

    @property
    def dimensions_are_exact(self) -> bool:
        return (
            self.minimum_pitch_length_m == self.maximum_pitch_length_m
            and self.minimum_pitch_width_m == self.maximum_pitch_width_m
        )

    def contains_dimensions(self, *, length_m: float, width_m: float) -> bool:
        return (
            self.minimum_pitch_length_m <= length_m <= self.maximum_pitch_length_m
            and self.minimum_pitch_width_m <= width_m <= self.maximum_pitch_width_m
        )

    @property
    def soft_pitch_dimension_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        margin = self.boundary_layout_tolerance_m
        return (
            (max(0.1, self.minimum_pitch_length_m - margin), self.maximum_pitch_length_m + margin),
            (max(0.1, self.minimum_pitch_width_m - margin), self.maximum_pitch_width_m + margin),
        )


def create_detection_profile(match_format: MatchFormat | str) -> PitchDetectionProfile:
    selected = MatchFormat(match_format)
    if selected is MatchFormat.SIX_V_SIX:
        return PitchDetectionProfile(
            selected,
            "Kwart veld 6v6",
            pitch_length_m=42.5,
            pitch_width_m=30.0,
            minimum_pitch_length_m=42.5,
            maximum_pitch_length_m=42.5,
            minimum_pitch_width_m=30.0,
            maximum_pitch_width_m=30.0,
            goal_width_m=3.0,
            goal_height_m=1.0,
            white_line_evidence_weight=0.25,
            boundary_marker_evidence_weight=1.0,
            goal_evidence_weight=0.8,
            notes=(
                "Veldgrenzen worden doorgaans vooral met hoedjes gemarkeerd.",
                "Geschilderde witte lijnen behoren tot het onderliggende 11v11-veld.",
                "Een witte 11v11-lijn telt alleen als 6v6-grens wanneer die relatie apart is bevestigd.",
            ),
            boundary_layout_tolerance_m=2.0,
        )
    if selected is MatchFormat.EIGHT_V_EIGHT:
        return PitchDetectionProfile(
            selected,
            "Half veld 8v8",
            pitch_length_m=64.0,
            pitch_width_m=42.5,
            minimum_pitch_length_m=60.0,
            maximum_pitch_length_m=70.0,
            minimum_pitch_width_m=42.5,
            maximum_pitch_width_m=55.0,
            goal_width_m=5.0,
            goal_height_m=2.0,
            white_line_evidence_weight=0.55,
            boundary_marker_evidence_weight=0.85,
            goal_evidence_weight=1.0,
            notes=(
                "Alle geschilderde witte lijnen behoren tot het onderliggende 11v11-veld.",
                "De twee 11v11-zijlijnen kunnen tevens als 8v8-achterlijnen worden gebruikt.",
                "Die gedeelde functie moet per speelveldopstelling expliciet worden bevestigd.",
                "De 8v8-doelen van 5 bij 2 meter zijn afzonderlijke speelveldankers.",
                "Lange grenzen kunnen met hoedjes zijn gemarkeerd.",
                "64 bij 42,5 meter is alleen een nominale werkhypothese; KNVB staat 60-70 bij 42,5-55 meter toe.",
            ),
            boundary_layout_tolerance_m=4.0,
        )
    return PitchDetectionProfile(
        selected,
        "Volledig veld 11v11",
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        minimum_pitch_length_m=100.0,
        maximum_pitch_length_m=105.0,
        minimum_pitch_width_m=64.0,
        maximum_pitch_width_m=69.0,
        goal_width_m=7.32,
        goal_height_m=2.44,
        white_line_evidence_weight=1.0,
        boundary_marker_evidence_weight=0.2,
        goal_evidence_weight=0.9,
        notes=(
            "Doorlopende witte veldmarkeringen zijn primair bewijs.",
            "105 bij 68 meter is de internationale referentie, niet de veronderstelde maat van ieder Nederlands veld.",
        ),
        boundary_layout_tolerance_m=0.5,
    )
