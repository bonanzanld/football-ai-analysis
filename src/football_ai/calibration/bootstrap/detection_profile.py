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


def create_detection_profile(match_format: MatchFormat | str) -> PitchDetectionProfile:
    selected = MatchFormat(match_format)
    if selected is MatchFormat.SIX_V_SIX:
        return PitchDetectionProfile(
            selected,
            "Kwart veld 6v6",
            pitch_length_m=42.5,
            pitch_width_m=30.0,
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
        )
    if selected is MatchFormat.EIGHT_V_EIGHT:
        return PitchDetectionProfile(
            selected,
            "Half veld 8v8",
            pitch_length_m=64.0,
            pitch_width_m=42.5,
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
            ),
        )
    return PitchDetectionProfile(
        selected,
        "Volledig veld 11v11",
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        goal_width_m=7.32,
        goal_height_m=2.44,
        white_line_evidence_weight=1.0,
        boundary_marker_evidence_weight=0.2,
        goal_evidence_weight=0.9,
        notes=("Doorlopende witte veldmarkeringen zijn primair bewijs.",),
    )
