from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PlayableBoundaryRole(str, Enum):
    END_LINE_A = "end_line_a"
    END_LINE_B = "end_line_b"
    FAR_SIDELINE = "far_sideline"
    NEAR_SIDELINE = "near_sideline"


class BoundaryEvidenceSource(str, Enum):
    FULL_PITCH_SIDELINE = "full_pitch_sideline"
    FULL_PITCH_GOAL_AREA_LINE = "full_pitch_goal_area_line"
    FULL_PITCH_OTHER_MARKING = "full_pitch_other_marking"
    CONES = "cones"
    MANUAL_CORNERS = "manual_corners"
    INFERRED = "inferred"


@dataclass(frozen=True, slots=True)
class PlayableBoundaryBinding:
    """Explicitly bind an 8v8 boundary to evidence on the underlying 11v11 surface."""

    role: PlayableBoundaryRole
    source: BoundaryEvidenceSource
    source_id: str | None = None
    confirmed: bool = False

    def __post_init__(self) -> None:
        if self.source is BoundaryEvidenceSource.FULL_PITCH_OTHER_MARKING and self.confirmed:
            raise ValueError(
                "Een willekeurige 11v11-markering mag niet als bevestigde 8v8-grens worden gebruikt."
            )

    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "source": self.source.value,
            "source_id": self.source_id,
            "confirmed": self.confirmed,
        }
