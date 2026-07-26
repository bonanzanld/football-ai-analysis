from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

import numpy as np


class FullPitchGoalSide(str, Enum):
    UNKNOWN = "unknown"
    A = "A"
    B = "B"


@dataclass(frozen=True, slots=True)
class GoalZoneDepthLine:
    marking_id: str
    label: str
    offset_from_goal_line_m: float
    nominal_length_m: float


@dataclass(frozen=True, slots=True)
class PenaltyArcReference:
    center_from_goal_line_m: float
    radius_m: float


@dataclass(frozen=True, slots=True)
class GoalZoneReference:
    goal_side: FullPitchGoalSide
    pitch_length_m: float
    depth_lines: tuple[GoalZoneDepthLine, ...]
    penalty_arc: PenaltyArcReference


@dataclass(frozen=True, slots=True)
class GoalZoneMatch:
    goal_side: FullPitchGoalSide
    resolved: bool
    marking_ids: tuple[str, ...]
    detected_offsets_m: tuple[float, ...]
    scale: float | None
    translation_m: float | None
    rms_m: float | None
    score_margin: float | None
    reason: str

    def to_dict(self) -> dict:
        return {
            "goal_side": self.goal_side.value,
            "resolved": self.resolved,
            "marking_ids": list(self.marking_ids),
            "detected_offsets_m": list(self.detected_offsets_m),
            "scale": self.scale,
            "translation_m": self.translation_m,
            "rms_m": self.rms_m,
            "score_margin": self.score_margin,
            "reason": self.reason,
        }


def create_goal_zone_reference(
    goal_side: FullPitchGoalSide | str,
    pitch_length_m: float = 105.0,
) -> GoalZoneReference:
    side = FullPitchGoalSide(goal_side)
    return GoalZoneReference(
        side,
        pitch_length_m,
        (
            GoalZoneDepthLine("goal_line", "Achterlijn", 0.0, 68.0),
            GoalZoneDepthLine("goal_area", "5,5-meterlijn", 5.5, 18.32),
            GoalZoneDepthLine("penalty_area", "16,5-meterlijn", 16.5, 40.32),
        ),
        PenaltyArcReference(center_from_goal_line_m=11.0, radius_m=9.15),
    )


def match_goal_zone_depth_lines(
    detected_offsets_m: tuple[float, ...],
    reference: GoalZoneReference,
    scale_range: tuple[float, float] = (0.75, 1.25),
) -> GoalZoneMatch:
    detected = tuple(sorted(float(item) for item in detected_offsets_m))
    if len(detected) < 2:
        return GoalZoneMatch(
            reference.goal_side,
            False,
            (),
            detected,
            None,
            None,
            None,
            None,
            "Minimaal twee dieptelijnen in dezelfde doelstand vereist.",
        )
    hypotheses = []
    for detected_subset in combinations(detected, min(len(detected), 3)):
        for markings in combinations(reference.depth_lines, len(detected_subset)):
            base_offsets = np.asarray([item.offset_from_goal_line_m for item in markings], dtype=np.float64)
            if reference.goal_side is FullPitchGoalSide.B:
                model_offsets = reference.pitch_length_m - base_offsets
                ordered_markings = markings
            else:
                model_offsets = base_offsets
                ordered_markings = markings
            order = np.argsort(model_offsets)
            model_offsets = model_offsets[order]
            ordered_markings = tuple(ordered_markings[index] for index in order)
            design = np.column_stack((model_offsets, np.ones(len(model_offsets))))
            scale, translation = np.linalg.lstsq(design, np.asarray(detected_subset), rcond=None)[0]
            if not scale_range[0] <= float(scale) <= scale_range[1]:
                continue
            predicted = design @ np.asarray((scale, translation))
            rms = float(np.sqrt(np.mean(np.square(predicted - detected_subset))))
            scale_penalty = 0.5 * abs(float(scale) - 1.0)
            hypotheses.append(
                (
                    rms + scale_penalty,
                    rms,
                    float(scale),
                    float(translation),
                    tuple(item.marking_id for item in ordered_markings),
                    tuple(detected_subset),
                )
            )
    hypotheses.sort(key=lambda item: item[0])
    if not hypotheses:
        return GoalZoneMatch(
            reference.goal_side,
            False,
            (),
            detected,
            None,
            None,
            None,
            None,
            "Geen maatvaste combinatie voor achterlijn, 5,5-meterlijn en 16,5-meterlijn.",
        )
    best = hypotheses[0]
    distinct_competitor = next((item for item in hypotheses[1:] if item[4] != best[4]), None)
    margin = float("inf") if distinct_competitor is None else distinct_competitor[0] - best[0]
    resolved = best[1] <= 1.25 and margin >= 0.08
    reason = (
        (
            "Uniek 11v11-doelzonepatroon; de zijde van het grote veld is nog onbekend."
            if reference.goal_side is FullPitchGoalSide.UNKNOWN
            else "Unieke doelzonecombinatie voor de bevestigde 11v11-doelzijde."
        )
        if resolved
        else "Meerdere doelzonecombinaties passen nog vrijwel even goed."
    )
    return GoalZoneMatch(
        reference.goal_side,
        resolved,
        best[4],
        best[5],
        best[2],
        best[3],
        best[1],
        margin,
        reason,
    )
