from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from football_ai.calibration.ground_line_evidence import GroundLineFamily


@dataclass(frozen=True, slots=True)
class PitchMarkingLine:
    marking_id: str
    label: str
    family: GroundLineFamily
    offset_m: float
    nominal_length_m: float


@dataclass(frozen=True, slots=True)
class FullPitchMarkingModel:
    pitch_length_m: float
    pitch_width_m: float
    center_circle_radius_m: float
    penalty_area_depth_m: float
    penalty_area_width_m: float
    goal_area_depth_m: float
    goal_area_width_m: float
    lines: tuple[PitchMarkingLine, ...]

    def lines_for(self, family: GroundLineFamily) -> tuple[PitchMarkingLine, ...]:
        return tuple(item for item in self.lines if item.family is family)


@dataclass(frozen=True, slots=True)
class MarkingMatchHypothesis:
    family: GroundLineFamily
    detected_offsets_m: tuple[float, ...]
    marking_ids: tuple[str, ...]
    scale: float
    translation_m: float
    rms_m: float
    score: float

    def to_dict(self) -> dict:
        return {
            "family": self.family.value,
            "detected_offsets_m": list(self.detected_offsets_m),
            "marking_ids": list(self.marking_ids),
            "scale": self.scale,
            "translation_m": self.translation_m,
            "rms_m": self.rms_m,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class MarkingMatchResult:
    family: GroundLineFamily
    resolved: bool
    reason: str
    hypotheses: tuple[MarkingMatchHypothesis, ...]

    def to_dict(self) -> dict:
        return {
            "family": self.family.value,
            "resolved": self.resolved,
            "reason": self.reason,
            "hypotheses": [item.to_dict() for item in self.hypotheses],
        }


def circle_center_matches_halfway_line(
    center_length_coordinate_m: float,
    match: MarkingMatchResult,
    model: FullPitchMarkingModel,
    tolerance_m: float = 3.0,
) -> bool:
    return marking_coordinate_matches(
        center_length_coordinate_m,
        match,
        model,
        "halfway",
        tolerance_m=tolerance_m,
    )


def marking_coordinate_matches(
    coordinate_m: float,
    match: MarkingMatchResult,
    model: FullPitchMarkingModel,
    marking_id: str,
    tolerance_m: float = 3.0,
) -> bool:
    if tolerance_m <= 0.0 or not match.resolved or not match.hypotheses:
        return False
    best = match.hypotheses[0]
    if marking_id not in best.marking_ids:
        return False
    marking = next((item for item in model.lines if item.marking_id == marking_id), None)
    if marking is None:
        return False
    predicted = best.scale * marking.offset_m + best.translation_m
    return abs(float(coordinate_m) - predicted) <= tolerance_m


def create_standard_full_pitch_marking_model(
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> FullPitchMarkingModel:
    """Return the metric straight-line model for a conventional 11v11 pitch."""
    penalty_depth, penalty_width = 16.5, 40.32
    goal_depth, goal_width = 5.5, 18.32
    penalty_near = (pitch_width_m - penalty_width) / 2.0
    penalty_far = pitch_width_m - penalty_near
    goal_near = (pitch_width_m - goal_width) / 2.0
    goal_far = pitch_width_m - goal_near
    lines = (
        PitchMarkingLine("sideline_near", "Nabije zijlijn", GroundLineFamily.LONGITUDINAL, 0.0, pitch_length_m),
        PitchMarkingLine("penalty_side_near", "Rand strafschopgebied nabij", GroundLineFamily.LONGITUDINAL, penalty_near, penalty_depth),
        PitchMarkingLine("goal_area_side_near", "Rand doelgebied nabij", GroundLineFamily.LONGITUDINAL, goal_near, goal_depth),
        PitchMarkingLine("goal_area_side_far", "Rand doelgebied ver", GroundLineFamily.LONGITUDINAL, goal_far, goal_depth),
        PitchMarkingLine("penalty_side_far", "Rand strafschopgebied ver", GroundLineFamily.LONGITUDINAL, penalty_far, penalty_depth),
        PitchMarkingLine("sideline_far", "Verre zijlijn", GroundLineFamily.LONGITUDINAL, pitch_width_m, pitch_length_m),
        PitchMarkingLine("goal_line_a", "Achterlijn A", GroundLineFamily.TRANSVERSE, 0.0, pitch_width_m),
        PitchMarkingLine("goal_area_a", "Doelgebiedslijn A", GroundLineFamily.TRANSVERSE, goal_depth, goal_width),
        PitchMarkingLine("penalty_area_a", "Strafschopgebiedslijn A", GroundLineFamily.TRANSVERSE, penalty_depth, penalty_width),
        PitchMarkingLine("halfway", "Middenlijn", GroundLineFamily.TRANSVERSE, pitch_length_m / 2.0, pitch_width_m),
        PitchMarkingLine("penalty_area_b", "Strafschopgebiedslijn B", GroundLineFamily.TRANSVERSE, pitch_length_m - penalty_depth, penalty_width),
        PitchMarkingLine("goal_area_b", "Doelgebiedslijn B", GroundLineFamily.TRANSVERSE, pitch_length_m - goal_depth, goal_width),
        PitchMarkingLine("goal_line_b", "Achterlijn B", GroundLineFamily.TRANSVERSE, pitch_length_m, pitch_width_m),
    )
    return FullPitchMarkingModel(
        pitch_length_m,
        pitch_width_m,
        9.15,
        penalty_depth,
        penalty_width,
        goal_depth,
        goal_width,
        lines,
    )


def match_marking_offsets(
    detected_offsets_m: tuple[float, ...],
    family: GroundLineFamily,
    model: FullPitchMarkingModel,
    scale_range: tuple[float, float] = (0.8, 1.2),
    maximum_hypotheses: int = 5,
) -> MarkingMatchResult:
    detected = tuple(sorted(float(item) for item in detected_offsets_m))
    if len(detected) < 2:
        return MarkingMatchResult(family, False, "Minimaal twee lijnclusters vereist.", ())
    candidates = model.lines_for(family)
    hypotheses: list[MarkingMatchHypothesis] = []
    for selected in combinations(candidates, len(detected)):
        for ordered in (selected, tuple(reversed(selected))):
            model_offsets = np.asarray([item.offset_m for item in ordered], dtype=np.float64)
            design = np.column_stack((model_offsets, np.ones(len(model_offsets))))
            scale, translation = np.linalg.lstsq(design, np.asarray(detected), rcond=None)[0]
            if not scale_range[0] <= abs(float(scale)) <= scale_range[1]:
                continue
            predicted = design @ np.asarray((scale, translation))
            rms = float(np.sqrt(np.mean(np.square(predicted - detected))))
            scale_penalty = 0.35 * abs(abs(float(scale)) - 1.0)
            hypotheses.append(
                MarkingMatchHypothesis(
                    family,
                    detected,
                    tuple(item.marking_id for item in ordered),
                    float(scale),
                    float(translation),
                    rms,
                    rms + scale_penalty,
                )
            )
    hypotheses.sort(key=lambda item: item.score)
    unique: list[MarkingMatchHypothesis] = []
    for item in hypotheses:
        if item.marking_ids not in {other.marking_ids for other in unique}:
            unique.append(item)
        if len(unique) >= maximum_hypotheses:
            break
    if not unique:
        return MarkingMatchResult(family, False, "Geen maatvaste 11v11-combinatie gevonden.", ())
    if len(detected) < 3:
        return MarkingMatchResult(
            family,
            False,
            "Twee lijnen leveren meerdere mogelijke 11v11-identiteiten; een derde lijn is nodig.",
            tuple(unique),
        )
    competing = next(
        (
            item
            for item in unique[1:]
            if not _equivalent_under_pitch_reflection(unique[0].marking_ids, item.marking_ids)
        ),
        None,
    )
    margin = competing.score - unique[0].score if competing is not None else float("inf")
    resolved = unique[0].rms_m <= 1.25 and margin >= 0.35
    reason = (
        "Unieke 11v11-markeringenmatch, eventueel gespiegeld tussen veldkant A en B."
        if resolved
        else "Beste 11v11-match is nog niet duidelijk genoeg onderscheiden van alternatieven."
    )
    return MarkingMatchResult(family, resolved, reason, tuple(unique))


def _equivalent_under_pitch_reflection(
    first: tuple[str, ...],
    second: tuple[str, ...],
) -> bool:
    mirror = {
        "sideline_near": "sideline_far",
        "sideline_far": "sideline_near",
        "penalty_side_near": "penalty_side_far",
        "penalty_side_far": "penalty_side_near",
        "goal_area_side_near": "goal_area_side_far",
        "goal_area_side_far": "goal_area_side_near",
        "goal_line_a": "goal_line_b",
        "goal_line_b": "goal_line_a",
        "goal_area_a": "goal_area_b",
        "goal_area_b": "goal_area_a",
        "penalty_area_a": "penalty_area_b",
        "penalty_area_b": "penalty_area_a",
        "halfway": "halfway",
    }
    return tuple(mirror[item] for item in first) == second
