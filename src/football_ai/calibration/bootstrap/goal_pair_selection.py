from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np

from football_ai.calibration.bootstrap.temporal_goal_confirmation import ConfirmedGoal


@dataclass(frozen=True, slots=True)
class CameraStateGoalEvidence:
    camera_state: int
    view_position: float
    frame_width: int
    frame_height: int
    confirmed_goals: tuple[ConfirmedGoal, ...]


@dataclass(frozen=True, slots=True)
class OpposingGoalPair:
    first_state: int
    second_state: int
    first_goal: ConfirmedGoal
    second_goal: ConfirmedGoal
    confidence: float

    def to_dict(self) -> dict:
        return {
            "first_state": self.first_state,
            "second_state": self.second_state,
            "first_goal": self.first_goal.to_dict(),
            "second_goal": self.second_goal.to_dict(),
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class GoalPairSelection:
    pair: OpposingGoalPair | None
    reason: str
    evaluated_pair_count: int


def select_opposing_goal_pair(
    states: tuple[CameraStateGoalEvidence, ...],
    extreme_threshold: float = 0.24,
    minimum_confidence: float = 0.78,
) -> GoalPairSelection:
    """Selecteer alleen temporeel bevestigde doelen aan beide pan-uitersten."""
    first_side = [state for state in states if state.view_position <= extreme_threshold]
    second_side = [state for state in states if state.view_position >= 1.0 - extreme_threshold]
    if not first_side or not second_side:
        return GoalPairSelection(None, "Camerabereik bevat niet beide uitersten.", 0)
    scored: list[OpposingGoalPair] = []
    for first_state in first_side:
        for second_state in second_side:
            if first_state.camera_state == second_state.camera_state:
                continue
            for first_goal in first_state.confirmed_goals:
                for second_goal in second_state.confirmed_goals:
                    confidence = _pair_confidence(
                        first_state,
                        first_goal,
                        second_state,
                        second_goal,
                    )
                    scored.append(
                        OpposingGoalPair(
                            first_state.camera_state,
                            second_state.camera_state,
                            first_goal,
                            second_goal,
                            confidence,
                        )
                    )
    if not scored:
        return GoalPairSelection(
            None,
            "Geen temporeel bevestigde doelen aan beide camerauitersten.",
            0,
        )
    scored.sort(key=lambda item: item.confidence, reverse=True)
    best = scored[0]
    margin = best.confidence - (scored[1].confidence if len(scored) > 1 else 0.0)
    if best.confidence < minimum_confidence:
        return GoalPairSelection(
            None,
            f"Beste doelpaar heeft onvoldoende confidence ({best.confidence:.2f}).",
            len(scored),
        )
    if len(scored) > 1 and margin < 0.05:
        return GoalPairSelection(
            None,
            "Meerdere doelparen zijn vrijwel even waarschijnlijk.",
            len(scored),
        )
    return GoalPairSelection(best, "Betrouwbaar tegenoverliggend doelpaar gevonden.", len(scored))


def _pair_confidence(
    first_state: CameraStateGoalEvidence,
    first_goal: ConfirmedGoal,
    second_state: CameraStateGoalEvidence,
    second_goal: ConfirmedGoal,
) -> float:
    first_candidate = first_goal.representative
    second_candidate = second_goal.representative
    first_aspect = _goal_aspect(first_candidate)
    second_aspect = _goal_aspect(second_candidate)
    aspect_consistency = exp(-abs(log(max(first_aspect, 1e-6) / max(second_aspect, 1e-6))))
    first_width = abs(first_candidate.right_ground[0] - first_candidate.left_ground[0]) / first_state.frame_width
    second_width = abs(second_candidate.right_ground[0] - second_candidate.left_ground[0]) / second_state.frame_width
    width_consistency = exp(-abs(log(max(first_width, 1e-6) / max(second_width, 1e-6))) / 1.5)
    temporal = (first_goal.confidence + second_goal.confidence) / 2.0
    backline = (first_candidate.backline_support + second_candidate.backline_support) / 2.0
    return float(np.clip(0.48 * temporal + 0.22 * backline + 0.18 * aspect_consistency + 0.12 * width_consistency, 0.0, 1.0))


def _goal_aspect(candidate) -> float:
    width = abs(candidate.right_ground[0] - candidate.left_ground[0])
    height = (
        abs(candidate.left_ground[1] - candidate.left_top[1])
        + abs(candidate.right_ground[1] - candidate.right_top[1])
    ) / 2.0
    return width / max(height, 1e-6)
