from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from football_ai.calibration.bootstrap.goal_seed import GoalSeed
from football_ai.calibration.bootstrap.white_line_detection import WhiteLineCandidate
from football_ai.calibration.playable_boundary_semantics import (
    BoundaryEvidenceSource,
    PlayableBoundaryBinding,
    PlayableBoundaryRole,
)


@dataclass(frozen=True, slots=True)
class SharedEndLineAssessment:
    goal_id: str
    goal_width_m: float
    maximum_post_distance_px: float | None
    matched_line: WhiteLineCandidate | None
    binding: PlayableBoundaryBinding
    confirmation_origin: str
    visual_confirmation: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "goal_width_m": self.goal_width_m,
            "maximum_post_distance_px": self.maximum_post_distance_px,
            "matched_line": self.matched_line.to_dict() if self.matched_line is not None else None,
            "binding": self.binding.to_dict(),
            "confirmation_origin": self.confirmation_origin,
            "visual_confirmation": self.visual_confirmation,
            "reason": self.reason,
        }


def assess_shared_full_pitch_sideline(
    seed: GoalSeed,
    white_lines: tuple[WhiteLineCandidate, ...],
    expected_goal_width_m: float = 5.0,
    maximum_distance_px: float = 12.0,
    operator_confirmed_layout: bool = False,
) -> SharedEndLineAssessment:
    role = PlayableBoundaryRole.END_LINE_A if seed.goal_id == "A" else PlayableBoundaryRole.END_LINE_B
    if abs(seed.goal_width_m - expected_goal_width_m) > 0.25:
        binding = PlayableBoundaryBinding(role, BoundaryEvidenceSource.INFERRED, confirmed=False)
        return SharedEndLineAssessment(
            seed.goal_id,
            seed.goal_width_m,
            None,
            None,
            binding,
            "none",
            False,
            "Doelbreedte komt niet overeen met het 8v8-doel van 5 meter.",
        )
    ranked = []
    for candidate in white_lines:
        distances = (
            _point_line_distance(seed.first_ground, candidate),
            _point_line_distance(seed.second_ground, candidate),
        )
        maximum = max(distances)
        direction = np.asarray(candidate.end) - np.asarray(candidate.start)
        goal_direction = np.asarray(seed.second_ground) - np.asarray(seed.first_ground)
        denominator = float(np.linalg.norm(direction) * np.linalg.norm(goal_direction))
        if denominator < 1e-9:
            continue
        angle = float(
            np.degrees(
                np.arccos(np.clip(abs(float(direction @ goal_direction)) / denominator, 0.0, 1.0))
            )
        )
        if angle <= 8.0:
            ranked.append((maximum + 0.5 * angle, maximum, candidate))
    ranked.sort(key=lambda item: item[0])
    if not ranked or ranked[0][1] > maximum_distance_px:
        if operator_confirmed_layout:
            binding = PlayableBoundaryBinding(
                role,
                BoundaryEvidenceSource.FULL_PITCH_SIDELINE,
                f"11v11_sideline_at_8v8_goal_{seed.goal_id.lower()}",
                True,
            )
            reason = (
                "Veldopstelling is door de operator bevestigd; automatische witte-lijnsteun is onvoldoende."
            )
            origin = "operator_configuration"
        else:
            binding = PlayableBoundaryBinding(role, BoundaryEvidenceSource.INFERRED, confirmed=False)
            reason = "Geen betrouwbare witte 11v11-zijlijn door beide 8v8-doelpalen gevonden."
            origin = "none"
        distance = None if not ranked else ranked[0][1]
        return SharedEndLineAssessment(
            seed.goal_id,
            seed.goal_width_m,
            distance,
            None,
            binding,
            origin,
            False,
            reason,
        )
    _score, distance, candidate = ranked[0]
    binding = PlayableBoundaryBinding(
        role,
        BoundaryEvidenceSource.FULL_PITCH_SIDELINE,
        f"11v11_sideline_at_8v8_goal_{seed.goal_id.lower()}",
        True,
    )
    return SharedEndLineAssessment(
        seed.goal_id,
        seed.goal_width_m,
        distance,
        candidate,
        binding,
        "automatic_visual_evidence",
        True,
        "Witte 11v11-zijlijn loopt door beide grondpunten van het 8v8-doel.",
    )


def _point_line_distance(point: tuple[float, float], candidate: WhiteLineCandidate) -> float:
    start, end = np.asarray(candidate.start), np.asarray(candidate.end)
    direction = end - start
    offset = np.asarray(point) - start
    cross_product = direction[0] * offset[1] - direction[1] * offset[0]
    return abs(float(cross_product)) / max(float(np.linalg.norm(direction)), 1e-9)
