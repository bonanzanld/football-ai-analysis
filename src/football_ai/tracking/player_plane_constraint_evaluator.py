from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from football_ai.tracking.projected_track_evaluator import ProjectedTrackEvaluation
from football_ai.tracking.track_evaluator import TrackEvaluation


@dataclass(frozen=True)
class PlayerPlaneConstraintEvaluation:
    """Aggregate evidence that tracked player footpoints fit one ground plane."""

    score: float
    classification: str
    usable_track_count: int
    projected_frames: int
    valid_steps: int
    projection_coverage: float
    inside_pitch_ratio: float
    acceptable_pitch_ratio: float
    unrealistic_jump_ratio: float
    extreme_jump_ratio: float
    has_sufficient_evidence: bool


class PlayerPlaneConstraintEvaluator:
    """Turn moving player footpoints into a soft homography quality signal.

    Only technically usable identity tracks contribute. Players are not treated
    as stationary landmarks: the score uses field containment and physically
    plausible consecutive motion, never a fixed pitch coordinate.
    """

    def __init__(
        self,
        *,
        minimum_usable_tracks: int = 3,
        minimum_projected_frames: int = 90,
        minimum_valid_steps: int = 30,
    ) -> None:
        if minimum_usable_tracks < 1:
            raise ValueError("minimum_usable_tracks must be positive")
        if minimum_projected_frames < 1:
            raise ValueError("minimum_projected_frames must be positive")
        if minimum_valid_steps < 1:
            raise ValueError("minimum_valid_steps must be positive")
        self.minimum_usable_tracks = int(minimum_usable_tracks)
        self.minimum_projected_frames = int(minimum_projected_frames)
        self.minimum_valid_steps = int(minimum_valid_steps)

    def evaluate(
        self,
        track_evaluations: Mapping[int, TrackEvaluation],
        projected_evaluations: Mapping[int, ProjectedTrackEvaluation],
    ) -> PlayerPlaneConstraintEvaluation:
        selected = [
            projection
            for track_id, projection in projected_evaluations.items()
            if projection.is_projection_available
            and track_id in track_evaluations
            and track_evaluations[track_id].is_usable
        ]
        projected_frames = sum(item.projected_frames for item in selected)
        track_frames = sum(item.track_frames for item in selected)
        valid_steps = sum(item.valid_step_count for item in selected)

        projection_coverage = self._weighted_average(
            selected, "projection_coverage", "track_frames"
        )
        inside_pitch_ratio = self._weighted_average(
            selected, "inside_pitch_ratio", "projected_frames"
        )
        acceptable_pitch_ratio = self._weighted_average(
            selected, "acceptable_pitch_ratio", "projected_frames"
        )
        unrealistic_jumps = sum(item.unrealistic_jump_count for item in selected)
        extreme_jumps = sum(item.extreme_jump_count for item in selected)
        unrealistic_jump_ratio = unrealistic_jumps / max(1, valid_steps)
        extreme_jump_ratio = extreme_jumps / max(1, valid_steps)
        continuity = max(
            0.0,
            1.0 - unrealistic_jump_ratio - 2.0 * extreme_jump_ratio,
        )
        has_sufficient_evidence = (
            len(selected) >= self.minimum_usable_tracks
            and projected_frames >= self.minimum_projected_frames
            and valid_steps >= self.minimum_valid_steps
        )
        score = 100.0 * (
            0.20 * projection_coverage
            + 0.25 * inside_pitch_ratio
            + 0.30 * acceptable_pitch_ratio
            + 0.25 * continuity
        )
        if not selected:
            classification = "unavailable"
        elif not has_sufficient_evidence:
            classification = "insufficient_evidence"
        elif score >= 85.0 and extreme_jump_ratio <= 0.01:
            classification = "reliable"
        elif score >= 70.0 and extreme_jump_ratio <= 0.03:
            classification = "usable"
        else:
            classification = "rejected"
        return PlayerPlaneConstraintEvaluation(
            score=float(score),
            classification=classification,
            usable_track_count=len(selected),
            projected_frames=projected_frames,
            valid_steps=valid_steps,
            projection_coverage=projection_coverage,
            inside_pitch_ratio=inside_pitch_ratio,
            acceptable_pitch_ratio=acceptable_pitch_ratio,
            unrealistic_jump_ratio=float(unrealistic_jump_ratio),
            extreme_jump_ratio=float(extreme_jump_ratio),
            has_sufficient_evidence=has_sufficient_evidence,
        )

    @staticmethod
    def _weighted_average(
        evaluations: list[ProjectedTrackEvaluation],
        value_name: str,
        weight_name: str,
    ) -> float:
        weight = sum(float(getattr(item, weight_name)) for item in evaluations)
        if weight <= 0.0:
            return 0.0
        return float(
            sum(
                float(getattr(item, value_name)) * float(getattr(item, weight_name))
                for item in evaluations
            )
            / weight
        )
