from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from football_ai.calibration.camera_projection_3d import CameraProjection3D


@dataclass(frozen=True, slots=True)
class PlayerProjectionEvidence:
    footpoint_count: int
    exceeds_expected_player_count: bool
    projected_count: int
    inside_count: int
    tolerated_outside_count: int
    severe_outside_count: int
    inside_ratio: float
    acceptable_ratio: float
    classification: str


@dataclass(frozen=True, slots=True)
class PlayerProjectionSequenceEvidence:
    frame_count: int
    sufficient_frame_count: int
    projected_count: int
    acceptable_count: int
    acceptable_ratio: float
    classification: str


def evaluate_player_footpoints(
    projection: CameraProjection3D | None,
    footpoints: Iterable[tuple[float, float]],
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    tolerated_outside_m: float = 1.0,
    severe_outside_m: float = 5.0,
    minimum_acceptable_ratio: float = 0.60,
    minimum_players: int = 3,
    maximum_expected_players: int = 16,
) -> PlayerProjectionEvidence:
    """Measure whether detected player ground contacts fit a pitch projection.

    This is diagnostic evidence only. Players may legitimately stand just
    outside a touchline, so containment must never be treated as ground truth.
    """

    if pitch_length_m <= 0 or pitch_width_m <= 0:
        raise ValueError("Pitch dimensions must be positive")
    if tolerated_outside_m < 0 or severe_outside_m <= tolerated_outside_m:
        raise ValueError("Outside-distance thresholds are invalid")
    if not 0.5 <= minimum_acceptable_ratio <= 1.0 or minimum_players < 1:
        raise ValueError("Player-containment thresholds are invalid")
    if maximum_expected_players < minimum_players:
        raise ValueError("Maximum expected players must not be below the minimum")
    points = tuple((float(x), float(y)) for x, y in footpoints)
    if projection is None or not points:
        return PlayerProjectionEvidence(
            footpoint_count=len(points),
            exceeds_expected_player_count=len(points) > maximum_expected_players,
            projected_count=0,
            inside_count=0,
            tolerated_outside_count=0,
            severe_outside_count=0,
            inside_ratio=0.0,
            acceptable_ratio=0.0,
            classification="unavailable",
        )

    inside = tolerated = severe = projected = 0
    for point in points:
        try:
            x, y = projection.image_to_ground(point)
        except (ValueError, ArithmeticError):
            continue
        projected += 1
        dx = max(0.0, -x, x - pitch_length_m)
        dy = max(0.0, -y, y - pitch_width_m)
        outside = (dx * dx + dy * dy) ** 0.5
        if outside == 0.0:
            inside += 1
        elif outside <= tolerated_outside_m:
            tolerated += 1
        elif outside > severe_outside_m:
            severe += 1

    inside_ratio = inside / max(1, projected)
    acceptable_ratio = (inside + tolerated) / max(1, projected)
    if projected == 0:
        classification = "unavailable"
    elif projected < minimum_players:
        classification = "insufficient_evidence"
    elif severe / projected > (1.0 - minimum_acceptable_ratio) or acceptable_ratio < minimum_acceptable_ratio:
        classification = "rejected"
    elif acceptable_ratio >= 0.85:
        classification = "supportive"
    else:
        classification = "ambiguous"
    return PlayerProjectionEvidence(
        footpoint_count=len(points),
        exceeds_expected_player_count=len(points) > maximum_expected_players,
        projected_count=projected,
        inside_count=inside,
        tolerated_outside_count=tolerated,
        severe_outside_count=severe,
        inside_ratio=float(inside_ratio),
        acceptable_ratio=float(acceptable_ratio),
        classification=classification,
    )


def aggregate_player_projection_evidence(
    frames: Iterable[PlayerProjectionEvidence],
    *,
    minimum_acceptable_ratio: float = 0.60,
    minimum_frames: int = 3,
    minimum_total_players: int = 15,
) -> PlayerProjectionSequenceEvidence:
    """Judge field containment across nearby frames, not one noisy detection frame."""
    if not 0.5 <= minimum_acceptable_ratio <= 1.0:
        raise ValueError("Minimum acceptable ratio must be between 0.5 and 1.0")
    if minimum_frames < 1 or minimum_total_players < 1:
        raise ValueError("Sequence evidence minima must be positive")
    samples = tuple(frames)
    sufficient = tuple(
        item for item in samples
        if item.classification not in ("unavailable", "insufficient_evidence")
    )
    projected = sum(item.projected_count for item in sufficient)
    acceptable = sum(item.inside_count + item.tolerated_outside_count for item in sufficient)
    ratio = acceptable / max(1, projected)
    if len(sufficient) < minimum_frames or projected < minimum_total_players:
        classification = "insufficient_evidence"
    elif ratio < minimum_acceptable_ratio:
        classification = "rejected"
    elif ratio >= 0.85:
        classification = "supportive"
    else:
        classification = "ambiguous"
    return PlayerProjectionSequenceEvidence(
        len(samples), len(sufficient), projected, acceptable, float(ratio), classification
    )
