from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Iterable


Point = tuple[float, float]


class GoalkeeperDecision(StrEnum):
    PLAYER = "player"
    REVIEW = "review"
    GOALKEEPER_CANDIDATE = "goalkeeper_candidate"


@dataclass(frozen=True, slots=True)
class GoalLineReference:
    goal_id: str
    first_post: Point
    second_post: Point

    @property
    def center(self) -> Point:
        return (
            (self.first_post[0] + self.second_post[0]) / 2.0,
            (self.first_post[1] + self.second_post[1]) / 2.0,
        )


@dataclass(frozen=True, slots=True)
class GoalkeeperEvidence:
    track_id: int
    team_id: int | None
    uniform_outlier_score: float
    goal_proximity_score: float
    defensive_depth_score: float
    track_stability_score: float
    movement_confinement_score: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "uniform_outlier_score",
            "goal_proximity_score",
            "defensive_depth_score",
            "track_stability_score",
            "movement_confinement_score",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} moet tussen 0 en 1 liggen.")


@dataclass(frozen=True, slots=True)
class GoalkeeperAssessment:
    track_id: int
    team_id: int | None
    score: float
    decision: GoalkeeperDecision
    evidence: GoalkeeperEvidence
    reasons: tuple[str, ...]


class GoalkeeperClassifier:
    """Ranks keeper candidates without overriding authoritative human review."""

    def assess(self, evidence: GoalkeeperEvidence) -> GoalkeeperAssessment:
        spatial_score = max(
            evidence.goal_proximity_score,
            evidence.defensive_depth_score,
        )
        score = (
            0.58 * evidence.uniform_outlier_score
            + 0.18 * evidence.track_stability_score
            + 0.14 * evidence.movement_confinement_score
            + 0.06 * evidence.goal_proximity_score
            + 0.04 * evidence.defensive_depth_score
        )
        reasons = []
        if evidence.uniform_outlier_score >= 0.55:
            reasons.append("tenue wijkt af van de veldspelers")
        if evidence.goal_proximity_score >= 0.55:
            reasons.append("track staat vaak bij de eigen doellijn")
        if evidence.defensive_depth_score >= 0.65:
            reasons.append("track is vaak de laatste speler van het team")
        if evidence.track_stability_score >= 0.60:
            reasons.append("track is lang genoeg gevolgd")
        if evidence.movement_confinement_score >= 0.65:
            reasons.append("track beweegt relatief weinig ten opzichte van het team")

        if (
            evidence.team_id in (0, 1)
            and evidence.uniform_outlier_score >= 0.65
            and evidence.track_stability_score >= 0.35
            and score >= 0.58
        ):
            decision = GoalkeeperDecision.GOALKEEPER_CANDIDATE
        elif (
            evidence.uniform_outlier_score >= 0.50
            or spatial_score >= 0.75
            or (
                evidence.movement_confinement_score >= 0.80
                and evidence.track_stability_score >= 0.60
            )
        ):
            decision = GoalkeeperDecision.REVIEW
        else:
            decision = GoalkeeperDecision.PLAYER
        return GoalkeeperAssessment(
            track_id=evidence.track_id,
            team_id=evidence.team_id,
            score=float(score),
            decision=decision,
            evidence=evidence,
            reasons=tuple(reasons),
        )


def goal_line_proximity_score(
    footpoint: Point,
    goal_line: GoalLineReference,
    maximum_distance_pixels: float,
) -> float:
    """Score distance to the finite line segment between both goalposts."""

    if maximum_distance_pixels <= 0.0:
        raise ValueError("maximum_distance_pixels moet groter dan nul zijn.")
    distance = _point_to_segment_distance(
        footpoint,
        goal_line.first_post,
        goal_line.second_post,
    )
    return max(0.0, 1.0 - distance / maximum_distance_pixels)


def is_on_pitch_side_of_goal_line(
    footpoint: Point,
    goal_line: GoalLineReference,
    pitch_reference_point: Point,
    *,
    tolerance_pixels: float = 3.0,
) -> bool:
    """Reject people behind a goal line before goalkeeper ranking."""
    if tolerance_pixels < 0.0:
        raise ValueError("tolerance_pixels must not be negative")
    ax, ay = goal_line.first_post
    bx, by = goal_line.second_post
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("Goal posts must not coincide")
    reference_side = (dx * (pitch_reference_point[1] - ay) - dy * (pitch_reference_point[0] - ax)) / length
    point_side = (dx * (footpoint[1] - ay) - dy * (footpoint[0] - ax)) / length
    if abs(reference_side) <= tolerance_pixels:
        raise ValueError("Pitch reference point lies on the goal line")
    return point_side * (1.0 if reference_side > 0.0 else -1.0) >= -tolerance_pixels


def defensive_depth_score(
    candidate_footpoint: Point,
    teammate_footpoints: Iterable[Point],
    own_goal_line: GoalLineReference,
) -> float:
    """Return 1 for the teammate nearest its own goal, decreasing by rank."""

    points = [candidate_footpoint, *teammate_footpoints]
    distances = [math.dist(point, own_goal_line.center) for point in points]
    candidate_distance = distances[0]
    rank = sum(distance < candidate_distance for distance in distances[1:])
    if len(points) == 1:
        return 1.0
    return max(0.0, 1.0 - rank / (len(points) - 1))


def _point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.dist(point, start)
    projection = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / length_squared
    projection = min(1.0, max(0.0, projection))
    closest = (start[0] + projection * dx, start[1] + projection * dy)
    return math.dist(point, closest)
