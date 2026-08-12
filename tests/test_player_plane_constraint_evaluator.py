from dataclasses import replace

from football_ai.tracking.player_plane_constraint_evaluator import (
    PlayerPlaneConstraintEvaluator,
)
from football_ai.tracking.projected_track_evaluator import ProjectedTrackEvaluation
from football_ai.tracking.track_evaluator import TrackEvaluation


def _track(track_id: int, *, usable: bool = True) -> TrackEvaluation:
    return TrackEvaluation(
        track_id=track_id,
        quality_score=90.0,
        stability_score=90.0,
        confidence_score=90.0,
        duration_score=90.0,
        visibility_score=90.0,
        classification="usable" if usable else "short",
        is_noise=False,
        is_short=not usable,
        is_usable=usable,
        is_stable=usable,
    )


def _projection(track_id: int) -> ProjectedTrackEvaluation:
    return ProjectedTrackEvaluation(
        track_id=track_id,
        projection_quality_score=90.0,
        coverage_score=90.0,
        inside_pitch_score=90.0,
        continuity_score=90.0,
        jump_score=90.0,
        maturity_score=90.0,
        projection_coverage=0.95,
        inside_pitch_ratio=0.96,
        acceptable_pitch_ratio=0.98,
        track_frames=100,
        projected_frames=95,
        average_step_meters=0.3,
        maximum_step_meters=1.2,
        valid_step_count=94,
        unrealistic_jump_count=0,
        extreme_jump_count=0,
        outside_position_count=4,
        tolerated_outside_count=2,
        severe_outside_count=0,
        average_outside_distance_meters=0.2,
        maximum_outside_distance_meters=0.7,
        classification="reliable_projection",
        is_projection_available=True,
        is_projection_usable=True,
        is_projection_reliable=True,
        rejection_reasons=(),
    )


def test_scores_consistent_player_ground_plane_as_reliable() -> None:
    evaluator = PlayerPlaneConstraintEvaluator()
    tracks = {track_id: _track(track_id) for track_id in (1, 2, 3)}
    projections = {track_id: _projection(track_id) for track_id in tracks}

    result = evaluator.evaluate(tracks, projections)

    assert result.classification == "reliable"
    assert result.has_sufficient_evidence
    assert result.score > 90.0


def test_rejects_projection_with_outside_points_and_large_jumps() -> None:
    evaluator = PlayerPlaneConstraintEvaluator()
    tracks = {track_id: _track(track_id) for track_id in (1, 2, 3)}
    bad = replace(
        _projection(1),
        inside_pitch_ratio=0.25,
        acceptable_pitch_ratio=0.35,
        unrealistic_jump_count=30,
        extreme_jump_count=15,
    )
    projections = {track_id: replace(bad, track_id=track_id) for track_id in tracks}

    result = evaluator.evaluate(tracks, projections)

    assert result.classification == "rejected"
    assert result.score < 60.0


def test_ignores_unusable_identity_tracks_and_requires_enough_evidence() -> None:
    evaluator = PlayerPlaneConstraintEvaluator()
    tracks = {1: _track(1), 2: _track(2, usable=False)}
    projections = {1: _projection(1), 2: _projection(2)}

    result = evaluator.evaluate(tracks, projections)

    assert result.usable_track_count == 1
    assert result.classification == "insufficient_evidence"
    assert not result.has_sufficient_evidence
