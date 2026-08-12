import numpy as np

from football_ai.calibration.camera_projection_3d import CameraProjection3D
from football_ai.calibration.player_projection_evidence import (
    aggregate_player_projection_evidence,
    evaluate_player_footpoints,
)


def _identity_ground_projection() -> CameraProjection3D:
    return CameraProjection3D(
        np.asarray(
            ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        )
    )


def test_supports_projection_when_player_feet_fit_pitch() -> None:
    result = evaluate_player_footpoints(
        _identity_ground_projection(),
        ((10.0, 10.0), (20.0, 20.0), (63.5, 42.0)),
        pitch_length_m=64.0,
        pitch_width_m=42.5,
    )

    assert result.classification == "supportive"
    assert result.inside_ratio == 1.0


def test_rejects_projection_with_impossible_player_positions() -> None:
    result = evaluate_player_footpoints(
        _identity_ground_projection(),
        ((10.0, 10.0), (100.0, 80.0), (-30.0, 20.0), (20.0, 90.0)),
        pitch_length_m=64.0,
        pitch_width_m=42.5,
    )

    assert result.classification == "rejected"
    assert result.severe_outside_count == 3


def test_requires_multiple_players_before_supporting_projection() -> None:
    result = evaluate_player_footpoints(
        _identity_ground_projection(),
        ((10.0, 10.0),),
        pitch_length_m=64.0,
        pitch_width_m=42.5,
    )

    assert result.classification == "insufficient_evidence"


def test_accepts_field_line_geometry_at_sixty_percent_starting_threshold() -> None:
    result = evaluate_player_footpoints(
        _identity_ground_projection(),
        ((10.0, 10.0), (20.0, 20.0), (30.0, 30.0), (70.0, 10.0), (75.0, 20.0)),
        pitch_length_m=64.0,
        pitch_width_m=42.5,
        tolerated_outside_m=1.0,
    )

    assert result.acceptable_ratio == 0.6
    assert result.classification == "ambiguous"


def test_sequence_accepts_sixty_percent_but_can_be_tightened_later() -> None:
    frames = tuple(
        evaluate_player_footpoints(
            _identity_ground_projection(),
            points,
            pitch_length_m=64.0,
            pitch_width_m=42.5,
        )
        for points in (
            ((10, 10), (20, 20), (30, 30), (40, 20), (70, 10)),
            ((10, 10), (20, 20), (30, 30), (70, 10), (75, 20)),
            ((10, 10), (20, 20), (30, 30), (40, 20), (50, 10)),
        )
    )

    result = aggregate_player_projection_evidence(frames)

    assert result.acceptable_ratio == 0.80
    assert result.classification == "ambiguous"
    strict = aggregate_player_projection_evidence(
        frames, minimum_acceptable_ratio=0.80
    )
    assert strict.classification == "ambiguous"


def test_rejects_when_less_than_sixty_percent_fit() -> None:
    result = evaluate_player_footpoints(
        _identity_ground_projection(),
        ((10, 10), (20, 20), (70, 10), (75, 20), (80, 30)),
        pitch_length_m=64.0,
        pitch_width_m=42.5,
    )

    assert result.acceptable_ratio == 0.4
    assert result.classification == "rejected"


def test_flags_more_than_sixteen_detected_people_without_rejecting_geometry() -> None:
    result = evaluate_player_footpoints(
        _identity_ground_projection(),
        tuple((float(index % 8), float(index // 8)) for index in range(17)),
        pitch_length_m=64.0,
        pitch_width_m=42.5,
    )

    assert result.exceeds_expected_player_count is True
    assert result.classification == "supportive"
