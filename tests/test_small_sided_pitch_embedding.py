import pytest

from football_ai.calibration.small_sided_pitch_embedding import (
    FullPitchHalf,
    PerpendicularHalfPitchEmbedding,
    embed_full_pitch_markings,
)
from football_ai.calibration.full_pitch_markings import create_standard_full_pitch_marking_model
from football_ai.calibration.ground_line_evidence import GroundLineFamily


def test_centers_rotated_8v8_pitch_on_standard_full_pitch_half() -> None:
    embedding = PerpendicularHalfPitchEmbedding(
        105.0, 68.0, 64.0, 42.5, FullPitchHalf.GOAL_A
    )

    assert embedding.full_sideline_margin_m == 2.0
    assert embedding.half_end_margin_m == 5.0
    assert embedding.full_length_bounds_m == (5.0, 47.5)
    assert embedding.full_width_bounds_m == (2.0, 66.0)
    assert embedding.corners_on_full_pitch == (
        (5.0, 2.0),
        (5.0, 66.0),
        (47.5, 66.0),
        (47.5, 2.0),
    )
    assert embedding.full_to_small((52.5, 34.0)) == (32.0, 47.5)


def test_mirrors_embedding_on_other_half() -> None:
    embedding = PerpendicularHalfPitchEmbedding(
        105.0, 68.0, 64.0, 42.5, FullPitchHalf.GOAL_B
    )

    assert embedding.full_length_bounds_m == (57.5, 100.0)
    assert embedding.small_to_full((0.0, 0.0)) == (100.0, 2.0)
    assert embedding.small_to_full((64.0, 42.5)) == (57.5, 66.0)


def test_rejects_small_pitch_that_does_not_fit() -> None:
    with pytest.raises(ValueError):
        PerpendicularHalfPitchEmbedding(
            90.0, 60.0, 64.0, 42.5, FullPitchHalf.GOAL_A
        )


def test_rotates_and_offsets_full_pitch_markings_into_small_pitch_axes() -> None:
    embedding = PerpendicularHalfPitchEmbedding(
        105.0, 68.0, 64.0, 42.5, FullPitchHalf.GOAL_A
    )
    transformed = embed_full_pitch_markings(
        create_standard_full_pitch_marking_model(), embedding
    )

    by_id = {line.marking_id: line for line in transformed.lines}
    assert by_id["sideline_near"].family is GroundLineFamily.TRANSVERSE
    assert by_id["sideline_near"].offset_m == -2.0
    assert by_id["penalty_area_a"].family is GroundLineFamily.LONGITUDINAL
    assert by_id["penalty_area_a"].offset_m == 11.5
    assert by_id["halfway"].family is GroundLineFamily.LONGITUDINAL
    assert by_id["halfway"].offset_m == 47.5
