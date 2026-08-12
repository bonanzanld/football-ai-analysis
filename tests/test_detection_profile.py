import pytest

from football_ai.calibration.bootstrap.detection_profile import (
    MatchFormat,
    PitchDetectionProfile,
    create_detection_profile,
)


@pytest.mark.parametrize(
    ("match_format", "expected_bounds", "exact"),
    (
        (MatchFormat.SIX_V_SIX, (42.5, 42.5, 30.0, 30.0), True),
        (MatchFormat.EIGHT_V_EIGHT, (60.0, 70.0, 42.5, 55.0), False),
        (MatchFormat.ELEVEN_V_ELEVEN, (100.0, 105.0, 64.0, 69.0), False),
    ),
)
def test_knvb_pitch_dimension_bounds(match_format, expected_bounds, exact):
    profile = create_detection_profile(match_format)

    assert (
        profile.minimum_pitch_length_m,
        profile.maximum_pitch_length_m,
        profile.minimum_pitch_width_m,
        profile.maximum_pitch_width_m,
    ) == expected_bounds
    assert profile.dimensions_are_exact is exact


def test_eight_v_eight_dimension_membership_uses_knvb_bounds():
    profile = create_detection_profile(MatchFormat.EIGHT_V_EIGHT)

    assert profile.contains_dimensions(length_m=60.0, width_m=42.5)
    assert profile.contains_dimensions(length_m=70.0, width_m=55.0)
    assert not profile.contains_dimensions(length_m=59.9, width_m=42.5)
    assert not profile.contains_dimensions(length_m=64.0, width_m=55.1)
    assert profile.soft_pitch_dimension_bounds == ((56.0, 74.0), (38.5, 59.0))
    assert profile.boundary_layout_tolerance_m == 4.0


def test_profile_rejects_nominal_dimensions_outside_bounds():
    with pytest.raises(ValueError, match="Nominale veldlengte"):
        PitchDetectionProfile(
            match_format=MatchFormat.EIGHT_V_EIGHT,
            name="ongeldig",
            pitch_length_m=71.0,
            pitch_width_m=42.5,
            minimum_pitch_length_m=60.0,
            maximum_pitch_length_m=70.0,
            minimum_pitch_width_m=42.5,
            maximum_pitch_width_m=55.0,
            goal_width_m=5.0,
            goal_height_m=2.0,
            white_line_evidence_weight=1.0,
            boundary_marker_evidence_weight=1.0,
            goal_evidence_weight=1.0,
            notes=(),
        )
