import pytest

from football_ai.calibration.zoom_segment_intrinsics import (
    ZoomSegmentIntrinsics,
    focal_from_orthogonal_vanishing_points,
    horizontal_fov,
    select_widest_zoom_segment,
)


def test_recovers_focal_from_orthogonal_vanishing_points() -> None:
    assert focal_from_orthogonal_vanishing_points((-360.0, 360.0), (1640.0, 360.0), (640.0, 360.0)) == pytest.approx(1000.0)


def test_rejects_nonphysical_vanishing_pair() -> None:
    with pytest.raises(ValueError):
        focal_from_orthogonal_vanishing_points((700.0, 360.0), (800.0, 360.0), (640.0, 360.0))


def test_horizontal_fov_is_finite() -> None:
    assert horizontal_fov(640.0, 1280) == pytest.approx(90.0)


def test_selects_smallest_focal_length_as_most_zoomed_out() -> None:
    zoomed_out = ZoomSegmentIntrinsics(10, 20, 15, 250, (640, 360), 137, "manual")
    zoomed_in = ZoomSegmentIntrinsics(30, 40, 35, 900, (640, 360), 71, "manual")

    assert select_widest_zoom_segment((zoomed_in, zoomed_out)) == zoomed_out


def test_rejects_empty_segment_selection() -> None:
    with pytest.raises(ValueError):
        select_widest_zoom_segment(())
