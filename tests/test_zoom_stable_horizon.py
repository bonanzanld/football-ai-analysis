from football_ai.calibration.manual_perspective_reference import (
    ManualPerspectiveView,
    ManualReferenceLine,
    PerspectiveDirection,
)
from football_ai.calibration.zoom_stable_horizon import (
    ZoomStableSegment,
    select_zoom_stable_horizons,
)


def _line(direction, start, end):
    middle = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    return ManualReferenceLine(direction, (start, middle, end))


def _complete_view(time_seconds: float) -> ManualPerspectiveView:
    return ManualPerspectiveView(
        "right_goal",
        int(time_seconds * 30),
        time_seconds,
        (
            _line(PerspectiveDirection.BETWEEN_GOALS, (0, 0), (20, 20)),
            _line(PerspectiveDirection.BETWEEN_GOALS, (0, 10), (20, 20)),
            _line(PerspectiveDirection.ALONG_END_LINES, (0, 20), (20, 0)),
            _line(PerspectiveDirection.ALONG_END_LINES, (10, 20), (20, 0)),
        ),
    )


def test_selects_complete_horizon_inside_zoom_stable_segment():
    result = select_zoom_stable_horizons(
        (_complete_view(920.0),),
        (ZoomStableSegment(903.0, 928.5, 52),),
    )

    assert len(result) == 1
    assert result[0].segment_start_seconds == 903.0


def test_rejects_horizon_near_zoom_transition():
    result = select_zoom_stable_horizons(
        (_complete_view(928.0),),
        (ZoomStableSegment(903.0, 928.5, 52),),
        boundary_margin_seconds=1.0,
    )

    assert result == ()
