import pytest

from football_ai.calibration.manual_parallel_lines import (
    ManualParallelLine,
    ManualParallelLineReference,
)


def _line(kind, frame, equation):
    return ManualParallelLine(kind, frame, frame / 30, ((0, 0),) * 5, equation, 0.1, 0.2)


def test_parallel_lines_from_same_frame_recover_vanishing_point():
    reference = ManualParallelLineReference(
        "match.mp4",
        (
            _line("midfield", 60, (0, 1, -10)),
            _line("goal_area_5m", 30, (1, 0, -20)),
            _line("penalty_area_16m", 30, (0, 1, -10)),
        ),
    )

    assert reference.vanishing_point_at_frame(30) == pytest.approx((20.0, 10.0))
