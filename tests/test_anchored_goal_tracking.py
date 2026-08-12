import numpy as np

from football_ai.calibration.anchored_goal_tracking import contiguous_goal_windows, project_anchored_goal
from football_ai.calibration.global_frame_graph import FrameGraphEdge


def _edge(matrix):
    return FrameGraphEdge("a", "b", np.asarray(matrix, dtype=float), 100, 80, 0.8, 0.2, 0.2, 1.0)


def test_accepts_consistent_full_and_ground_motion():
    matrix = ((1, 0, 5), (0, 1, 3), (0, 0, 1))
    result = project_anchored_goal(((10, 20), (30, 20)), ((10, 5), (30, 5)), _edge(matrix), _edge(matrix))
    assert result.valid
    assert result.ground_points == ((15.0, 23.0), (35.0, 23.0))


def test_rejects_parallax_disagreement_at_goal_feet():
    full = ((1, 0, 30), (0, 1, 0), (0, 0, 1))
    ground = ((1, 0, 5), (0, 1, 0), (0, 0, 1))
    result = project_anchored_goal(((10, 20), (30, 20)), ((10, 5), (30, 5)), _edge(full), _edge(ground))
    assert not result.valid
    assert result.model_disagreement_px == 25.0


def test_groups_goal_samples_into_local_windows():
    assert contiguous_goal_windows((10.0, 10.5, 11.0, 20.0, 20.5), maximum_gap_seconds=0.75) == (
        (10.0, 11.0, 3),
        (20.0, 20.5, 2),
    )
