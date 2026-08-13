import pytest

from football_ai.classification.goalkeeper_classifier import GoalLineReference
from tools.qa_goal_window_people import _goal_relative_position


def test_goal_relative_position_is_normalized_by_moving_goal_line():
    line = GoalLineReference("A", (100, 200), (200, 200))

    assert _goal_relative_position((150, 220), line) == pytest.approx((.5, .2))
