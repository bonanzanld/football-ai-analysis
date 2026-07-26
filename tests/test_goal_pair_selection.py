import unittest

from football_ai.calibration.bootstrap.goal_detection import GoalCandidate
from football_ai.calibration.bootstrap.goal_pair_selection import (
    CameraStateGoalEvidence,
    select_opposing_goal_pair,
)
from football_ai.calibration.bootstrap.temporal_goal_confirmation import ConfirmedGoal


def confirmed(center_x: float) -> ConfirmedGoal:
    candidate = GoalCandidate(
        left_ground=(center_x - 45.0, 140.0),
        right_ground=(center_x + 45.0, 140.0),
        left_top=(center_x - 45.0, 95.0),
        right_top=(center_x + 45.0, 95.0),
        confidence=0.95,
        crossbar_supported=True,
        backline_support=0.9,
    )
    return ConfirmedGoal(candidate, 10, 12, 10 / 12, 0.01, 0.92)


class GoalPairSelectionTests(unittest.TestCase):
    def test_accepts_only_opposite_camera_extremes(self) -> None:
        states = (
            CameraStateGoalEvidence(1, 0.02, 640, 360, (confirmed(240.0),)),
            CameraStateGoalEvidence(2, 0.50, 640, 360, (confirmed(320.0),)),
            CameraStateGoalEvidence(3, 0.98, 640, 360, (confirmed(400.0),)),
        )

        result = select_opposing_goal_pair(states)

        self.assertIsNotNone(result.pair)
        self.assertEqual({result.pair.first_state, result.pair.second_state}, {1, 3})

    def test_rejects_goal_only_in_central_camera_state(self) -> None:
        states = (
            CameraStateGoalEvidence(1, 0.03, 640, 360, ()),
            CameraStateGoalEvidence(2, 0.50, 640, 360, (confirmed(320.0),)),
            CameraStateGoalEvidence(3, 0.97, 640, 360, ()),
        )

        result = select_opposing_goal_pair(states)

        self.assertIsNone(result.pair)
        self.assertIn("beide camerauitersten", result.reason)


if __name__ == "__main__":
    unittest.main()
