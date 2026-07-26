import unittest

from football_ai.calibration.bootstrap.goal_detection import GoalCandidate, GoalDetection
from football_ai.calibration.bootstrap.temporal_goal_confirmation import confirm_goals_temporally


def goal(center_x: float, confidence: float = 0.9) -> GoalCandidate:
    return GoalCandidate(
        left_ground=(center_x - 50.0, 150.0),
        right_ground=(center_x + 50.0, 150.0),
        left_top=(center_x - 50.0, 100.0),
        right_top=(center_x + 50.0, 100.0),
        confidence=confidence,
        crossbar_supported=True,
        backline_support=0.8,
    )


class TemporalGoalConfirmationTests(unittest.TestCase):
    def test_confirms_recurring_goal_and_rejects_single_frame_candidate(self) -> None:
        detections = [
            GoalDetection((goal(300.0), goal(520.0))),
            GoalDetection((goal(302.0),)),
            GoalDetection((goal(298.0),)),
            GoalDetection((goal(301.0),)),
        ]

        confirmed = confirm_goals_temporally(
            detections,
            [(640, 360)] * len(detections),
        )

        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].supporting_frame_count, 4)
        self.assertGreater(confirmed[0].confidence, 0.8)

    def test_requires_matching_input_lengths(self) -> None:
        with self.assertRaises(ValueError):
            confirm_goals_temporally([GoalDetection(())], [])


if __name__ == "__main__":
    unittest.main()
