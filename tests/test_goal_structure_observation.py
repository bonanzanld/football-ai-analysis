from __future__ import annotations

import unittest

from football_ai.calibration.goal_structure_observation import GoalStructureLine, GoalStructureObservation


class GoalStructureObservationTests(unittest.TestCase):
    def test_robust_line_fit_tolerates_click_margin_and_one_outlier(self) -> None:
        line = GoalStructureLine.fit(
            "crossbar",
            ((10.0, 20.2), (30.0, 19.8), (50.0, 20.1), (70.0, 28.0), (90.0, 19.9), (110.0, 20.1)),
        )
        self.assertLess(abs(line.equation[0]), 0.02)
        self.assertGreater(line.maximum_error_px, 5.0)

    def test_requires_five_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimaal vijf"):
            GoalStructureLine.fit("far_post", ((1.0, 1.0),) * 4)

    def test_intersects_four_goal_corners(self) -> None:
        def line(name, equation):
            return GoalStructureLine(name, ((0.0, 0.0),) * 5, equation, 0.0, 0.0)
        goal = GoalStructureObservation(
            "A", 10, 1.0,
            (
                line("far_post", (1.0, 0.0, -10.0)),
                line("crossbar", (0.0, 1.0, -20.0)),
                line("near_post", (1.0, 0.0, -30.0)),
                line("goal_line", (0.0, 1.0, -50.0)),
            ),
        )
        self.assertEqual(goal.corners()["far_top"], (10.0, 20.0))
        self.assertEqual(goal.corners()["near_bottom"], (30.0, 50.0))


if __name__ == "__main__":
    unittest.main()
