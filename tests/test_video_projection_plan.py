import unittest

from football_ai.calibration.video_projection_plan import PlannedProjection, VideoProjectionPlan


class VideoProjectionPlanTests(unittest.TestCase):
    def test_round_trip_and_ratios(self) -> None:
        matrix = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        plan = VideoProjectionPlan(
            "match.mp4",
            "8v8",
            10.0,
            1.0,
            0.5,
            (
                PlannedProjection(10.0, 300, "valid", "goal-a", matrix, "ok", supporting_line_count=2, supporting_line_length_m=12.0),
                PlannedProjection(10.5, 315, "candidate", "goal-b", matrix, "check"),
                PlannedProjection(11.0, 330, "unknown", None, None, "missing"),
            ),
        )
        restored = VideoProjectionPlan.from_dict(plan.to_dict())
        self.assertEqual(restored, plan)
        self.assertAlmostEqual(restored.resolved_ratio, 2.0 / 3.0)
        self.assertAlmostEqual(restored.trusted_ratio, 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
