import unittest

from football_ai.calibration.video_projection_plan import (
    PlannedProjection,
    VideoProjectionPlan,
    gate_projection_plan_with_player_evidence,
)


class VideoProjectionPlanTests(unittest.TestCase):
    @staticmethod
    def _matrix():
        return ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

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

    def test_player_evidence_can_reject_but_never_promote_projection(self) -> None:
        matrix = self._matrix()
        plan = VideoProjectionPlan(
            "match.mp4", "8v8", 0.0, 1.0, 0.5,
            (
                PlannedProjection(0.0, 0, "valid", "a", matrix, "lines"),
                PlannedProjection(0.5, 15, "candidate", "b", matrix, "candidate"),
                PlannedProjection(1.0, 30, "unknown", None, None, "missing"),
            ),
        )

        gated = gate_projection_plan_with_player_evidence(
            plan, {0: "rejected", 15: "supportive", 30: "supportive"}
        )

        self.assertEqual(gated.records[0].status, "unknown")
        self.assertIsNone(gated.records[0].projection_matrix)
        self.assertEqual(gated.records[1], plan.records[1])
        self.assertEqual(gated.records[2], plan.records[2])


if __name__ == "__main__":
    unittest.main()
