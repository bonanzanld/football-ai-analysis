from __future__ import annotations

import unittest

from football_ai.calibration.bootstrap.goal_seed import build_goal_sample_window


class GoalSeedFallbackTests(unittest.TestCase):
    def test_builds_bounded_samples_around_both_goal_views(self) -> None:
        samples = build_goal_sample_window((10.0, 90.0), fps=30.0, frame_count=3001)
        self.assertTrue(samples)
        self.assertEqual({item["view_position"] for item in samples}, {0.0, 1.0})
        self.assertGreaterEqual(min(item["time_seconds"] for item in samples), 0.0)
        self.assertLessEqual(max(item["time_seconds"] for item in samples), 100.0)
        self.assertEqual(len({item["frame_number"] for item in samples}), len(samples))


if __name__ == "__main__":
    unittest.main()
