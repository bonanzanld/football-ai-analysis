import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from football_ai.calibration.bootstrap.goal_detection import measure_backline_support
from football_ai.calibration.bootstrap.goal_seed import (
    GoalSeed,
    estimate_backline_endpoints,
    estimate_pitch_width_from_centered_goal,
    fit_average_support_line,
    load_goal_seeds,
    save_goal_seeds,
)


class GoalSeedTests(unittest.TestCase):
    def test_estimates_pitch_width_from_goal_and_both_corners(self) -> None:
        width = 49.0
        goal = 5.0

        def project(metres: float) -> tuple[float, float]:
            return ((3.0 * metres + 40.0) / (0.015 * metres + 1.0), 80.0)

        estimate = estimate_pitch_width_from_centered_goal(
            project(0.0),
            project((width - goal) / 2.0),
            project((width + goal) / 2.0),
            project(width),
            goal_width_m=goal,
            pitch_width_bounds_m=(42.5, 55.0),
        )

        self.assertAlmostEqual(estimate.pitch_width_m, width, places=4)
        self.assertLess(estimate.rms_error_px, 1e-6)

    def test_average_support_line_absorbs_click_margin(self) -> None:
        start, end, rms = fit_average_support_line(
            ((100.0, 202.0), (300.0, 198.0), (500.0, 204.0))
        )

        self.assertLess(rms, 4.0)
        self.assertLess(abs(start[1] - 200.0), 5.0)
        self.assertLess(abs(end[1] - 202.0), 5.0)

    def test_estimated_backline_is_capped_to_pitch_width(self) -> None:
        start, end = estimate_backline_endpoints(
            (100.0, 50.0),
            (150.0, 50.0),
            goal_width_m=5.0,
            pitch_width_m=42.5,
        )

        self.assertEqual(start, (-87.5, 50.0))
        self.assertEqual(end, (337.5, 50.0))

    def test_one_known_corner_corrects_projective_distance(self) -> None:
        def image_x(metres: float) -> float:
            return (2.0 * metres + 10.0) / (0.02 * metres + 1.0)

        rear_post_m = (42.5 - 5.0) / 2.0
        front_post_m = rear_post_m + 5.0
        start, end = estimate_backline_endpoints(
            (image_x(rear_post_m), 80.0),
            (image_x(front_post_m), 80.0),
            goal_width_m=5.0,
            pitch_width_m=42.5,
            rear_corner=(image_x(0.0), 80.0),
        )

        self.assertAlmostEqual(start[0], image_x(0.0), places=6)
        self.assertAlmostEqual(end[0], image_x(42.5), places=6)
        self.assertAlmostEqual(start[1], 80.0, places=6)
        self.assertAlmostEqual(end[1], 80.0, places=6)

    def test_two_known_corners_are_used_as_direct_anchors(self) -> None:
        start, end = estimate_backline_endpoints(
            (100.0, 80.0),
            (140.0, 82.0),
            goal_width_m=5.0,
            pitch_width_m=42.5,
            rear_corner=(20.0, 70.0),
            front_corner=(400.0, 110.0),
        )

        self.assertEqual(start, (20.0, 70.0))
        self.assertEqual(end, (400.0, 110.0))

    def test_white_backline_has_more_support_than_grass(self) -> None:
        frame = np.full((240, 400, 3), (45, 120, 45), np.uint8)
        cv2.line(frame, (20, 160), (380, 160), (250, 250, 250), 7)

        on_line = measure_backline_support(frame, (150.0, 160.0), (250.0, 160.0))
        off_line = measure_backline_support(frame, (150.0, 100.0), (250.0, 100.0))

        self.assertGreater(on_line, 0.7)
        self.assertLess(off_line, 0.1)

    def test_seed_json_preserves_semantic_goal_information(self) -> None:
        seeds = tuple(
            GoalSeed(
                goal,
                index * 100,
                index * 3.0,
                index + 1,
                float(index),
                (10.0, 20.0),
                (30.0, 20.0),
                5.0,
                0.8,
                rear_corner=(0.0, 20.0),
                rear_sideline_support=(5.0, 10.0),
                front_sideline_support=(35.0, 10.0),
                front_sideline_support_end=(55.0, 10.0),
            )
            for index, goal in enumerate(("A", "B"))
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seeds.json"
            save_goal_seeds(seeds, path)
            data = json.loads(path.read_text(encoding="utf-8"))
            loaded = load_goal_seeds(path)

        self.assertEqual([item["goal_id"] for item in data["goals"]], ["A", "B"])
        self.assertTrue(all(item["goal_width_m"] == 5.0 for item in data["goals"]))
        self.assertTrue(all(item["rear_corner"] == [0.0, 20.0] for item in data["goals"]))
        self.assertTrue(all(item["front_corner"] is None for item in data["goals"]))
        self.assertEqual(loaded, seeds)
        self.assertEqual(data["schema_version"], 5)
        self.assertIn("field_contour", data)
        self.assertTrue(all(item["rear_sideline_support"] == [5.0, 10.0] for item in data["goals"]))
        self.assertTrue(all(item["front_sideline_support"] == [35.0, 10.0] for item in data["goals"]))
        self.assertTrue(all(item["front_sideline_support_end"] == [55.0, 10.0] for item in data["goals"]))


if __name__ == "__main__":
    unittest.main()
