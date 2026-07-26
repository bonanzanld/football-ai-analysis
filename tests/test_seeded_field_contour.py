import unittest

from football_ai.calibration.bootstrap.goal_seed import GoalSeed
from football_ai.calibration.bootstrap.seeded_field_contour import build_seeded_field_contour
from football_ai.calibration.bootstrap.detection_profile import create_detection_profile


class SeededFieldContourTests(unittest.TestCase):
    def test_connects_rear_and_front_corners_semantically(self) -> None:
        seeds = (
            GoalSeed("A", 100, 3.0, 1, 0.0, (20.0, 20.0), (30.0, 20.0), 5.0, 0.8, (0.0, 20.0), (100.0, 20.0)),
            GoalSeed("B", 200, 6.0, 2, 1.0, (20.0, 30.0), (30.0, 30.0), 5.0, 0.8, (0.0, 30.0), (100.0, 30.0)),
        )

        contour = build_seeded_field_contour(seeds, create_detection_profile("8v8"))
        data = contour.to_dict()

        self.assertEqual(
            [corner["name"] for corner in data["corners"]],
            ["linksachter", "linksvoor", "rechtsachter", "rechtsvoor"],
        )
        self.assertIn(
            {"name": "zijlijn_achter", "from": "linksachter", "to": "rechtsachter"},
            data["boundaries"],
        )
        self.assertIn(
            {"name": "zijlijn_voor", "from": "linksvoor", "to": "rechtsvoor"},
            data["boundaries"],
        )

    def test_uses_selected_match_profile_dimensions(self) -> None:
        seeds = (
            GoalSeed("A", 100, 3.0, 1, 0.0, (20.0, 20.0), (30.0, 20.0), 3.0, 0.8, (0.0, 20.0)),
            GoalSeed("B", 200, 6.0, 2, 1.0, (20.0, 30.0), (30.0, 30.0), 3.0, 0.8, (0.0, 30.0)),
        )

        contour = build_seeded_field_contour(seeds, create_detection_profile("6v6"))

        self.assertEqual(contour.match_format, "6v6")
        self.assertEqual(contour.pitch_length_m, 42.5)
        self.assertEqual(contour.pitch_width_m, 30.0)


if __name__ == "__main__":
    unittest.main()
