import unittest

from football_ai.calibration.goal_zone_markings import (
    create_goal_zone_reference,
    match_goal_zone_depth_lines,
)


class GoalZoneMarkingTests(unittest.TestCase):
    def test_reference_contains_goal_area_penalty_area_and_arc(self) -> None:
        reference = create_goal_zone_reference("A")
        self.assertEqual([item.offset_from_goal_line_m for item in reference.depth_lines], [0.0, 5.5, 16.5])
        self.assertEqual(reference.penalty_arc.center_from_goal_line_m, 11.0)
        self.assertEqual(reference.penalty_arc.radius_m, 9.15)

    def test_known_goal_side_resolves_goal_and_penalty_area_lines(self) -> None:
        result = match_goal_zone_depth_lines((20.0, 36.5), create_goal_zone_reference("A"))
        self.assertTrue(result.resolved)
        self.assertEqual(result.marking_ids, ("goal_line", "penalty_area"))

    def test_known_goal_side_resolves_goal_area_and_penalty_area_lines(self) -> None:
        result = match_goal_zone_depth_lines((67.32, 76.19), create_goal_zone_reference("B"))
        self.assertTrue(result.resolved)
        self.assertEqual(result.marking_ids, ("penalty_area", "goal_area"))
        self.assertAlmostEqual(result.scale, 8.87 / 11.0, places=2)

    def test_single_line_remains_insufficient(self) -> None:
        result = match_goal_zone_depth_lines((27.4,), create_goal_zone_reference("A"))
        self.assertFalse(result.resolved)

    def test_unknown_full_pitch_side_can_match_pattern_without_claiming_side(self) -> None:
        result = match_goal_zone_depth_lines((67.32, 76.19), create_goal_zone_reference("unknown"))
        self.assertTrue(result.resolved)
        self.assertEqual(result.goal_side.value, "unknown")
        self.assertIn("zijde", result.reason)


if __name__ == "__main__":
    unittest.main()
