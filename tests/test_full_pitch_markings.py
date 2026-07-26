import unittest

from football_ai.calibration.full_pitch_markings import (
    circle_center_matches_halfway_line,
    create_standard_full_pitch_marking_model,
    match_marking_offsets,
)
from football_ai.calibration.ground_line_evidence import GroundLineFamily


class FullPitchMarkingTests(unittest.TestCase):
    def test_contains_standard_metric_markings(self) -> None:
        model = create_standard_full_pitch_marking_model()
        self.assertEqual(model.pitch_length_m, 105.0)
        self.assertEqual(model.pitch_width_m, 68.0)
        self.assertEqual(model.center_circle_radius_m, 9.15)
        self.assertEqual(model.penalty_area_depth_m, 16.5)
        self.assertEqual(model.goal_area_depth_m, 5.5)

    def test_two_lines_remain_an_explicitly_ambiguous_hypothesis(self) -> None:
        result = match_marking_offsets(
            (10.0, 26.5),
            GroundLineFamily.TRANSVERSE,
            create_standard_full_pitch_marking_model(),
        )
        self.assertFalse(result.resolved)
        self.assertGreater(len(result.hypotheses), 0)
        self.assertIn("derde lijn", result.reason)

    def test_three_metric_lines_can_resolve_the_identity(self) -> None:
        result = match_marking_offsets(
            (3.0, 19.5, 55.5),
            GroundLineFamily.TRANSVERSE,
            create_standard_full_pitch_marking_model(),
        )
        self.assertTrue(result.resolved)
        self.assertIn(
            result.hypotheses[0].marking_ids,
            (
                ("goal_line_a", "penalty_area_a", "halfway"),
                ("goal_line_b", "penalty_area_b", "halfway"),
            ),
        )
        self.assertTrue(circle_center_matches_halfway_line(55.5, result, create_standard_full_pitch_marking_model()))
        self.assertFalse(circle_center_matches_halfway_line(88.0, result, create_standard_full_pitch_marking_model()))


if __name__ == "__main__":
    unittest.main()
