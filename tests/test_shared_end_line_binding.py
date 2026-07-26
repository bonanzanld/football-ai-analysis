import unittest

from football_ai.calibration.bootstrap.goal_seed import GoalSeed
from football_ai.calibration.bootstrap.white_line_detection import WhiteLineCandidate
from football_ai.calibration.shared_end_line_binding import assess_shared_full_pitch_sideline


class SharedEndLineBindingTests(unittest.TestCase):
    @staticmethod
    def _seed(width: float = 5.0) -> GoalSeed:
        return GoalSeed("A", 10, 1.0, 1, 0.0, (100.0, 200.0), (160.0, 206.0), width, 0.8)

    @staticmethod
    def _line(start=(20.0, 192.0), end=(300.0, 220.0)) -> WhiteLineCandidate:
        return WhiteLineCandidate(start, end, 5.7, 281.0, 0.9, 0.8, 0.85, 0.5)

    def test_confirms_full_pitch_sideline_through_both_goal_posts(self) -> None:
        result = assess_shared_full_pitch_sideline(self._seed(), (self._line(),))
        self.assertTrue(result.binding.confirmed)
        self.assertEqual(result.binding.source.value, "full_pitch_sideline")

    def test_rejects_white_line_away_from_goal_posts(self) -> None:
        result = assess_shared_full_pitch_sideline(
            self._seed(),
            (self._line((20.0, 100.0), (300.0, 128.0)),),
        )
        self.assertFalse(result.binding.confirmed)

    def test_rejects_non_eight_v_eight_goal_width(self) -> None:
        result = assess_shared_full_pitch_sideline(self._seed(7.32), (self._line(),))
        self.assertFalse(result.binding.confirmed)

    def test_operator_can_explicitly_confirm_known_shared_field_layout(self) -> None:
        result = assess_shared_full_pitch_sideline(
            self._seed(),
            (),
            operator_confirmed_layout=True,
        )
        self.assertTrue(result.binding.confirmed)
        self.assertFalse(result.visual_confirmation)
        self.assertEqual(result.confirmation_origin, "operator_configuration")


if __name__ == "__main__":
    unittest.main()
