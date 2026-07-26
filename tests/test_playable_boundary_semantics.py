import unittest

from football_ai.calibration.playable_boundary_semantics import (
    BoundaryEvidenceSource,
    PlayableBoundaryBinding,
    PlayableBoundaryRole,
)


class PlayableBoundarySemanticsTests(unittest.TestCase):
    def test_full_pitch_sideline_can_be_confirmed_as_eight_v_eight_end_line(self) -> None:
        binding = PlayableBoundaryBinding(
            PlayableBoundaryRole.END_LINE_A,
            BoundaryEvidenceSource.FULL_PITCH_SIDELINE,
            "11v11_sideline_near",
            True,
        )
        self.assertTrue(binding.confirmed)

    def test_other_white_marking_cannot_silently_become_playable_boundary(self) -> None:
        with self.assertRaises(ValueError):
            PlayableBoundaryBinding(
                PlayableBoundaryRole.NEAR_SIDELINE,
                BoundaryEvidenceSource.FULL_PITCH_OTHER_MARKING,
                "11v11_penalty_area",
                True,
            )

    def test_confirmed_goal_area_line_can_be_reused_when_field_layout_is_known(self) -> None:
        binding = PlayableBoundaryBinding(
            PlayableBoundaryRole.NEAR_SIDELINE,
            BoundaryEvidenceSource.FULL_PITCH_GOAL_AREA_LINE,
            "11v11_goal_area_5_5m",
            True,
        )
        self.assertTrue(binding.confirmed)


if __name__ == "__main__":
    unittest.main()
