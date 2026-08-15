import unittest

from tools.qa_moving_local_field_atlas import (
    _minimum_white_support,
    _support_band_thickness,
    _temporally_joint_evidence,
)


class MovingLocalFieldAtlasQATests(unittest.TestCase):
    def test_white_support_scales_with_resolution(self):
        self.assertEqual(_support_band_thickness(720), 6)
        self.assertEqual(_support_band_thickness(2160), 18)
        self.assertAlmostEqual(_minimum_white_support(720), 0.06)
        self.assertAlmostEqual(_minimum_white_support(2160), 0.08)

    def test_temporal_evidence_requires_both_recent_sources(self):
        self.assertTrue(_temporally_joint_evidence(12.0, 11.0, 9.0))
        self.assertFalse(_temporally_joint_evidence(12.1, 11.0, 9.0))
        self.assertFalse(_temporally_joint_evidence(12.0, None, 11.0))


if __name__ == "__main__":
    unittest.main()
