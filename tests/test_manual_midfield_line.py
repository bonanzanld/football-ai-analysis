import unittest

import numpy as np

from football_ai.calibration.manual_midfield_line import ManualMidfieldLine


class ManualMidfieldLineTests(unittest.TestCase):
    def test_fits_five_imperfect_clicks(self) -> None:
        observation = ManualMidfieldLine.fit(
            "match.mp4",
            120,
            4.0,
            ((100.0, 200.0), (250.0, 203.0), (400.0, 205.0), (550.0, 209.0), (700.0, 211.0)),
        )

        self.assertLess(observation.rms_error_px, 1.0)
        self.assertLess(observation.maximum_error_px, 2.0)
        self.assertEqual(observation.frame_number, 120)

    def test_json_round_trip(self) -> None:
        observation = ManualMidfieldLine.fit(
            "match.mp4",
            300,
            10.0,
            ((50.0, 100.0), (100.0, 150.0), (150.0, 200.0), (200.0, 250.0), (250.0, 300.0)),
        )

        restored = ManualMidfieldLine.from_dict(observation.to_dict())

        self.assertEqual(restored, observation)
        self.assertTrue(np.allclose(restored.equation, observation.equation))

    def test_requires_exactly_five_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "precies vijf"):
            ManualMidfieldLine.fit("match.mp4", 0, 0.0, ((0.0, 0.0), (1.0, 1.0)))


if __name__ == "__main__":
    unittest.main()
