import unittest

import numpy as np

from football_ai.calibration.bootstrap.white_line_detection import WhiteLineCandidate
from football_ai.calibration.image_line_perspective import estimate_sideline_perspective


class ImageLinePerspectiveTests(unittest.TestCase):
    @staticmethod
    def _line(start, end, length=200.0):
        return WhiteLineCandidate(start, end, 0.0, length, 0.9, 0.8, 0.85, 0.85)

    def test_recovers_supported_sideline_vanishing_point(self) -> None:
        point = np.asarray((1000.0, 100.0))
        first_start = np.asarray((100.0, 300.0))
        second_start = np.asarray((200.0, 500.0))
        candidates = (
            self._line(tuple(first_start), tuple(first_start + 0.35 * (point - first_start))),
            self._line(tuple(second_start), tuple(second_start + 0.40 * (point - second_start))),
        )
        polygon = np.asarray(((100.0, 300.0), (700.0, 166.6667), (650.0, 275.0), (200.0, 500.0)))
        result = estimate_sideline_perspective(candidates, polygon, (1280, 720))
        self.assertTrue(result.valid, result.reason)
        np.testing.assert_allclose(result.vanishing_point, point, atol=1.0)

    def test_rejects_single_white_line(self) -> None:
        polygon = np.asarray(((100.0, 300.0), (700.0, 166.6667), (650.0, 275.0), (200.0, 500.0)))
        result = estimate_sideline_perspective(
            (self._line((100.0, 300.0), (400.0, 233.3333)),),
            polygon,
            (1280, 720),
        )
        self.assertFalse(result.valid)


if __name__ == "__main__":
    unittest.main()
