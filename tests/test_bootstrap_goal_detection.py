import unittest

import cv2
import numpy as np

from football_ai.calibration.bootstrap.goal_detection import detect_goal_candidates


class BootstrapGoalDetectionTests(unittest.TestCase):
    def test_detects_two_posts_connected_by_crossbar(self) -> None:
        frame = np.full((360, 640, 3), (45, 120, 45), dtype=np.uint8)
        cv2.line(frame, (220, 170), (220, 70), (250, 250, 250), 7)
        cv2.line(frame, (440, 170), (440, 70), (250, 250, 250), 7)
        cv2.line(frame, (220, 70), (440, 70), (250, 250, 250), 7)
        cv2.line(frame, (100, 170), (560, 170), (250, 250, 250), 7)

        detection = detect_goal_candidates(frame)

        self.assertGreaterEqual(len(detection.candidates), 1)
        strongest = detection.candidates[0]
        self.assertTrue(strongest.crossbar_supported)
        self.assertAlmostEqual(strongest.center_ground[0], 330.0, delta=15.0)

    def test_does_not_accept_single_vertical_white_line_as_goal(self) -> None:
        frame = np.full((360, 640, 3), (45, 120, 45), dtype=np.uint8)
        cv2.line(frame, (300, 250), (300, 100), (250, 250, 250), 7)

        detection = detect_goal_candidates(frame)

        self.assertEqual(detection.candidates, ())


if __name__ == "__main__":
    unittest.main()
