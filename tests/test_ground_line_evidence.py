import unittest

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.ground_line_evidence import GroundLineFamily, detect_metric_ground_lines


class GroundLineEvidenceTests(unittest.TestCase):
    def test_merges_fragments_and_keeps_only_lines_of_at_least_three_metres(self) -> None:
        frame = np.full((400, 700, 3), (45, 125, 45), dtype=np.uint8)
        ground_to_image = np.asarray(((10.0, 0.0, 20.0), (0.0, 10.0, 20.0), (0.0, 0.0, 1.0)))
        cv2.line(frame, (40, 120), (210, 120), (250, 250, 250), 7)
        cv2.line(frame, (220, 120), (390, 120), (250, 250, 250), 7)
        cv2.line(frame, (500, 160), (520, 160), (250, 250, 250), 7)

        detection = detect_metric_ground_lines(
            frame, create_detection_profile("8v8"), ground_to_image, minimum_length_m=3.0
        )

        self.assertTrue(any(item.metric_length >= 30.0 for item in detection.lines))
        self.assertTrue(all(item.metric_length >= 3.0 for item in detection.lines))
        self.assertTrue(all(item.family == GroundLineFamily.LONGITUDINAL for item in detection.lines))

    def test_rejects_diagonal_non_pitch_direction(self) -> None:
        frame = np.full((400, 700, 3), (45, 125, 45), dtype=np.uint8)
        cv2.line(frame, (100, 300), (500, 100), (250, 250, 250), 7)
        detection = detect_metric_ground_lines(
            frame, create_detection_profile("8v8"), np.eye(3), minimum_length_m=3.0
        )
        self.assertEqual(detection.lines, ())


if __name__ == "__main__":
    unittest.main()
