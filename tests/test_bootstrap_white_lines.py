import unittest

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import (
    MatchFormat,
    create_detection_profile,
)
from football_ai.calibration.bootstrap.white_line_detection import detect_white_field_lines


class BootstrapWhiteLineTests(unittest.TestCase):
    def test_match_profiles_prioritize_different_boundary_evidence(self) -> None:
        six = create_detection_profile(MatchFormat.SIX_V_SIX)
        eight = create_detection_profile(MatchFormat.EIGHT_V_EIGHT)
        eleven = create_detection_profile(MatchFormat.ELEVEN_V_ELEVEN)

        self.assertGreater(six.boundary_marker_evidence_weight, six.white_line_evidence_weight)
        self.assertGreater(eight.goal_evidence_weight, eight.white_line_evidence_weight)
        self.assertEqual(eleven.white_line_evidence_weight, 1.0)
        self.assertLess(six.goal_width_m, eight.goal_width_m)
        self.assertLess(eight.goal_width_m, eleven.goal_width_m)

    def test_detects_white_lines_surrounded_by_grass(self) -> None:
        frame = np.full((360, 640, 3), (45, 125, 45), dtype=np.uint8)
        cv2.line(frame, (40, 290), (600, 250), (245, 245, 245), 7)
        cv2.line(frame, (180, 330), (210, 80), (245, 245, 245), 7)

        detection = detect_white_field_lines(frame, create_detection_profile("11v11"))

        self.assertGreaterEqual(len(detection.candidates), 2)
        self.assertTrue(all(item.visual_confidence > 0.4 for item in detection.candidates))
        self.assertTrue(all(item.profile_evidence == item.visual_confidence for item in detection.candidates))

    def test_profile_changes_evidence_not_visual_detection(self) -> None:
        frame = np.full((240, 400, 3), (45, 125, 45), dtype=np.uint8)
        cv2.line(frame, (25, 180), (375, 150), (250, 250, 250), 6)

        six = detect_white_field_lines(frame, create_detection_profile("6v6"))
        eleven = detect_white_field_lines(frame, create_detection_profile("11v11"))

        self.assertEqual(len(six.candidates), len(eleven.candidates))
        self.assertLess(six.candidates[0].profile_evidence, eleven.candidates[0].profile_evidence)


if __name__ == "__main__":
    unittest.main()
