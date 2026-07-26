import unittest

import cv2
import numpy as np

from football_ai.calibration.automatic_field_lines import (
    detect_goal_end_field_lines,
)


class AutomaticFieldLineTests(unittest.TestCase):
    def test_detects_backline_and_sidelines_from_clicked_corners(self) -> None:
        image = np.full((500, 800, 3), (45, 120, 45), dtype=np.uint8)
        far = (220, 180)
        near = (260, 400)
        cv2.line(image, far, near, (245, 245, 245), 6)
        cv2.line(image, far, (680, 120), (245, 245, 245), 6)
        cv2.line(image, near, (720, 350), (245, 245, 245), 6)

        result = detect_goal_end_field_lines(image, far, near, 1)

        self.assertEqual([item.line_key for item in result], [1, 3, 4])
        self.assertTrue(all(len(item.points) >= 3 for item in result))


if __name__ == "__main__":
    unittest.main()
