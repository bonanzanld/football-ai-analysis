import unittest

import cv2
import numpy as np

from football_ai.calibration.bootstrap.local_mask_tracker import LocalMaskTracker
from football_ai.calibration.bootstrap.anchored_mask_tracker import AnchoredMaskTracker


class LocalMaskTrackerTests(unittest.TestCase):
    def test_tracks_short_camera_translation(self) -> None:
        rng = np.random.default_rng(12)
        first = np.zeros((300, 500, 3), np.uint8)
        for x, y in rng.integers((10, 10), (490, 290), size=(500, 2)):
            cv2.circle(first, (int(x), int(y)), 2, (255, 255, 255), -1)
        matrix = np.float32([[1.0, 0.0, 14.0], [0.0, 1.0, -6.0]])
        second = cv2.warpAffine(first, matrix, (500, 300))
        polygon = np.array([[50.0, 50.0], [450.0, 50.0], [450.0, 250.0], [50.0, 250.0]])
        tracker = LocalMaskTracker(first, polygon)

        result = tracker.update(second)

        self.assertTrue(result.reliable)
        np.testing.assert_allclose(result.polygon, polygon + (14.0, -6.0), atol=1.0)

    def test_recovers_from_a_trusted_anchor_after_unrelated_frame(self) -> None:
        rng = np.random.default_rng(24)
        anchor = np.zeros((300, 500, 3), np.uint8)
        for x, y in rng.integers((10, 10), (490, 290), size=(900, 2)):
            cv2.circle(anchor, (int(x), int(y)), 2, (255, 255, 255), -1)
        polygon = np.array([[50.0, 50.0], [450.0, 50.0], [450.0, 250.0], [50.0, 250.0]])
        tracker = AnchoredMaskTracker(anchor, polygon, 100, "goal-A")
        blank = np.zeros_like(anchor)
        self.assertFalse(tracker.update(blank, 101).reliable)

        matrix = np.float32([[1.0, 0.0, 22.0], [0.0, 1.0, -8.0]])
        returned = cv2.warpAffine(anchor, matrix, (500, 300))
        result = tracker.update(returned, 102)

        self.assertTrue(result.reliable)
        self.assertEqual(result.mode, "anchor")
        self.assertEqual(result.anchor_id, "goal-A")
        np.testing.assert_allclose(result.polygon, polygon + (22.0, -8.0), atol=1.5)


if __name__ == "__main__":
    unittest.main()
