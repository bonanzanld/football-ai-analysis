import unittest

import cv2
import numpy as np

from football_ai.calibration.camera_anchor_recognition import (
    AnchorRecognitionStatus,
    CameraAnchorRecognizer,
    _point_coverage,
)


class CameraAnchorRecognitionTests(unittest.TestCase):
    @staticmethod
    def _pattern(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        for _ in range(120):
            center = tuple(rng.integers((10, 10), (630, 350)).tolist())
            cv2.circle(image, center, int(rng.integers(2, 9)), tuple(int(v) for v in rng.integers(80, 256, 3)), -1)
        return image

    def test_recognizes_exact_anchor_and_rejects_unrelated_view(self) -> None:
        frames = {"goal-a": self._pattern(1), "goal-b": self._pattern(2)}
        recognizer = CameraAnchorRecognizer.from_frames(frames)
        matched = recognizer.recognize(frames["goal-a"])
        unrelated = recognizer.recognize(self._pattern(3))
        self.assertEqual(matched.status, AnchorRecognitionStatus.MATCHED)
        self.assertEqual(matched.anchor_id, "goal-a")
        self.assertEqual(unrelated.status, AnchorRecognitionStatus.UNKNOWN)

    def test_coverage_uses_spatial_hull_not_only_point_count(self) -> None:
        spread = np.asarray(((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)))
        line = np.asarray(((0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (100.0, 0.0)))
        self.assertGreater(_point_coverage(spread, (100, 100)), 0.9)
        self.assertEqual(_point_coverage(line, (100, 100)), 0.0)


if __name__ == "__main__":
    unittest.main()
