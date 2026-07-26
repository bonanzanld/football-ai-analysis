import unittest

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.camera_anchor_bank_3d import CameraAnchor3D, CameraAnchorBank3D
from football_ai.calibration.camera_anchor_runtime import CameraAnchorRuntime
from football_ai.calibration.camera_projection_3d import CameraProjection3D
from football_ai.calibration.reference_3d import create_field_reference_3d


class CameraAnchorRuntimeTests(unittest.TestCase):
    @staticmethod
    def _frame(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        frame = np.zeros((480, 800, 3), dtype=np.uint8)
        for _ in range(400):
            point = tuple(rng.integers((8, 8), (792, 472)).tolist())
            cv2.circle(frame, point, int(rng.integers(2, 7)), tuple(int(v) for v in rng.integers(70, 256, 3)), -1)
        return frame

    @staticmethod
    def _anchor(anchor_id: str, goal: str, projection: CameraProjection3D) -> CameraAnchor3D:
        return CameraAnchor3D(anchor_id, goal, 1, 0.0, 1, 0.0 if goal == "A" else 1.0, projection, 1.0, 2.0)

    def test_exact_anchor_produces_valid_projection(self) -> None:
        projection = CameraProjection3D(np.asarray(((8.0, 0.0, 0.0, 120.0), (0.0, 8.0, 0.0, 80.0), (0.0, 0.0, 0.0, 1.0))))
        anchors = (self._anchor("goal-a", "A", projection), self._anchor("goal-b", "B", projection))
        frames = {"goal-a": self._frame(1), "goal-b": self._frame(2)}
        runtime = CameraAnchorRuntime(
            CameraAnchorBank3D("8v8", "match.mp4", 64.0, 42.5, anchors),
            create_field_reference_3d(create_detection_profile("8v8")),
            frames,
        )
        result = runtime.project(frames["goal-a"])
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.anchor_id, "goal-a")


if __name__ == "__main__":
    unittest.main()
