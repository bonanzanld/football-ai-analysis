import unittest

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.camera_projection_3d import CameraProjection3D
from football_ai.calibration.local_anchor_projection import estimate_local_anchor_projection
from football_ai.calibration.reference_3d import create_field_reference_3d


class LocalAnchorProjectionTests(unittest.TestCase):
    @staticmethod
    def _textured_frame() -> np.ndarray:
        rng = np.random.default_rng(7)
        frame = np.zeros((480, 800, 3), dtype=np.uint8)
        for _ in range(350):
            point = tuple(rng.integers((8, 8), (792, 472)).tolist())
            cv2.circle(frame, point, int(rng.integers(2, 7)), tuple(int(v) for v in rng.integers(70, 256, 3)), -1)
        return frame

    def test_moves_projection_directly_from_anchor(self) -> None:
        anchor = self._textured_frame()
        transform = np.asarray(((1.0, 0.0, 22.0), (0.0, 1.0, 12.0), (0.0, 0.0, 1.0)))
        frame = cv2.warpPerspective(anchor, transform, (800, 480))
        projection = CameraProjection3D(
            np.asarray(((8.0, 0.0, 0.0, 120.0), (0.0, 8.0, 0.0, 80.0), (0.0, 0.0, 0.0, 1.0)))
        )
        reference = create_field_reference_3d(create_detection_profile("8v8"))
        result = estimate_local_anchor_projection(anchor, frame, projection, reference)
        self.assertTrue(result.valid, result.reason)
        expected = transform @ projection.matrix
        expected /= expected[2, 3]
        actual = result.projection.matrix / result.projection.matrix[2, 3]
        np.testing.assert_allclose(actual, expected, atol=0.8)

    def test_rejects_unrelated_frame(self) -> None:
        anchor = self._textured_frame()
        unrelated = np.zeros_like(anchor)
        projection = CameraProjection3D(np.asarray(((8.0, 0.0, 0.0, 120.0), (0.0, 8.0, 0.0, 80.0), (0.0, 0.0, 0.0, 1.0))))
        reference = create_field_reference_3d(create_detection_profile("8v8"))
        result = estimate_local_anchor_projection(anchor, unrelated, projection, reference)
        self.assertFalse(result.valid)


if __name__ == "__main__":
    unittest.main()
