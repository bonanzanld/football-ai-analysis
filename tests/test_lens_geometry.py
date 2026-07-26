from __future__ import annotations

import unittest

import cv2
import numpy as np

from football_ai.calibration.lens_geometry import LensIntrinsics, StraightLineObservation, estimate_radial_distortion_from_lines


class LensGeometryTests(unittest.TestCase):
    def test_estimator_straightens_distorted_lines(self) -> None:
        size = (1920, 1080)
        truth = LensIntrinsics(size, 1200.0, (960.0, 540.0), (-0.18, 0.035))
        lines = []
        for y in (180.0, 820.0):
            undistorted = np.asarray([(x, y) for x in np.linspace(80.0, 1840.0, 9)], dtype=np.float64)
            normalized = cv2.undistortPoints(undistorted.reshape(-1, 1, 2), truth.camera_matrix, None).reshape(-1, 2)
            world = np.column_stack((normalized, np.ones(len(normalized))))
            distorted, _ = cv2.projectPoints(world, np.zeros(3), np.zeros(3), truth.camera_matrix, truth.distortion_coefficients)
            lines.append(StraightLineObservation(distorted.reshape(-1, 2)))

        estimate = estimate_radial_distortion_from_lines(size, tuple(lines), initial_focal_length_px=truth.focal_length_px)

        self.assertLess(estimate.rms_straightness_px, 0.2)
        self.assertAlmostEqual(estimate.intrinsics.radial_distortion[0], -0.18, delta=0.04)


if __name__ == "__main__":
    unittest.main()
