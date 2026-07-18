from __future__ import annotations

import unittest

import cv2
import numpy as np

from football_ai.calibration.line_calibration import (
    LinePointObservation,
    create_boundary_line_definitions,
    estimate_homography_with_line_constraints,
    fit_image_line_robustly,
)


class LineCalibrationTests(unittest.TestCase):
    def test_recovers_homography_from_points_and_line_constraints(self) -> None:
        definitions = create_boundary_line_definitions(42.5, 64.0)
        pitch_to_image = np.array(
            [[12.0, 1.5, 150.0], [0.8, 7.5, 80.0], [0.002, 0.001, 1.0]]
        )
        exact_pitch = np.array([[0.0, 0.0], [42.5, 64.0]])
        exact_image = self._project(exact_pitch, pitch_to_image)
        observations: list[LinePointObservation] = []
        line_samples = {
            1: np.array([[8.0, 0.0], [22.0, 0.0], [38.0, 0.0]]),
            2: np.array([[5.0, 64.0], [20.0, 64.0], [35.0, 64.0]]),
            3: np.array([[0.0, 12.0], [0.0, 35.0], [0.0, 55.0]]),
            4: np.array([[42.5, 10.0], [42.5, 32.0], [42.5, 53.0]]),
        }
        for line_key, samples in line_samples.items():
            for point in self._project(samples, pitch_to_image):
                observations.append(
                    LinePointObservation(line_key, tuple(point))
                )

        estimated = estimate_homography_with_line_constraints(
            exact_image,
            exact_pitch,
            observations,
            definitions,
        )

        expected = np.linalg.inv(pitch_to_image)
        expected /= expected[2, 2]
        np.testing.assert_allclose(estimated, expected, atol=1e-6)

    def test_rejects_constraints_from_only_one_line(self) -> None:
        definitions = create_boundary_line_definitions(42.5, 64.0)
        observations = [
            LinePointObservation(3, (100.0, float(y)))
            for y in range(0, 100, 10)
        ]

        with self.assertRaisesRegex(ValueError, "geen unieke homography"):
            estimate_homography_with_line_constraints(
                np.empty((0, 2)),
                np.empty((0, 2)),
                observations,
                definitions,
            )

    def test_robust_line_fit_ignores_a_wrong_click(self) -> None:
        points = np.array(
            [[10.0, 50.0], [80.0, 50.5], [160.0, 49.5], [90.0, 110.0]]
        )

        fitted = fit_image_line_robustly(points)

        self.assertEqual(fitted.inlier_mask, (True, True, True, False))
        self.assertLess(fitted.rms_error_pixels, 1.0)

    @staticmethod
    def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(
            points.astype(np.float64).reshape(-1, 1, 2),
            matrix,
        ).reshape(-1, 2)


if __name__ == "__main__":
    unittest.main()
