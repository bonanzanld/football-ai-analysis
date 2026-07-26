import unittest

import numpy as np

from football_ai.calibration.manual_homography_refinement import (
    refine_ground_homography_with_lines,
    refine_ground_homography_with_vanishing_points,
)


class ManualHomographyRefinementTests(unittest.TestCase):
    def test_recovers_perspective_from_three_points_and_two_directions(self) -> None:
        expected = np.asarray(((20.0, 3.0, 500.0), (2.0, 15.0, 200.0), (0.01, -0.02, 1.0)))
        ground = np.asarray(((0.0, 0.0), (20.0, 0.0), (0.0, 15.0)))
        homogeneous = np.column_stack((ground, np.ones(3)))
        image_h = (expected @ homogeneous.T).T
        image = image_h[:, :2] / image_h[:, 2:3]
        vanishing = (
            tuple(expected[:2, 0] / expected[2, 0]),
            tuple(expected[:2, 1] / expected[2, 1]),
        )
        initial = expected.copy()
        initial[0, 0] *= 0.9
        result = refine_ground_homography_with_vanishing_points(initial, ground, image, vanishing)
        projected_h = (result.homography @ homogeneous.T).T
        projected = projected_h[:, :2] / projected_h[:, 2:3]
        np.testing.assert_allclose(projected, image, atol=0.5)
        self.assertLess(result.rms_point_error_px, 0.5)

    def test_recovers_projection_from_three_points_and_three_unlabelled_lines(self) -> None:
        expected = np.asarray(((20.0, 3.0, 500.0), (2.0, 15.0, 200.0), (0.01, -0.02, 1.0)))
        ground = np.asarray(((0.0, 0.0), (20.0, 0.0), (0.0, 15.0)))
        homogeneous = np.column_stack((ground, np.ones(3)))
        image_h = (expected @ homogeneous.T).T
        image = image_h[:, :2] / image_h[:, 2:3]
        vx = expected[:, 0] / expected[2, 0]
        vy = expected[:, 1] / expected[2, 1]
        lines = (
            np.cross(vx, (100.0, 300.0, 1.0)),
            np.cross(vx, (200.0, 450.0, 1.0)),
            np.cross(vy, (800.0, 250.0, 1.0)),
        )
        initial = expected.copy()
        initial[0, 0] *= 0.85
        initial[0, 1] *= 1.15
        result = refine_ground_homography_with_lines(initial, ground, image, lines)
        projected_h = (result.homography @ homogeneous.T).T
        projected = projected_h[:, :2] / projected_h[:, 2:3]
        np.testing.assert_allclose(projected, image, atol=0.75)
        self.assertLess(result.rms_line_error_px, 0.75)


if __name__ == "__main__":
    unittest.main()
