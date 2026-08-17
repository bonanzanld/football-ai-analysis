import cv2
import numpy as np

from football_ai.calibration.fixed_camera_rotation import (
    constrain_homography_to_camera_rotation,
)


def test_recovers_pure_camera_rotation_from_scaled_homography():
    intrinsics = np.asarray(((900.0, 0.0, 640.0), (0.0, 900.0, 360.0), (0.0, 0.0, 1.0)))
    rotation, _ = cv2.Rodrigues(np.asarray((0.02, -0.08, 0.01), dtype=np.float64))
    expected = intrinsics @ rotation @ np.linalg.inv(intrinsics)
    expected /= expected[2, 2]

    actual = constrain_homography_to_camera_rotation(3.7 * expected, intrinsics)

    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_removes_non_rotational_scale_and_shear():
    intrinsics = np.asarray(((1000.0, 0.0, 640.0), (0.0, 1000.0, 360.0), (0.0, 0.0, 1.0)))
    distorted = np.asarray(((1.08, 0.04, 12.0), (0.01, 0.93, -8.0), (0.00002, -0.00003, 1.0)))

    constrained = constrain_homography_to_camera_rotation(distorted, intrinsics)
    normalized = np.linalg.inv(intrinsics) @ constrained @ intrinsics
    normalized /= np.cbrt(np.linalg.det(normalized))

    np.testing.assert_allclose(normalized.T @ normalized, np.eye(3), atol=1e-9)
    assert np.linalg.det(normalized) > 0.0
