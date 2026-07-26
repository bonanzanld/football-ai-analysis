import unittest

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.goal_plane_camera import estimate_camera_from_goal_plane
from football_ai.calibration.reference_3d import create_field_reference_3d
from football_ai.calibration.reference_observation import CameraViewObservations, ReferenceObservation2D


class GoalPlaneCameraTests(unittest.TestCase):
    def test_recovers_ground_projection_from_one_goal_plane(self) -> None:
        reference = create_field_reference_3d(create_detection_profile("8v8"))
        width, height, focal = 1280, 720, 900.0
        camera_matrix = np.asarray(((focal, 0.0, 640.0), (0.0, focal, 360.0), (0.0, 0.0, 1.0)))
        camera_position = np.asarray((32.0, 48.0, 4.0))
        target = np.asarray((64.0, 21.25, 1.0))
        forward = (target - camera_position) / np.linalg.norm(target - camera_position)
        right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.vstack((right, down, forward))
        translation = (-rotation @ camera_position).reshape(3, 1)
        projection = camera_matrix @ np.hstack((rotation, translation))
        names = (
            "corner_b_rear", "corner_b_front",
            "goal_b_rear_bottom", "goal_b_front_bottom",
            "goal_b_rear_top", "goal_b_front_top",
        )
        observations = []
        for name in names:
            point = reference.landmark(name).point
            image = projection @ np.asarray((*point.as_tuple(), 1.0))
            image /= image[2]
            observations.append(ReferenceObservation2D(name, (float(image[0]), float(image[1]))))
        view = CameraViewObservations(1, 1, tuple(observations))

        estimate = estimate_camera_from_goal_plane(reference, view, (width, height))

        self.assertLess(estimate.rms_error_px, 1.0)
        expected = projection @ np.asarray((32.0, 10.0, 0.0, 1.0))
        expected /= expected[2]
        actual = estimate.projection.project((32.0, 10.0, 0.0))
        np.testing.assert_allclose(actual, expected[:2], atol=15.0)

    def test_rejects_missing_goal_height(self) -> None:
        reference = create_field_reference_3d(create_detection_profile("8v8"))
        names = ("corner_b_rear", "corner_b_front", "goal_b_rear_bottom", "goal_b_front_bottom")
        view = CameraViewObservations(
            1,
            1,
            tuple(ReferenceObservation2D(name, (float(index * 10), float(index * 5))) for index, name in enumerate(names)),
        )
        with self.assertRaisesRegex(ValueError, "verticale rechthoek"):
            estimate_camera_from_goal_plane(reference, view, (1280, 720))



if __name__ == "__main__":
    unittest.main()
