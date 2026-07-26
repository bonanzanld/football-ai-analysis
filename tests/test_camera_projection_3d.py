import unittest

import numpy as np

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.camera_projection_3d import estimate_camera_projection_dlt
from football_ai.calibration.reference_3d import create_field_reference_3d
from football_ai.calibration.reference_observation import CameraViewObservations, ReferenceObservation2D


class CameraProjection3DTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = create_field_reference_3d(create_detection_profile("8v8"))
        camera_matrix = np.asarray(((1100.0, 0.0, 640.0), (0.0, 1080.0, 360.0), (0.0, 0.0, 1.0)))
        angle = np.deg2rad(8.0)
        rotation = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, np.cos(angle), -np.sin(angle)), (0.0, np.sin(angle), np.cos(angle)))
        )
        translation = np.asarray(((-32.0,), (-18.0,), (70.0,)))
        self.projection = camera_matrix @ np.hstack((rotation, translation))

    def test_recovers_projection_and_ground_mapping(self) -> None:
        observations = []
        for landmark in self.reference.landmarks:
            world = np.asarray((*landmark.point.as_tuple(), 1.0))
            image = self.projection @ world
            image /= image[2]
            observations.append(ReferenceObservation2D(landmark.landmark_id, (float(image[0]), float(image[1]))))
        view = CameraViewObservations(100, 2, tuple(observations))

        estimate = estimate_camera_projection_dlt(self.reference, view)

        self.assertLess(estimate.rms_error_px, 1e-7)
        test_ground = (21.5, 17.0, 0.0)
        expected = self.projection @ np.asarray((*test_ground, 1.0))
        expected /= expected[2]
        actual = estimate.projection.project(test_ground)
        np.testing.assert_allclose(actual, expected[:2], atol=1e-6)
        recovered = estimate.projection.image_to_ground(actual)
        np.testing.assert_allclose(recovered, test_ground[:2], atol=1e-6)

    def test_rejects_a_view_without_height_information(self) -> None:
        names = ("corner_a_rear", "corner_a_front", "corner_b_rear", "corner_b_front")
        view = CameraViewObservations(
            100,
            2,
            tuple(ReferenceObservation2D(name, (float(index * 20), float(index % 2 * 30))) for index, name in enumerate(names)),
        )
        with self.assertRaisesRegex(ValueError, "hoogtepunten"):
            estimate_camera_projection_dlt(self.reference, view)


if __name__ == "__main__":
    unittest.main()
