import unittest

import cv2
import numpy as np

from football_ai.calibration.fixed_camera_pose import FixedCameraPointConstraint, estimate_fixed_camera_poses
from football_ai.calibration.reference_3d import create_field_reference_3d
from football_ai.calibration.reference_observation import CameraViewObservations, ReferenceObservation2D
from football_ai.calibration.bootstrap.detection_profile import create_detection_profile


class FixedCameraPoseTests(unittest.TestCase):
    def test_recovers_shared_camera_center_from_two_views(self) -> None:
        reference = create_field_reference_3d(create_detection_profile("8v8"))
        center = np.asarray((32.0, 55.0, 8.0), dtype=np.float64)
        identifiers = (
            "goal_a_rear_bottom", "goal_a_front_bottom", "corner_a_front",
            "goal_a_rear_top", "goal_a_front_top",
        )
        second_identifiers = (
            "goal_b_rear_bottom", "goal_b_front_bottom", "corner_b_rear", "corner_b_front",
            "goal_b_rear_top", "goal_b_front_top",
        )
        views = []
        for frame_number, target, ids, focal in (
            (100, np.asarray((0.0, 21.25, 0.0)), identifiers, 1100.0),
            (200, np.asarray((64.0, 21.25, 0.0)), second_identifiers, 1250.0),
        ):
            rotation = _look_at(center, target)
            rvec, _ = cv2.Rodrigues(rotation)
            translation = -rotation @ center.reshape(3, 1)
            camera_matrix = np.asarray(((focal, 0.0, 640.0), (0.0, focal, 360.0), (0.0, 0.0, 1.0)))
            world = np.asarray([reference.landmark(item).point.as_tuple() for item in ids])
            image, _ = cv2.projectPoints(world, rvec, translation, camera_matrix, None)
            observations = tuple(
                ReferenceObservation2D(item, tuple(point))
                for item, point in zip(ids, image.reshape(-1, 2))
            )
            views.append(CameraViewObservations(frame_number, frame_number / 30.0, observations))
        result = estimate_fixed_camera_poses(reference, tuple(views), (1280, 720))
        np.testing.assert_allclose(result.camera_center, center, atol=0.5)
        self.assertLess(result.rms_error_px, 0.5)
        self.assertEqual(result.pitch_length_m, 64.0)
        self.assertEqual(result.pitch_width_m, 42.5)

    def test_keeps_estimated_metric_pitch_dimensions_inside_bounds(self) -> None:
        nominal = create_detection_profile("8v8")
        reference = create_field_reference_3d(nominal)
        actual_length, actual_width = 61.0, 44.0
        actual_profile = type(nominal)(
            nominal.match_format, nominal.name, actual_length, actual_width,
            nominal.goal_width_m, nominal.goal_height_m,
            nominal.white_line_evidence_weight, nominal.boundary_marker_evidence_weight,
            nominal.goal_evidence_weight, nominal.notes,
        )
        actual_reference = create_field_reference_3d(actual_profile)
        center = np.asarray((30.0, 50.0, 3.8), dtype=np.float64)
        views = []
        constraints = []
        for frame, goal, target in (
            (100, "a", np.asarray((0.0, actual_width / 2.0, 1.0))),
            (200, "b", np.asarray((actual_length, actual_width / 2.0, 1.0))),
        ):
            ids = tuple(
                f"goal_{goal}_{suffix}"
                for suffix in ("rear_bottom", "rear_top", "front_top", "front_bottom")
            )
            rotation = _look_at(center, target)
            rvec, _ = cv2.Rodrigues(rotation)
            translation = -rotation @ center.reshape(3, 1)
            focal = 1800.0
            camera_matrix = np.asarray(((focal, 0.0, 640.0), (0.0, focal, 360.0), (0.0, 0.0, 1.0)))
            world = np.asarray([actual_reference.landmark(item).point.as_tuple() for item in ids])
            image, _ = cv2.projectPoints(world, rvec, translation, camera_matrix, None)
            views.append(CameraViewObservations(frame, 0.0, tuple(
                ReferenceObservation2D(item, tuple(point))
                for item, point in zip(ids, image.reshape(-1, 2))
            )))
            for side in ("rear", "front"):
                corner_id = f"corner_{goal}_{side}"
                corner_world = actual_reference.landmark(corner_id).point.as_tuple()
                corner_image, _ = cv2.projectPoints(
                    np.asarray((corner_world,), dtype=np.float64),
                    rvec, translation, camera_matrix, None,
                )
                nominal_corner = reference.landmark(corner_id).point.as_tuple()
                constraints.append(
                    FixedCameraPointConstraint(
                        len(views) - 1, nominal_corner, tuple(corner_image.reshape(2)), 100.0
                    )
                )
        result = estimate_fixed_camera_poses(
            reference, tuple(views), (1280, 720),
            point_constraints=tuple(constraints),
            focal_length_prior_px=1800.0, focal_prior_weight=100.0,
            camera_height_prior_m=3.8, camera_height_weight=100.0,
            camera_center_prior_xy=(30.0, 50.0), camera_center_weight=10.0,
            pitch_dimension_bounds=((55.0, 70.0), (38.0, 47.0)),
            camera_outside_clearance_m=0.5, camera_outside_weight=100.0,
        )
        self.assertGreaterEqual(result.pitch_length_m, 55.0)
        self.assertLessEqual(result.pitch_length_m, 70.0)
        self.assertGreaterEqual(result.pitch_width_m, 38.0)
        self.assertLessEqual(result.pitch_width_m, 47.0)
        self.assertTrue(np.isfinite(result.rms_error_px))

    def test_rejects_invalid_optional_priors(self) -> None:
        reference = create_field_reference_3d(create_detection_profile("8v8"))
        with self.assertRaisesRegex(ValueError, "brandpuntsprior"):
            estimate_fixed_camera_poses(
                reference, (), (1280, 720), focal_length_prior_px=-1.0
            )


def _look_at(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.vstack((right, down, forward))


if __name__ == "__main__":
    unittest.main()
