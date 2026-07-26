import unittest

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.reference_3d import create_field_reference_3d
from football_ai.calibration.reference_observation import (
    CameraViewObservations,
    ReferenceObservation2D,
)


class Reference3DTests(unittest.TestCase):
    def test_8v8_reference_has_metric_ground_and_goal_geometry(self) -> None:
        reference = create_field_reference_3d(create_detection_profile("8v8"))
        self.assertEqual(reference.pitch_length_m, 64.0)
        self.assertEqual(reference.pitch_width_m, 42.5)
        self.assertEqual(reference.goal_width_m, 5.0)
        self.assertEqual(reference.goal_height_m, 2.0)
        rear = reference.landmark("goal_a_rear_bottom").point
        front = reference.landmark("goal_a_front_bottom").point
        top = reference.landmark("goal_a_rear_top").point
        self.assertAlmostEqual(front.y - rear.y, 5.0)
        self.assertEqual(top.z, 2.0)

    def test_every_supported_format_builds_fixed_field_and_goal_landmarks(self) -> None:
        for match_format in ("6v6", "8v8", "11v11"):
            reference = create_field_reference_3d(create_detection_profile(match_format))
            self.assertEqual(len(reference.landmarks), 15)
            self.assertEqual(len(reference.ground_landmarks), 11)
            self.assertEqual(len(reference.elevated_landmarks), 4)

    def test_observations_round_trip_through_dict(self) -> None:
        original = CameraViewObservations(
            123,
            4,
            (ReferenceObservation2D("midline_rear", (12.5, 34.5)),),
        )
        restored = CameraViewObservations.from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_observations_require_spread_ground_points_and_height(self) -> None:
        reference = create_field_reference_3d(create_detection_profile("8v8"))
        identifiers = (
            "corner_a_rear", "corner_a_front", "corner_b_rear", "corner_b_front",
            "goal_a_rear_top", "goal_a_front_top",
        )
        view = CameraViewObservations(
            100,
            2,
            tuple(
                ReferenceObservation2D(name, point)
                for name, point in zip(
                    identifiers,
                    ((10.0, 10.0), (10.0, 90.0), (190.0, 10.0), (190.0, 90.0), (20.0, 0.0), (40.0, 0.0)),
                )
            ),
        )
        view.validate(reference)
        self.assertTrue(view.supports_ground_homography(reference))
        self.assertTrue(view.supports_3d_pose(reference))

    def test_four_collinear_ground_points_do_not_support_homography(self) -> None:
        reference = create_field_reference_3d(create_detection_profile("8v8"))
        identifiers = (
            "corner_a_rear", "goal_a_rear_bottom", "goal_a_front_bottom", "corner_a_front",
        )
        view = CameraViewObservations(
            100,
            1,
            tuple(ReferenceObservation2D(name, (10.0, float(index))) for index, name in enumerate(identifiers)),
        )
        self.assertFalse(view.supports_ground_homography(reference))


if __name__ == "__main__":
    unittest.main()
