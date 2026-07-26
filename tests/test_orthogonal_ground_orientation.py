import unittest

import numpy as np

from football_ai.calibration.ground_line_evidence import GroundLineFamily
from football_ai.calibration.orthogonal_ground_orientation import (
    FarGoalObservation2D,
    ImageLineObservation,
    estimate_orthogonal_ground_orientation,
    cluster_physical_lines,
    transform_line_observation,
)


class OrthogonalGroundOrientationTests(unittest.TestCase):
    @staticmethod
    def _line(family: GroundLineFamily, start: tuple[float, float], vanishing: tuple[float, float]) -> ImageLineObservation:
        start_array, vanishing_array = np.asarray(start), np.asarray(vanishing)
        end = start_array + 0.25 * (vanishing_array - start_array)
        return ImageLineObservation(family, start, tuple(map(float, end)))

    def test_recovers_two_orthogonal_vanishing_directions(self) -> None:
        first = (1640.0, 1360.0)
        second = (-360.0, 360.0)
        observations = tuple(
            self._line(GroundLineFamily.LONGITUDINAL, start, first)
            for start in ((0.0, 0.0), (0.0, 700.0), (1200.0, 0.0))
        ) + tuple(
            self._line(GroundLineFamily.TRANSVERSE, start, second)
            for start in ((100.0, 0.0), (700.0, 700.0), (1250.0, 650.0))
        )
        result = estimate_orthogonal_ground_orientation(observations, (1280, 720))
        np.testing.assert_allclose(result.longitudinal_vanishing_point, first, atol=1e-5)
        np.testing.assert_allclose(result.transverse_vanishing_point, second, atol=1e-5)
        self.assertAlmostEqual(result.focal_length_pixels, 1000.0, places=5)

    def test_transforms_line_between_frames(self) -> None:
        line = ImageLineObservation(GroundLineFamily.LONGITUDINAL, (10.0, 20.0), (30.0, 40.0))
        moved = transform_line_observation(line, np.asarray(((1.0, 0.0, 5.0), (0.0, 1.0, 7.0), (0.0, 0.0, 1.0))))
        self.assertEqual(moved.start, (15.0, 27.0))
        self.assertEqual(moved.end, (35.0, 47.0))

    def test_far_goal_is_optional_validated_metadata(self) -> None:
        goal = FarGoalObservation2D((10.0, 30.0), (30.0, 30.0), (10.0, 10.0), (30.0, 10.0))
        self.assertEqual(goal.to_dict()["goal_width_m"], 7.32)

    def test_repeated_observations_of_one_physical_line_form_one_cluster(self) -> None:
        observations = tuple(
            ImageLineObservation(
                GroundLineFamily.LONGITUDINAL,
                (0.0, float(y)),
                (500.0, float(y + 50)),
                ground_offset_m=10.0 + delta,
                source_id=f"frame-{index}",
            )
            for index, (y, delta) in enumerate(((100, 0.0), (104, 0.2), (96, -0.2)))
        )
        clusters = cluster_physical_lines(observations)
        self.assertEqual(len(clusters[GroundLineFamily.LONGITUDINAL]), 1)
        self.assertEqual(clusters[GroundLineFamily.LONGITUDINAL][0].source_count, 3)

    def test_rejects_three_lines_without_physical_spread(self) -> None:
        first = (1640.0, 1360.0)
        second = (-360.0, 360.0)
        observations = tuple(
            ImageLineObservation(
                line.family,
                line.start,
                line.end,
                ground_offset_m=float(index),
            )
            for index, line in enumerate(
                tuple(self._line(GroundLineFamily.LONGITUDINAL, start, first) for start in ((0.0, 0.0), (0.0, 20.0), (0.0, 40.0)))
                + tuple(self._line(GroundLineFamily.TRANSVERSE, start, second) for start in ((100.0, 0.0), (120.0, 0.0), (140.0, 0.0)))
            )
        )
        with self.assertRaisesRegex(ValueError, "verschillende lijnen|te dicht"):
            estimate_orthogonal_ground_orientation(observations, (1280, 720))


if __name__ == "__main__":
    unittest.main()
