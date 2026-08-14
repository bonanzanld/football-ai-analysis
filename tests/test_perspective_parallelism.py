import unittest

import numpy as np

from football_ai.calibration.perspective_parallelism import (
    align_playable_sidelines_to_vanishing_point,
    align_playable_sidelines_to_support_points,
    assess_playable_sideline_parallelism,
    estimate_vanishing_point_from_lines,
    rebuild_from_endline_goal_area_and_far_support,
    sideline_rays_from_confirmed_endline,
    sideline_support_deviation_degrees,
)


class PerspectiveParallelismTests(unittest.TestCase):
    def test_confirmed_endline_creates_two_rays_but_no_opposite_endline(self) -> None:
        rays = sideline_rays_from_confirmed_endline(
            (800.0, 300.0),
            (400.0, 260.0),
            (-500.0, 200.0),
        )

        self.assertEqual(rays[0], ((800.0, 300.0), (-500.0, 200.0)))
        self.assertEqual(rays[1], ((400.0, 260.0), (-500.0, 200.0)))
        self.assertEqual(len(rays), 2)

    def test_cone_support_only_measures_deviation_from_official_direction(self) -> None:
        observed = sideline_support_deviation_degrees(
            (100.0, 100.0),
            (200.0, 100.0),
            (0.0, 120.0),
            away_from_vanishing=True,
        )

        self.assertGreater(observed, 5.0)
        self.assertAlmostEqual(observed, 11.309932474)
    def test_estimates_common_vanishing_point(self) -> None:
        point = estimate_vanishing_point_from_lines(
            (((0.0, 100.0), (500.0, 50.0)), ((0.0, 300.0), (500.0, 150.0)))
        )
        np.testing.assert_allclose(point, (1000.0, 0.0), atol=1e-6)

    def test_aligns_both_playable_sidelines_to_reference_direction(self) -> None:
        polygon = np.asarray(((100.0, 100.0), (500.0, 80.0), (520.0, 300.0), (120.0, 340.0)))
        point = (1000.0, 0.0)
        corrected = align_playable_sidelines_to_vanishing_point(polygon, point, "B")
        quality = assess_playable_sideline_parallelism(corrected, point)
        self.assertTrue(quality.valid)
        np.testing.assert_allclose(corrected[1:3], polygon[1:3])

    def test_rejects_wrong_sideline_direction(self) -> None:
        polygon = np.asarray(((100.0, 100.0), (500.0, 200.0), (520.0, 400.0), (120.0, 300.0)))
        quality = assess_playable_sideline_parallelism(polygon, (1000.0, 0.0))
        self.assertFalse(quality.valid)

    def test_support_alignment_runs_through_clicked_cones(self) -> None:
        polygon = np.asarray(((100.0, 100.0), (500.0, 80.0), (520.0, 300.0), (120.0, 340.0)))
        rear_support = (250.0, 92.5)
        front_support = (260.0, 326.0)
        corrected = align_playable_sidelines_to_support_points(
            polygon, "A", rear_support, front_support
        )
        for start, end, support in (
            (corrected[0], corrected[1], rear_support),
            (corrected[3], corrected[2], front_support),
        ):
            direction = end - start
            offset = np.asarray(support) - start
            self.assertAlmostEqual(
                direction[0] * offset[1] - direction[1] * offset[0],
                0.0,
                places=5,
            )

    def test_rebuilds_near_corner_at_endline_goal_area_intersection(self) -> None:
        polygon = np.asarray(((100.0, 100.0), (500.0, 80.0), (520.0, 300.0), (120.0, 340.0)))
        rebuilt, vanishing = rebuild_from_endline_goal_area_and_far_support(
            polygon,
            "A",
            ((100.0, 100.0), (100.0, 340.0)),
            ((0.0, 300.0), (1000.0, 0.0)),
            (300.0, 75.0),
        )
        self.assertAlmostEqual(rebuilt[3, 0], 100.0, places=5)
        self.assertAlmostEqual(rebuilt[3, 1], 270.0, places=5)
        self.assertTrue(np.all(np.isfinite(vanishing)))


if __name__ == "__main__":
    unittest.main()
