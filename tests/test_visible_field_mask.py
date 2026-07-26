import unittest

import cv2
import numpy as np

from football_ai.calibration.bootstrap.goal_seed import GoalSeed
from football_ai.calibration.bootstrap.visible_field_mask import (
    build_field_boundary_geometry,
    build_visible_field_mask,
    interpolate_sideline_geometry,
    polygon_from_field_boundaries,
)
from football_ai.calibration.field_zone import FieldZone


class VisibleFieldMaskTests(unittest.TestCase):
    def test_builds_goal_side_wedge_and_classifies_points(self) -> None:
        seed = GoalSeed(
            "A", 10, 0.3, 1, 0.0, (100.0, 180.0), (100.0, 220.0), 5.0, 1.0,
            rear_corner=(100.0, 80.0), front_corner=(100.0, 320.0),
            rear_sideline_support=(300.0, 120.0), front_sideline_support=(300.0, 280.0),
        )
        mask = build_visible_field_mask(seed, 42.5, (640, 360))

        self.assertGreaterEqual(len(mask.tracking_polygon), 3)

        strip = build_visible_field_mask(seed, 42.5, (640, 360), include_backline=False)
        self.assertGreater(strip.frame_area_ratio, mask.frame_area_ratio)

        self.assertTrue(mask.contains((300.0, 200.0)))
        self.assertFalse(mask.contains((40.0, 200.0)))
        self.assertFalse(mask.contains((300.0, 40.0)))
        self.assertTrue(mask.contains((95.0, 200.0), margin_pixels=10.0))
        self.assertEqual(mask.classify((300.0, 200.0)), FieldZone.INSIDE)
        self.assertEqual(mask.classify((100.0, 200.0)), FieldZone.EDGE)
        self.assertEqual(mask.classify((40.0, 200.0)), FieldZone.OUTSIDE)

    def test_interpolates_infinite_sidelines_without_collapsing_the_field(self) -> None:
        first = GoalSeed(
            "A", 10, 0.3, 1, 0.0, (100.0, 180.0), (100.0, 220.0), 5.0, 1.0,
            rear_corner=(100.0, 80.0), front_corner=(100.0, 320.0),
            rear_sideline_support=(300.0, 120.0), front_sideline_support=(300.0, 280.0),
        )
        second = GoalSeed(
            "B", 20, 0.6, 2, 1.0, (540.0, 260.0), (540.0, 300.0), 5.0, 1.0,
            rear_corner=(540.0, 160.0), front_corner=(540.0, 350.0),
            rear_sideline_support=(300.0, 200.0), front_sideline_support=(300.0, 340.0),
        )
        geometry = interpolate_sideline_geometry(
            build_field_boundary_geometry(first, 42.5),
            build_field_boundary_geometry(second, 42.5),
            0.5,
            (640, 360),
        )
        polygon = polygon_from_field_boundaries(geometry, (640, 360), include_backline=False)
        self.assertGreater(abs(cv2.contourArea(polygon.astype(np.float32))), 640 * 360 * 0.20)


if __name__ == "__main__":
    unittest.main()
