import unittest

import numpy as np

from football_ai.calibration.midfield_atlas_anchor import (
    create_bridged_midfield_patch,
    create_midfield_patch,
    create_positioned_midfield_patch,
)


class MidfieldAtlasAnchorTests(unittest.TestCase):
    def test_transports_metric_ground_to_midfield_frame(self):
        ground_to_reference = np.asarray(((8, 0, 100), (0, 7, 80), (0.001, 0, 1)), float)
        midfield_to_reference = np.asarray(((1, 0, 40), (0, 1, 5), (0, 0, 1)), float)
        patch = create_midfield_patch(100, ground_to_reference, midfield_to_reference, 64, 42.5, 1.0, 3.0)
        expected = np.linalg.inv(midfield_to_reference) @ ground_to_reference
        expected /= expected[2, 2]
        np.testing.assert_allclose(patch.ground_to_anchor, expected)
        self.assertEqual(patch.patch_id, "midfield")

    def test_rejects_inconsistent_graph(self):
        with self.assertRaises(ValueError):
            create_midfield_patch(100, np.eye(3), np.eye(3), 64, 42.5, 9.0, 10.0)

    def test_bridges_each_end_line_from_its_own_goal_projection(self):
        goal_a = np.asarray(((8, 0, 100), (0, 7, 80), (0.001, 0, 1)), float)
        goal_b = np.asarray(((7, 0, 130), (0, 8, 60), (0.0005, 0, 1)), float)
        patch = create_bridged_midfield_patch(100, goal_a, goal_b, 64, 42.5, 1.0, 3.0)
        for point, source in (
            ((0.0, 0.0), goal_a),
            ((0.0, 42.5), goal_a),
            ((64.0, 0.0), goal_b),
            ((64.0, 42.5), goal_b),
        ):
            expected = source @ np.asarray((*point, 1.0))
            expected = expected[:2] / expected[2]
            np.testing.assert_allclose(patch.project(point), expected, atol=1e-4)

    def test_positioned_patch_passes_through_manual_sideline_points(self):
        goal_a = np.asarray(((8, 0, 100), (0, 7, 80), (0.001, 0, 1)), float)
        goal_b = np.asarray(((7, 0, 130), (0, 8, 60), (0.0005, 0, 1)), float)
        reference_line = np.asarray((0.02, -1.0, 310.0), float)
        reference_line /= np.linalg.norm(reference_line[:2])
        rear, front = (500.0, 320.0), (500.0, 650.0)
        patch = create_positioned_midfield_patch(
            100, goal_a, goal_b, reference_line, rear, front, 64, 42.5, 1.0, 3.0
        )
        for point, y in ((rear, 0.0), (front, 42.5)):
            start = np.asarray((*patch.project((0.0, y)), 1.0))
            end = np.asarray((*patch.project((64.0, y)), 1.0))
            line = np.cross(start, end)
            line /= np.linalg.norm(line[:2])
            self.assertAlmostEqual(float(line @ np.asarray((*point, 1.0))), 0.0, delta=1e-4)

    def test_positioned_patch_can_infer_invisible_front_sideline(self):
        goal_a = np.asarray(((8, 0, 100), (0, 7, 80), (0.001, 0, 1)), float)
        goal_b = np.asarray(((7, 0, 130), (0, 8, 60), (0.0005, 0, 1)), float)
        reference_line = np.asarray((0.02, -1.0, 310.0), float)
        reference_line /= np.linalg.norm(reference_line[:2])
        rear = (500.0, 320.0)
        patch = create_positioned_midfield_patch(
            100, goal_a, goal_b, reference_line, rear, None, 64, 42.5, 1.0, 3.0
        )
        start = np.asarray((*patch.project((0.0, 0.0)), 1.0))
        end = np.asarray((*patch.project((64.0, 0.0)), 1.0))
        line = np.cross(start, end)
        line /= np.linalg.norm(line[:2])
        self.assertAlmostEqual(float(line @ np.asarray((*rear, 1.0))), 0.0, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
