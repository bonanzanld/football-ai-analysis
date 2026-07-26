import unittest

import numpy as np

from football_ai.calibration.local_field_atlas import (
    LocalFieldAtlas,
    LocalFieldPatch,
    anchor_patch_to_measured_endline,
    align_patch_to_front_sideline,
)
from football_ai.calibration.manual_midfield_line import ManualMidfieldLine
from football_ai.calibration.manual_parallel_lines import (
    ManualParallelLine,
    ManualParallelLineReference,
)


class LocalFieldAtlasTests(unittest.TestCase):
    def test_blends_overlapping_local_patches(self) -> None:
        first = LocalFieldPatch(
            "left", 1, np.eye(3), ((0, 0), (40, 0), (40, 42.5), (0, 42.5)), 0.8, "goal-a",
            ("end_line_a", "sideline_front"),
            ("sideline_rear",),
        )
        second_h = np.asarray(((1, 0, 4), (0, 1, 0), (0, 0, 1)), dtype=float)
        second = LocalFieldPatch(
            "right", 2, second_h, ((24, 0), (64, 0), (64, 42.5), (24, 42.5)), 0.8, "goal-b",
            ("end_line_b", "sideline_front"),
            ("sideline_rear",),
        )
        atlas = LocalFieldAtlas("test.mp4", "8v8", 64.0, 42.5, (first, second))
        point, disagreement, used = atlas.blended_projection(
            (32.0, 20.0), {"left": np.eye(3), "right": np.eye(3)}
        )
        np.testing.assert_allclose(point, (34.0, 20.0))
        self.assertAlmostEqual(disagreement, 2.0)
        self.assertEqual(used, ("left", "right"))

    def test_uses_only_patch_covering_edge(self) -> None:
        first = LocalFieldPatch(
            "left", 1, np.eye(3), ((0, 0), (40, 0), (40, 42.5), (0, 42.5)), 1.0, "goal-a",
            ("end_line_a", "sideline_front"),
            ("sideline_rear",),
        )
        second = LocalFieldPatch(
            "right", 2, np.eye(3), ((24, 0), (64, 0), (64, 42.5), (24, 42.5)), 1.0, "goal-b",
            ("end_line_b", "sideline_front"),
            ("sideline_rear",),
        )
        atlas = LocalFieldAtlas("test.mp4", "8v8", 64.0, 42.5, (first, second))
        _point, disagreement, used = atlas.blended_projection((5.0, 10.0), {"left": np.eye(3)})
        self.assertEqual(disagreement, 0.0)
        self.assertEqual(used, ("left",))

    def test_visible_evidence_omits_artificial_patch_seam(self) -> None:
        homography = np.asarray(((10, 0, 20), (0, 10, 20), (0, 0, 1)), dtype=float)
        left = LocalFieldPatch(
            "left", 1, homography, ((0, 0), (40, 0), (40, 42.5), (0, 42.5)), 1.0, "goal-a",
            ("end_line_a", "sideline_front"),
            ("sideline_rear",),
        )
        right = LocalFieldPatch(
            "right", 2, homography, ((24, 0), (64, 0), (64, 42.5), (24, 42.5)), 1.0, "goal-b",
            ("end_line_b", "sideline_front"),
            ("sideline_rear",),
        )
        atlas = LocalFieldAtlas("test.mp4", "8v8", 64.0, 42.5, (left, right))
        evidence = atlas.visible_evidence("left", (800, 600))
        names = {item.name for item in evidence.boundary_segments}
        self.assertIn("end_line_a", names)
        self.assertNotIn("end_line_b", names)
        self.assertEqual(evidence.boundary_status["end_line_b"], "UNKNOWN")
        self.assertEqual(evidence.boundary_status["sideline_rear"], "INFERRED")
        rear = next(item for item in evidence.boundary_segments if item.name == "sideline_rear")
        self.assertEqual(rear.status, "INFERRED")
        self.assertGreater(evidence.frame_coverage, 0.0)

    def test_goal_patch_never_draws_opposite_end_line(self) -> None:
        homography = np.asarray(((10, 0, 20), (0, 10, 20), (0, 0, 1)), dtype=float)
        patch = LocalFieldPatch(
            "goal-b", 2, homography, ((24, 0), (64, 0), (64, 42.5), (24, 42.5)),
            0.9, "goal-b", ("end_line_b", "sideline_front"), ("sideline_rear",),
        )
        other = LocalFieldPatch(
            "goal-a", 1, homography, ((0, 0), (40, 0), (40, 42.5), (0, 42.5)),
            0.9, "goal-a", ("end_line_a",), (),
        )
        atlas = LocalFieldAtlas("test.mp4", "8v8", 64.0, 42.5, (other, patch))
        evidence = atlas.visible_evidence("goal-b", (800, 600))
        names = {item.name for item in evidence.boundary_segments}
        self.assertNotIn("end_line_a", names)
        self.assertIn("end_line_b", names)
        self.assertIn("sideline_front", names)

    def test_sideline_alignment_preserves_end_line(self) -> None:
        original = np.asarray(((10, 0, 100), (0, 10, 50), (0, 0, 1)), dtype=float)
        observed = np.asarray(((100, 475), (300, 515), (500, 555)), dtype=float)
        refined, rms = align_patch_to_front_sideline(original, "A", 64.0, 42.5, observed)
        for point in ((0.0, 0.0), (0.0, 42.5)):
            before = original @ np.asarray((*point, 1.0))
            after = refined @ np.asarray((*point, 1.0))
            np.testing.assert_allclose(before[:2] / before[2], after[:2] / after[2], atol=1e-6)
        self.assertLess(rms, 3.0)

    def test_both_sidelines_share_supplied_vanishing_point(self) -> None:
        original = np.asarray(((10, 0, 100), (0, 10, 50), (0.001, 0, 1)), dtype=float)
        vanishing = (2100.0, 650.0)
        observed = np.asarray(((100, 475), (300, 495), (500, 515)), dtype=float)
        refined, _rms = align_patch_to_front_sideline(
            original, "A", 64.0, 42.5, observed,
            direction_vanishing_point=vanishing,
        )
        for y in (0.0, 42.5):
            first = refined @ np.asarray((0.0, y, 1.0))
            second = refined @ np.asarray((30.0, y, 1.0))
            line = np.cross(first, second)
            line /= np.linalg.norm(line[:2])
            self.assertAlmostEqual(float(line @ np.asarray((*vanishing, 1.0))), 0.0, delta=1e-5)

    def test_measured_endline_corners_are_hard_anchors(self) -> None:
        original = np.asarray(((18, 2, 220), (1, 12, 170), (0.004, 0.001, 1)), dtype=float)
        vanishing = (3800.0, 620.0)
        rear = (310.0, 430.0)
        front = (980.0, 510.0)
        anchored = anchor_patch_to_measured_endline(
            original, "B", 64.0, 42.5, rear, front, vanishing
        )
        projected = []
        for point in ((64.0, 0.0), (64.0, 42.5)):
            value = anchored @ np.asarray((*point, 1.0))
            projected.append(value[:2] / value[2])
        np.testing.assert_allclose(projected, (rear, front), atol=1e-3)
        for y in (0.0, 42.5):
            first = anchored @ np.asarray((64.0, y, 1.0))
            second = anchored @ np.asarray((52.0, y, 1.0))
            line = np.cross(first, second)
            line /= np.linalg.norm(line[:2])
            self.assertAlmostEqual(float(line @ np.asarray((*vanishing, 1.0))), 0.0, delta=1e-3)
        full_field = []
        for point in ((0.0, 0.0), (64.0, 0.0), (64.0, 42.5), (0.0, 42.5)):
            value = anchored @ np.asarray((*point, 1.0))
            self.assertGreater(abs(float(value[2])), 1e-4)
            full_field.append(value[:2] / value[2])
        self.assertTrue(np.all(np.isfinite(full_field)))

    def test_round_trip_preserves_manual_midfield_direction(self) -> None:
        first = LocalFieldPatch(
            "left", 1, np.eye(3), ((0, 0), (40, 0), (40, 42.5), (0, 42.5)),
            0.8, "goal-a",
        )
        second = LocalFieldPatch(
            "right", 2, np.eye(3), ((24, 0), (64, 0), (64, 42.5), (24, 42.5)),
            0.8, "goal-b",
        )
        midfield = ManualMidfieldLine.fit(
            "match.mp4", 100, 4.0,
            ((10.0, 20.0), (30.0, 21.0), (50.0, 22.0), (70.0, 23.0), (90.0, 24.0)),
        )
        atlas = LocalFieldAtlas(
            "match.mp4", "8v8", 64.0, 42.5, (first, second), midfield,
            ManualParallelLineReference(
                "match.mp4",
                (
                    ManualParallelLine.from_midfield(midfield),
                    ManualParallelLine.fit("goal_area_5m", 100, 4.0, midfield.points),
                    ManualParallelLine.fit("penalty_area_16m", 100, 4.0, midfield.points),
                ),
            ),
        )

        restored = LocalFieldAtlas.from_dict(atlas.to_dict())

        self.assertEqual(restored.manual_midfield_line, midfield)
        self.assertEqual(restored.manual_parallel_lines, atlas.manual_parallel_lines)
        self.assertEqual(
            restored.manual_midfield_line.to_dict()["direction_role"],
            "parallel_to_8v8_sidelines",
        )


if __name__ == "__main__":
    unittest.main()
