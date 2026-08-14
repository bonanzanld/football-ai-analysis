import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from football_ai.calibration.local_field_atlas import LocalFieldAtlas, LocalFieldPatch
from football_ai.calibration.local_field_atlas_runtime import (
    AtlasRuntimeProjection,
    FixedPatchTracker,
    MINIMUM_RUNTIME_INLIERS,
    MINIMUM_RUNTIME_INLIER_RATIO,
    LocalFieldAtlasTracker,
    LocalFieldAtlasRuntime,
    sideline_vanishing_error_degrees,
)


class LocalFieldAtlasRuntimeTests(unittest.TestCase):
    def test_runtime_accepts_one_patch_for_fixed_segment(self):
        patch = LocalFieldPatch(
            "goal-b", 100, np.eye(3),
            ((28.0, 0.0), (64.0, 0.0), (64.0, 42.5), (28.0, 42.5)),
            0.8, "fixed segment",
        )
        atlas = LocalFieldAtlas("match.mp4", "8v8", 64.0, 42.5, (patch,))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, "texture", (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10)
        noise = np.random.default_rng(3).integers(0, 40, frame.shape, dtype=np.uint8)

        runtime = LocalFieldAtlasRuntime(atlas, {"goal-b": cv2.add(frame, noise)})

        self.assertEqual(tuple(runtime.patch_by_id), ("goal-b",))

    @staticmethod
    def _frame(seed):
        rng = np.random.default_rng(seed)
        frame = np.zeros((480, 800, 3), np.uint8)
        for _ in range(500):
            point = tuple(rng.integers((10, 10), (790, 470)).tolist())
            cv2.circle(frame, point, int(rng.integers(2, 6)), tuple(map(int, rng.integers(60, 255, 3))), -1)
        return frame

    def test_exact_anchor_projects_atlas(self):
        homography = np.asarray(((8.0, 0.2, 120.0), (0.1, 7.0, 80.0), (0.001, 0.0005, 1.0)))
        patches = (
            LocalFieldPatch("goal-a", 1, homography, ((0, 0), (40, 0), (40, 42.5), (0, 42.5)), 0.9, "test"),
            LocalFieldPatch("goal-b", 2, homography, ((24, 0), (64, 0), (64, 42.5), (24, 42.5)), 0.9, "test"),
        )
        atlas = LocalFieldAtlas("match.mp4", "8v8", 64.0, 42.5, patches)
        frames = {"goal-a": self._frame(1), "goal-b": self._frame(2)}
        runtime = LocalFieldAtlasRuntime(atlas, frames)

        result = runtime.project(frames["goal-a"])

        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.patch_id, "goal-a")
        self.assertEqual(len(result.polygon), 4)
        self.assertIsNotNone(result.anchor_to_frame)
        self.assertIsNotNone(result.predicted_vanishing_point)

    def test_vanishing_error(self):
        self.assertAlmostEqual(
            sideline_vanishing_error_degrees((200.0, 100.0), (200.0, 100.0), (100.0, 100.0)),
            0.0,
        )
        self.assertAlmostEqual(
            sideline_vanishing_error_degrees((200.0, 100.0), (100.0, 200.0), (100.0, 100.0)),
            90.0,
        )
        self.assertAlmostEqual(
            sideline_vanishing_error_degrees((200.0, 100.0), (0.0, 100.0), (100.0, 100.0)),
            0.0,
        )

    def test_runtime_thresholds_reject_reported_false_positive(self):
        self.assertGreater(MINIMUM_RUNTIME_INLIERS, 38)
        self.assertGreater(MINIMUM_RUNTIME_INLIER_RATIO, 0.487)

    @staticmethod
    def _projection(patch_id):
        return AtlasRuntimeProjection(
            True, patch_id, np.eye(3), np.eye(3),
            ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
            Mock(), 100, 0.8, 0.3, (100.0, 100.0), "test",
        )

    def test_tracker_rejects_unapproved_patch_switch(self):
        runtime = Mock()
        goal_b = self._projection("goal-b")
        goal_a = self._projection("goal-a")
        runtime.project.side_effect = (goal_b, goal_a)
        runtime.propagate.return_value = goal_b
        tracker = LocalFieldAtlasTracker(
            runtime, lambda _frame, candidate: candidate.patch_id == "goal-b"
        )
        frame = np.zeros((20, 20, 3), np.uint8)

        tracker.update(frame)
        result = tracker.update(frame)

        self.assertEqual(result.patch_id, "goal-b")

    def test_tracker_allows_semantically_approved_patch_switch(self):
        runtime = Mock()
        goal_b = self._projection("goal-b")
        goal_a = self._projection("goal-a")
        runtime.project.side_effect = (goal_b, goal_a)
        runtime.propagate.return_value = goal_b
        tracker = LocalFieldAtlasTracker(runtime, lambda _frame, _candidate: True)
        frame = np.zeros((20, 20, 3), np.uint8)

        tracker.update(frame)
        result = tracker.update(frame)

        self.assertEqual(result.patch_id, "goal-a")

    def test_tracker_checks_all_patches_after_tracking_loss(self):
        runtime = Mock()
        runtime.patch_by_id = {
            "midfield-rear": Mock(), "goal-a": Mock(), "midfield-front": Mock()
        }
        rear = self._projection("midfield-rear")
        failed = AtlasRuntimeProjection(
            False, "midfield-rear", None, None, (), Mock(), 0, 0.0, 0.0,
            None, "lost",
        )
        front = self._projection("midfield-front")
        runtime.project.side_effect = (rear, self._projection("goal-a"))
        runtime.propagate.return_value = failed
        runtime.project_with_patch.side_effect = lambda _frame, patch_id, _recognition: (
            front if patch_id == "midfield-front" else failed
        )
        tracker = LocalFieldAtlasTracker(
            runtime, lambda _frame, candidate: candidate.patch_id.startswith("midfield")
        )
        frame = np.zeros((20, 20, 3), np.uint8)

        tracker.update(frame)
        result = tracker.update(frame)

        self.assertEqual(result.patch_id, "midfield-front")

    def test_fixed_patch_tracker_never_substitutes_another_patch(self):
        runtime = Mock()
        runtime.patch_by_id = {"midfield-rear": Mock(), "goal-a": Mock()}
        runtime.recognizer.recognize.return_value = Mock()
        rear = self._projection("midfield-rear")
        failed = AtlasRuntimeProjection(
            False, "midfield-rear", None, None, (), Mock(), 0, 0.0, 0.0,
            None, "lost",
        )
        runtime.project_with_patch.side_effect = (rear, failed)
        runtime.propagate.return_value = failed
        tracker = FixedPatchTracker(runtime, "midfield-rear")
        frame = np.zeros((20, 20, 3), np.uint8)

        self.assertTrue(tracker.update(frame).valid)
        result = tracker.update(frame)

        self.assertFalse(result.valid)
        self.assertEqual(result.patch_id, "midfield-rear")
        self.assertEqual(runtime.project_with_patch.call_count, 1)
        requested_patch = runtime.project_with_patch.call_args_list[-1].args[1]
        self.assertEqual(requested_patch, "midfield-rear")


if __name__ == "__main__":
    unittest.main()
