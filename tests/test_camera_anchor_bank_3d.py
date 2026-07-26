from pathlib import Path
import tempfile
import unittest

import numpy as np

from football_ai.calibration.camera_anchor_bank_3d import (
    CameraAnchor3D,
    CameraAnchorBank3D,
    load_camera_anchor_bank,
    refine_camera_anchor_bank_ground,
    save_camera_anchor_bank,
)
from football_ai.calibration.camera_projection_3d import CameraProjection3D


class CameraAnchorBank3DTests(unittest.TestCase):
    @staticmethod
    def _anchor(anchor_id: str, goal_id: str, position: float, offset: float) -> CameraAnchor3D:
        projection = CameraProjection3D(
            np.asarray(((10.0, 0.0, 0.0, offset), (0.0, 10.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0)))
        )
        return CameraAnchor3D(anchor_id, goal_id, 10, 1.0, 2, position, projection, 2.0, 4.0)

    def test_round_trip_preserves_projection_and_metric_mapping(self) -> None:
        bank = CameraAnchorBank3D(
            "8v8", "match.mp4", 64.0, 42.5,
            (self._anchor("goal-a", "A", 0.1, 0.0), self._anchor("goal-b", "B", 0.9, 5.0)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchors.json"
            save_camera_anchor_bank(bank, path)
            restored = load_camera_anchor_bank(path)
        self.assertEqual(restored.match_format, "8v8")
        np.testing.assert_allclose(restored.anchors[0].projection.image_to_ground((100.0, 50.0)), (10.0, 5.0))

    def test_selects_nearest_validated_view_without_interpolation(self) -> None:
        left = self._anchor("goal-a", "A", 0.1, 0.0)
        right = self._anchor("goal-b", "B", 0.9, 5.0)
        bank = CameraAnchorBank3D("8v8", "match.mp4", 64.0, 42.5, (left, right))
        self.assertEqual(bank.nearest_view(0.2).anchor_id, "goal-a")
        self.assertEqual(bank.nearest_view(0.8).anchor_id, "goal-b")

    def test_round_trip_preserves_intermediate_provenance(self) -> None:
        primary = self._anchor("goal-a", "A", 0.1, 0.0)
        intermediate = CameraAnchor3D(
            "local-20", "A", 20, 2.0, 2, None, primary.projection, 2.0, 4.0,
            anchor_type="intermediate", parent_anchor_id="goal-a",
            local_inliers=100, local_inlier_ratio=0.8, local_coverage=0.2,
        )
        restored = CameraAnchorBank3D.from_dict(
            CameraAnchorBank3D("8v8", "match.mp4", 64.0, 42.5, (primary, intermediate)).to_dict()
        )
        self.assertEqual(restored.anchors[1].parent_anchor_id, "goal-a")
        self.assertIsNone(restored.anchors[1].view_position)

    def test_refined_primary_ground_is_propagated_to_intermediate(self) -> None:
        primary = self._anchor("goal-a", "A", 0.1, 0.0)
        motion = np.asarray(((1.0, 0.0, 8.0), (0.0, 1.0, 3.0), (0.0, 0.0, 1.0)))
        intermediate = CameraAnchor3D(
            "local-20", "A", 20, 2.0, 2, None,
            CameraProjection3D(motion @ primary.projection.matrix), 2.0, 4.0,
            anchor_type="intermediate", parent_anchor_id="goal-a",
        )
        other = self._anchor("goal-b", "B", 0.9, 5.0)
        refined_h = primary.projection.ground_homography().copy()
        refined_h[0, 2] += 20.0
        report = {"parallelism_quality": {"goal-a": {"refined_ground_homography": refined_h.tolist()}}}
        refined = refine_camera_anchor_bank_ground(
            CameraAnchorBank3D("8v8", "match.mp4", 64.0, 42.5, (primary, other, intermediate)),
            report,
        )
        items = {item.anchor_id: item for item in refined.anchors}
        np.testing.assert_allclose(
            items["local-20"].projection.ground_homography(),
            motion @ items["goal-a"].projection.ground_homography(),
        )
        self.assertFalse(np.allclose(
            items["goal-a"].projection.ground_homography(),
            primary.projection.ground_homography(),
        ))


if __name__ == "__main__":
    unittest.main()
