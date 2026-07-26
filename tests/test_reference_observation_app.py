import json
from pathlib import Path
import tempfile
import unittest

from football_ai.calibration.bootstrap.goal_seed import GoalSeed
from football_ai.calibration.reference_observation_app import (
    ObservationCollectionResult,
    prefill_goal_observations,
    orient_projection_toward_field,
    save_observation_result,
)
from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.camera_projection_3d import CameraProjection3D, CameraProjectionEstimate
from football_ai.calibration.reference_3d import create_field_reference_3d
import numpy as np
from football_ai.calibration.reference_observation import CameraViewObservations


class ReferenceObservationAppTests(unittest.TestCase):
    def test_prefill_maps_goal_b_order_to_metric_landmarks(self) -> None:
        seed = GoalSeed(
            "B", 100, 3.3, 2, 1.0,
            (10.0, 20.0), (30.0, 40.0), 5.0, 0.2,
            rear_corner=(5.0, 6.0), front_corner=(50.0, 60.0),
        )
        observations = prefill_goal_observations(seed)
        self.assertEqual(
            [item.landmark_id for item in observations],
            ["goal_b_rear_bottom", "goal_b_front_bottom", "corner_b_rear", "corner_b_front"],
        )
        self.assertEqual(observations[0].image_point, (10.0, 20.0))

    def test_incomplete_result_is_saved_honestly(self) -> None:
        view = CameraViewObservations(100, 2, ())
        result = ObservationCollectionResult(view, ("midline_rear",), ("midline_rear",), None, "onvoldoende")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save_observation_result(result, path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(data["solved"])
        self.assertEqual(data["skipped_landmarks"], ["midline_rear"])
        self.assertEqual(data["failure_reason"], "onvoldoende")

    def test_sideline_support_selects_pitch_side_of_goal_plane(self) -> None:
        reference = create_field_reference_3d(create_detection_profile("8v8"))
        projection = CameraProjection3D(
            np.asarray(((10.0, 0.0, 0.0, 0.0), (0.0, 10.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
        )
        estimate = CameraProjectionEstimate(projection, (), 0.0, 0.0)
        seed = GoalSeed(
            "B", 1, 0.0, 1, 1.0,
            (640.0, 187.5), (640.0, 237.5), 5.0, 1.0,
            rear_corner=(640.0, 0.0), front_corner=(640.0, 425.0),
            rear_sideline_support=(500.0, 0.0), front_sideline_support=(500.0, 425.0),
        )
        oriented = orient_projection_toward_field(reference, estimate, seed)
        projected_other_end = oriented.projection.project(reference.landmark("corner_a_rear").point)
        self.assertLess(projected_other_end[0], 640.0)


if __name__ == "__main__":
    unittest.main()
