import unittest

import numpy as np

from football_ai.detection.active_ball_classifier import (
    candidate_patch_features,
    candidate_temporal_features,
)


class ActiveBallClassifierTests(unittest.TestCase):
    def test_candidate_patch_features_have_stable_shape_and_values(self) -> None:
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[30:50, 40:60] = (255, 255, 255)

        features = candidate_patch_features(image, (40, 30, 60, 50))

        self.assertEqual(features.shape, (1600,))
        self.assertTrue(np.isfinite(features).all())

    def test_candidate_patch_rejects_box_outside_image(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "outside"):
            candidate_patch_features(image, (30, 30, 40, 40))

    def test_temporal_features_are_label_free_and_have_stable_shape(self) -> None:
        candidate = {
            "frame_number": 10,
            "box": [40, 30, 50, 40],
            "confidence": 0.4,
            "label": "positive",
        }
        candidates_by_frame = {
            9: [{"box": [38, 30, 48, 40], "confidence": 0.35}],
            10: [candidate, {"box": [70, 50, 80, 60], "confidence": 0.8}],
            11: [{"box": [42, 30, 52, 40], "confidence": 0.45}],
        }

        positive = candidate_temporal_features(
            candidate,
            candidates_by_frame,
            frame_width=100,
            frame_height=80,
        )
        candidate["label"] = "negative"
        negative = candidate_temporal_features(
            candidate,
            candidates_by_frame,
            frame_width=100,
            frame_height=80,
        )

        self.assertEqual(positive.shape, (16,))
        np.testing.assert_array_equal(positive, negative)
        self.assertTrue(np.isfinite(positive).all())

    def test_temporal_features_reject_invalid_frame_dimensions(self) -> None:
        candidate = {"frame_number": 0, "box": [0, 0, 2, 2], "confidence": 0.5}

        with self.assertRaisesRegex(ValueError, "dimensions"):
            candidate_temporal_features(
                candidate,
                {0: [candidate]},
                frame_width=0,
                frame_height=20,
            )


if __name__ == "__main__":
    unittest.main()
