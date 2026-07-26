import unittest

import cv2
import numpy as np

from football_ai.calibration.ground_circle_evidence import (
    GroundCircleEvidence,
    detect_fixed_radius_circle_from_mask,
    estimate_circle_consensus,
)


class GroundCircleEvidenceTests(unittest.TestCase):
    def test_detects_partly_occluded_fixed_radius_circle(self) -> None:
        mask = np.zeros((400, 600), dtype=np.uint8)
        cv2.circle(mask, (310, 210), 73, 255, 5, cv2.LINE_AA)
        cv2.rectangle(mask, (260, 130), (340, 205), 0, -1)
        result = detect_fixed_radius_circle_from_mask(mask, 73.2)
        self.assertIsNotNone(result)
        center, support, coverage = result
        np.testing.assert_allclose(center, (310, 210), atol=4.0)
        self.assertGreater(support, 0.45)
        self.assertGreater(coverage, 0.55)

    def test_rejects_empty_mask(self) -> None:
        self.assertIsNone(detect_fixed_radius_circle_from_mask(np.zeros((300, 400), dtype=np.uint8), 73.2))

    def test_consensus_requires_repeated_metric_center(self) -> None:
        observations = tuple(
            GroundCircleEvidence(center, 9.15, 0.7, 0.8, 0.75)
            for center in ((40.0, 30.0), (40.4, 29.8), (39.7, 30.3), (70.0, 70.0))
        )
        consensus = estimate_circle_consensus(observations)
        self.assertIsNotNone(consensus)
        self.assertEqual(consensus.observations, 3)
        np.testing.assert_allclose(consensus.ground_center, (40.0, 30.0), atol=0.4)

    def test_consensus_rejects_single_false_positive(self) -> None:
        observation = GroundCircleEvidence((40.0, 30.0), 9.15, 0.7, 0.8, 0.75)
        self.assertIsNone(estimate_circle_consensus((observation,)))


if __name__ == "__main__":
    unittest.main()
