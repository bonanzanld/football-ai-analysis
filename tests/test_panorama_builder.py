from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from football_ai.pitch.panorama_builder import (
    FrameRegistrationDiagnostics,
    PanoramaBuilder,
)


class PanoramaBuilderDiagnosticsTests(unittest.TestCase):
    def test_marks_geometrically_duplicate_frame_as_redundant(self) -> None:
        frames = [
            SimpleNamespace(frame=np.zeros((100, 200, 3), dtype=np.uint8)),
            SimpleNamespace(frame=np.zeros((100, 200, 3), dtype=np.uint8)),
        ]
        diagnostics = FrameRegistrationDiagnostics(
            source_frame_index=1,
            target_frame_index=0,
            method="ORB affine",
            candidate_matches=100,
            inlier_count=80,
            median_error_pixels=1.5,
        )

        report = PanoramaBuilder(
            selected_frames=frames,
            frame_transforms=[np.eye(3), np.eye(3)],
            registration_diagnostics=[diagnostics],
        ).analyze()

        second = report.frame_reports[1]
        self.assertTrue(second.redundant)
        self.assertEqual(second.maximum_overlap_ratio, 1.0)
        self.assertEqual(second.new_coverage_ratio, 0.0)
        self.assertEqual(second.registration, diagnostics)
        self.assertAlmostEqual(diagnostics.inlier_ratio, 0.8)
        self.assertTrue(report.warnings)

    def test_reports_new_coverage_for_translated_frame(self) -> None:
        frames = [
            SimpleNamespace(frame=np.zeros((100, 200, 3), dtype=np.uint8)),
            SimpleNamespace(frame=np.zeros((100, 200, 3), dtype=np.uint8)),
        ]
        translated = np.array(
            [[1.0, 0.0, 100.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        report = PanoramaBuilder(
            selected_frames=frames,
            frame_transforms=[np.eye(3), translated],
        ).analyze()

        second = report.frame_reports[1]
        self.assertFalse(second.redundant)
        self.assertAlmostEqual(second.maximum_overlap_ratio, 99.0 / 199.0)
        self.assertAlmostEqual(second.new_coverage_ratio, 100.0 / 199.0)


if __name__ == "__main__":
    unittest.main()
