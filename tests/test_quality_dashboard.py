from __future__ import annotations

import unittest

import numpy as np

from football_ai.calibration.quality_report import (
    ControlPointContext,
    assess_calibration_quality,
    calculate_quality_report,
)
from football_ai.pitch.manual_calibrator import MultiFramePitchCalibrator


class QualityDashboardTests(unittest.TestCase):
    def test_draws_dashboard_with_quality_report(self) -> None:
        image = np.zeros((400, 800, 3), dtype=np.uint8)
        report = calculate_quality_report(
            image_points=np.array([[0.0, 0.0], [3.0, 4.0]]),
            pitch_points=np.zeros((2, 2)),
            image_to_pitch_matrix=np.eye(3),
            inlier_mask=np.array([1, 0]),
            point_contexts=[
                ControlPointContext(1, "Hoek linksachter", 0, 100),
                ControlPointContext(4, "Hoek rechtsvoor", 2, 300),
            ],
        )
        report = assess_calibration_quality(report, 42.5, 64.0)

        MultiFramePitchCalibrator._draw_quality_dashboard(image, report)

        self.assertGreater(np.count_nonzero(image), 0)
        self.assertEqual(
            MultiFramePitchCalibrator._get_point_quality(report, 1),
            report.point_errors[1],
        )
        self.assertEqual(
            MultiFramePitchCalibrator._format_dashboard_outlier(
                report.point_errors[1]
            ),
            "Hoek rechtsvoor | F3 | 5.0 px",
        )
        self.assertIn(
            "STATUS FAIL",
            MultiFramePitchCalibrator._format_dashboard_assessment(report),
        )

    def test_draws_fallback_for_legacy_calibration(self) -> None:
        image = np.zeros((240, 500, 3), dtype=np.uint8)

        MultiFramePitchCalibrator._draw_quality_dashboard(image, None)

        self.assertGreater(np.count_nonzero(image), 0)
        self.assertIsNone(
            MultiFramePitchCalibrator._get_point_quality(None, 0)
        )


if __name__ == "__main__":
    unittest.main()
