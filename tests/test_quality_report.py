from __future__ import annotations

import unittest

import numpy as np

from football_ai.calibration.quality_report import (
    CalibrationStatus,
    ControlPointContext,
    assess_calibration_quality,
    calculate_quality_report,
)


class CalculateQualityReportTests(unittest.TestCase):
    PITCH_POINTS = np.array(
        [
            [0.0, 0.0],
            [42.5, 0.0],
            [42.5, 64.0],
            [0.0, 64.0],
            [21.25, 0.0],
            [42.5, 32.0],
            [21.25, 64.0],
            [0.0, 32.0],
        ]
    )

    def test_calculates_per_point_and_separate_inlier_statistics(self) -> None:
        image_points = np.array(
            [[0.0, 0.0], [3.0, 4.0], [20.0, 0.0]],
            dtype=np.float64,
        )
        pitch_points = np.array(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float64,
        )

        report = calculate_quality_report(
            image_points=image_points,
            pitch_points=pitch_points,
            image_to_pitch_matrix=np.eye(3),
            inlier_mask=np.array([[1], [1], [0]], dtype=np.uint8),
        )

        self.assertEqual(report.point_count, 3)
        self.assertEqual(report.inlier_count, 2)
        self.assertEqual(report.outlier_count, 1)
        self.assertEqual(
            [point.error_pixels for point in report.point_errors],
            [0.0, 5.0, 20.0],
        )
        self.assertAlmostEqual(report.all_points.mean_error, 25.0 / 3.0)
        self.assertEqual(report.all_points.median_error, 5.0)
        self.assertAlmostEqual(
            report.all_points.rms_error,
            np.sqrt(425.0 / 3.0),
        )
        self.assertEqual(report.all_points.max_error, 20.0)
        self.assertEqual(report.inlier_points.mean_error, 2.5)
        self.assertEqual(report.inlier_points.median_error, 2.5)
        self.assertAlmostEqual(
            report.inlier_points.rms_error,
            np.sqrt(12.5),
        )
        self.assertEqual(report.inlier_points.max_error, 5.0)

    def test_json_dictionary_round_trip_preserves_report(self) -> None:
        report = calculate_quality_report(
            image_points=np.array([[0.0, 0.0], [3.0, 4.0]]),
            pitch_points=np.array([[0.0, 0.0], [0.0, 0.0]]),
            image_to_pitch_matrix=np.eye(3),
            inlier_mask=np.array([1, 0]),
        )

        restored = type(report).from_dict(report.to_dict())

        self.assertEqual(restored, report)
        self.assertEqual(report.to_dict()["inliers"], 1)
        self.assertEqual(report.to_dict()["outliers"], 1)

    def test_terminal_report_shows_both_statistic_sets_and_outliers(self) -> None:
        report = calculate_quality_report(
            image_points=np.array(
                [[0.0, 0.0], [3.0, 4.0], [20.0, 0.0]]
            ),
            pitch_points=np.zeros((3, 2)),
            image_to_pitch_matrix=np.eye(3),
            inlier_mask=np.array([1, 1, 0]),
        )

        output = report.format_terminal_report()

        self.assertIn("Controlepunten : 3", output)
        self.assertIn("Inliers        : 2", output)
        self.assertIn("Outliers       : 1", output)
        self.assertIn("Alle punten", output)
        self.assertIn("Alleen inliers", output)
        self.assertIn("Gemiddelde", output)
        self.assertIn("Mediaan", output)
        self.assertIn("RMS", output)
        self.assertIn("Maximum", output)
        self.assertIn("Punt 3: 20.0 px", output)

    def test_preserves_point_context_in_json_and_terminal_report(self) -> None:
        context = ControlPointContext(
            landmark_key=4,
            landmark_name="Hoek rechtsvoor",
            frame_index=2,
            frame_number=1234,
        )
        report = calculate_quality_report(
            image_points=np.array([[20.0, 0.0]]),
            pitch_points=np.array([[0.0, 0.0]]),
            image_to_pitch_matrix=np.eye(3),
            inlier_mask=np.array([0]),
            point_contexts=[context],
        )

        restored = type(report).from_dict(report.to_dict())
        output = restored.format_terminal_report()

        self.assertEqual(restored.point_errors[0].landmark_key, 4)
        self.assertIn("Hoek rechtsvoor", output)
        self.assertIn("selectieframe 3", output)
        self.assertIn("videoframe 1234", output)

    def test_terminal_report_handles_no_outliers(self) -> None:
        points = np.array([[1.0, 2.0]])
        report = calculate_quality_report(
            image_points=points,
            pitch_points=points,
            image_to_pitch_matrix=np.eye(3),
        )

        output = report.format_terminal_report()

        self.assertIn("Outliers       : 0", output)
        self.assertIn("Geen outliers gedetecteerd.", output)

    def test_terminal_report_handles_empty_inlier_statistics(self) -> None:
        report = calculate_quality_report(
            image_points=np.array([[1.0, 2.0]]),
            pitch_points=np.array([[1.0, 2.0]]),
            image_to_pitch_matrix=np.eye(3),
            inlier_mask=np.array([0]),
        )

        output = report.format_terminal_report()

        self.assertIn("n.v.t.", output)

    def test_assessment_fails_exact_four_point_fit(self) -> None:
        report = calculate_quality_report(
            image_points=self.PITCH_POINTS,
            pitch_points=self.PITCH_POINTS,
            image_to_pitch_matrix=np.eye(3),
            inlier_mask=np.array([1, 1, 1, 1, 0, 0, 0, 0]),
        )

        assessed = assess_calibration_quality(report, 42.5, 64.0)

        self.assertEqual(assessed.assessment.status, CalibrationStatus.FAIL)
        self.assertLessEqual(assessed.assessment.confidence_score, 49.0)
        self.assertTrue(
            any("Vier of minder" in reason for reason in assessed.assessment.failures)
        )
        self.assertFalse(assessed.is_usable)

    def test_assessment_passes_well_distributed_inliers(self) -> None:
        report = calculate_quality_report(
            image_points=self.PITCH_POINTS,
            pitch_points=self.PITCH_POINTS,
            image_to_pitch_matrix=np.eye(3),
        )

        assessed = assess_calibration_quality(report, 42.5, 64.0)

        self.assertEqual(assessed.assessment.status, CalibrationStatus.PASS)
        self.assertEqual(assessed.assessment.confidence_score, 100.0)
        self.assertTrue(assessed.is_usable)
        restored = type(assessed).from_dict(assessed.to_dict())
        self.assertEqual(restored, assessed)

    def test_assessment_warns_for_seventy_five_percent_inliers(self) -> None:
        report = calculate_quality_report(
            image_points=self.PITCH_POINTS,
            pitch_points=self.PITCH_POINTS,
            image_to_pitch_matrix=np.eye(3),
            inlier_mask=np.array([1, 1, 1, 1, 1, 1, 0, 0]),
        )

        assessed = assess_calibration_quality(report, 42.5, 64.0)

        self.assertEqual(
            assessed.assessment.status,
            CalibrationStatus.WARNING,
        )
        self.assertTrue(assessed.is_usable)

    def test_assessment_includes_frame_coverage(self) -> None:
        report = calculate_quality_report(
            image_points=self.PITCH_POINTS,
            pitch_points=self.PITCH_POINTS,
            image_to_pitch_matrix=np.eye(3),
        )

        assessed = assess_calibration_quality(
            report,
            42.5,
            64.0,
            frame_new_coverage=[1.0, 0.56, 0.0, 0.007],
        )

        self.assertEqual(assessed.assessment.selected_frame_count, 4)
        self.assertEqual(assessed.assessment.non_redundant_frame_count, 2)
        self.assertEqual(
            assessed.assessment.status,
            CalibrationStatus.WARNING,
        )

    def test_assessment_accepts_four_goalposts_with_line_support(self) -> None:
        goalposts = np.array(
            [[18.75, 0.0], [23.75, 0.0], [18.75, 64.0], [23.75, 64.0]]
        )
        report = calculate_quality_report(
            image_points=goalposts,
            pitch_points=goalposts,
            image_to_pitch_matrix=np.eye(3),
        )

        assessed = assess_calibration_quality(
            report,
            42.5,
            64.0,
            supporting_line_point_count=6,
            geometry_coverage=(1.0, 1.0, 1.0),
        )

        self.assertNotEqual(assessed.assessment.status, CalibrationStatus.FAIL)
        self.assertTrue(assessed.is_usable)

    def test_assessment_allows_goalpost_only_geometry_with_warning(self) -> None:
        goalposts = np.array(
            [[18.75, 0.0], [23.75, 0.0], [18.75, 64.0], [23.75, 64.0]]
        )
        report = calculate_quality_report(
            image_points=goalposts,
            pitch_points=goalposts,
            image_to_pitch_matrix=np.eye(3),
        )

        assessed = assess_calibration_quality(
            report,
            42.5,
            64.0,
            geometry_coverage=(1.0, 1.0, 1.0),
            model_geometry_support=True,
        )

        self.assertEqual(assessed.assessment.status, CalibrationStatus.WARNING)
        self.assertTrue(assessed.is_usable)
        self.assertTrue(
            any("vier doelpalen" in warning for warning in assessed.assessment.warnings)
        )

    def test_missing_mask_treats_every_point_as_inlier(self) -> None:
        points = np.array([[1.0, 2.0], [3.0, 4.0]])

        report = calculate_quality_report(
            image_points=points,
            pitch_points=points,
            image_to_pitch_matrix=np.eye(3),
        )

        self.assertEqual(report.inlier_count, 2)
        self.assertEqual(report.outlier_count, 0)
        self.assertEqual(report.inlier_points.mean_error, 0.0)

    def test_empty_inlier_selection_has_empty_statistics(self) -> None:
        report = calculate_quality_report(
            image_points=np.array([[1.0, 2.0]]),
            pitch_points=np.array([[1.0, 2.0]]),
            image_to_pitch_matrix=np.eye(3),
            inlier_mask=np.array([0]),
        )

        self.assertEqual(report.inlier_points.point_count, 0)
        self.assertIsNone(report.inlier_points.mean_error)
        self.assertIsNone(report.inlier_points.median_error)
        self.assertIsNone(report.inlier_points.rms_error)
        self.assertIsNone(report.inlier_points.max_error)

    def test_rejects_mismatched_point_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "evenveel punten"):
            calculate_quality_report(
                image_points=np.zeros((2, 2)),
                pitch_points=np.zeros((1, 2)),
                image_to_pitch_matrix=np.eye(3),
            )

    def test_rejects_singular_homography(self) -> None:
        with self.assertRaisesRegex(ValueError, "niet inverteerbaar"):
            calculate_quality_report(
                image_points=np.zeros((1, 2)),
                pitch_points=np.zeros((1, 2)),
                image_to_pitch_matrix=np.zeros((3, 3)),
            )


if __name__ == "__main__":
    unittest.main()
