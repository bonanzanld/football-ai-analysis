from __future__ import annotations

import unittest

import numpy as np

from football_ai.calibration.quality_report import (
    ControlPointContext,
    calculate_quality_report,
)


class CalculateQualityReportTests(unittest.TestCase):
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
