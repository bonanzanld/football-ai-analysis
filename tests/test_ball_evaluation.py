import json
from pathlib import Path
import tempfile
import unittest

from football_ai.detection.ball_evaluation import (
    BallGroundTruth,
    BallPrediction,
    boxes_match,
    evaluate_ball_predictions,
    load_ball_annotations,
    load_ball_predictions,
)


class BallEvaluationTests(unittest.TestCase):
    def test_tiny_boxes_can_match_by_center_distance(self) -> None:
        self.assertTrue(boxes_match((10, 10, 14, 14), (16, 10, 20, 14), 7.0, 0.5))

    def test_counts_precision_recall_and_error_categories(self) -> None:
        annotations = [
            BallGroundTruth(1, "visible", (10, 10, 20, 20)),
            BallGroundTruth(2, "occluded", (30, 30, 40, 40), "player"),
            BallGroundTruth(3, "not_visible"),
            BallGroundTruth(4, "unreviewed"),
        ]
        predictions = [
            BallPrediction(1, (11, 11, 21, 21), 0.9),
            BallPrediction(1, (80, 80, 90, 90), 0.2),
            BallPrediction(3, (50, 50, 60, 60), 0.7),
        ]

        report = evaluate_ball_predictions(annotations, predictions)

        self.assertEqual(report["reviewed_frames"], 3)
        self.assertEqual(report["true_positives"], 1)
        self.assertEqual(report["false_positives"], 2)
        self.assertEqual(report["false_negatives"], 1)
        self.assertAlmostEqual(report["precision"], 1 / 3)
        self.assertAlmostEqual(report["recall"], 1 / 2)
        self.assertEqual(report["errors"]["duplicate_false_positive"], 1)
        self.assertEqual(report["errors"]["missed_occluded_ball"], 1)
        self.assertEqual(report["errors"]["false_positive_without_visible_ball"], 1)

    def test_wrong_location_counts_false_positive_and_false_negative(self) -> None:
        report = evaluate_ball_predictions(
            [BallGroundTruth(1, "visible", (10, 10, 20, 20))],
            [BallPrediction(1, (100, 100, 110, 110), 0.8)],
        )
        self.assertEqual(report["false_positives"], 1)
        self.assertEqual(report["false_negatives"], 1)
        self.assertEqual(report["errors"]["localization_miss"], 1)

    def test_loads_annotation_and_tracker_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations_path = root / "annotations.json"
            annotations_path.write_text(
                json.dumps(
                    {
                        "annotations": [
                            {
                                "frame_number": 7,
                                "visibility": "visible",
                                "ball_box": [1, 2, 3, 4],
                                "occlusion": "none",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            predictions_path = root / "tracking.json"
            predictions_path.write_text(
                json.dumps(
                    {
                        "observations": [
                            {
                                "frame_number": 7,
                                "box": [1, 2, 3, 4],
                                "confidence": 0.8,
                                "source": "detected",
                            },
                            {
                                "frame_number": 8,
                                "box": [1, 2, 3, 4],
                                "confidence": 0.4,
                                "source": "predicted",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            _, annotations = load_ball_annotations(annotations_path)
            predictions = load_ball_predictions(predictions_path)

        self.assertEqual(annotations[0].box, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual([item.frame_number for item in predictions], [7])
        self.assertEqual(predictions[0].stage, "tracker")

    def test_loads_multiple_prediction_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            path.write_text(
                json.dumps(
                    {
                        "predictions": [
                            {
                                "frame_number": 1,
                                "box": [1, 2, 3, 4],
                                "confidence": 0.5,
                                "stage": "raw_detector",
                            },
                            {
                                "frame_number": 1,
                                "box": [1, 2, 3, 4],
                                "confidence": 0.5,
                                "stage": "person_filtered",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            predictions = load_ball_predictions(path)

        self.assertEqual(
            [item.stage for item in predictions],
            ["raw_detector", "person_filtered"],
        )

    def test_visible_annotation_requires_box(self) -> None:
        with self.assertRaises(ValueError):
            BallGroundTruth(1, "visible")

    def test_occluded_annotation_requires_box(self) -> None:
        with self.assertRaises(ValueError):
            BallGroundTruth(1, "occluded", occlusion="player")

    def test_draft_annotation_is_excluded_from_metrics(self) -> None:
        report = evaluate_ball_predictions(
            [
                BallGroundTruth(
                    1,
                    "visible",
                    (10, 10, 20, 20),
                    review_status="ai_draft",
                )
            ],
            [BallPrediction(1, (10, 10, 20, 20), 0.9)],
        )
        self.assertEqual(report["reviewed_frames"], 0)


if __name__ == "__main__":
    unittest.main()
