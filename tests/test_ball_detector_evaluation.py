from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from football_ai.detection.ball_detector_evaluation import (
    evaluate_coco_ball_predictions,
    evaluate_score_thresholds,
    evaluate_tiny_ball_center_hits,
    non_maximum_suppression,
)


def _annotations(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "one.jpg", "width": 100, "height": 80}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 20, 20, 10],
                        "area": 200,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "sports ball"}],
            }
        ),
        encoding="utf-8",
    )


def test_perfect_prediction_has_perfect_coco_metrics() -> None:
    with tempfile.TemporaryDirectory() as directory:
        annotations = Path(directory) / "annotations.json"
        _annotations(annotations)

        report = evaluate_coco_ball_predictions(
            annotations,
            [{"image_id": 1, "category_id": 1, "bbox": [10, 20, 20, 10], "score": 0.9}],
        )

        assert report["map_50_95"] > 0.99
        assert report["mar"] > 0.99


def test_empty_predictions_return_zero_metrics() -> None:
    with tempfile.TemporaryDirectory() as directory:
        annotations = Path(directory) / "annotations.json"
        _annotations(annotations)

        report = evaluate_coco_ball_predictions(annotations, [])

        assert report["predictions"] == 0
        assert report["map_50_95"] == 0.0


def test_score_threshold_sweep_filters_one_inference_result() -> None:
    with tempfile.TemporaryDirectory() as directory:
        annotations = Path(directory) / "annotations.json"
        _annotations(annotations)
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 20, 20, 10], "score": 0.8},
            {"image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "score": 0.2},
        ]

        reports = evaluate_score_thresholds(annotations, predictions, [0.5, 0.1])

        assert [report["threshold"] for report in reports] == [0.1, 0.5]
        assert [report["predictions"] for report in reports] == [2, 1]
        assert reports[1]["map_50_95"] > 0.99


def test_score_threshold_sweep_rejects_invalid_threshold() -> None:
    with tempfile.TemporaryDirectory() as directory:
        annotations = Path(directory) / "annotations.json"
        _annotations(annotations)

        try:
            evaluate_score_thresholds(annotations, [], [1.1])
        except ValueError as error:
            assert "between zero and one" in str(error)
        else:
            raise AssertionError("invalid score threshold was accepted")


def test_nms_keeps_best_overlap_and_distant_box() -> None:
    boxes = np.asarray(((0, 0, 10, 10), (1, 1, 11, 11), (30, 30, 40, 40)))
    scores = np.asarray((0.8, 0.9, 0.7))

    kept = non_maximum_suppression(boxes, scores, iou_threshold=0.5)

    assert kept.tolist() == [1, 2]


def test_tiny_ball_center_hit_accepts_shift_that_has_poor_box_iou() -> None:
    with tempfile.TemporaryDirectory() as directory:
        annotations = Path(directory) / "annotations.json"
        _annotations(annotations)
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [22, 22, 4, 4], "score": 0.8}
        ]

        report = evaluate_tiny_ball_center_hits(annotations, predictions)

        assert report["center_hits"] == 1
        assert report["center_hit_recall"] == 1.0
        assert report["center_hit_recall_at_1"] == 1.0


def test_tiny_ball_center_hit_reports_candidate_rank():
    with tempfile.TemporaryDirectory() as directory:
        annotations = Path(directory) / "annotations.json"
        _annotations(annotations)
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [70, 60, 4, 4], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [22, 22, 4, 4], "score": 0.8},
        ]

        report = evaluate_tiny_ball_center_hits(annotations, predictions)

        assert report["center_hit_median_rank"] == 2.0
        assert report["center_hit_recall_at_1"] == 0.0
        assert report["center_hit_recall_at_5"] == 1.0
