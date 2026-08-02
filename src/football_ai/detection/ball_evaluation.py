from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable


VALID_VISIBILITY = {"visible", "occluded", "not_visible", "unreviewed"}


@dataclass(frozen=True)
class BallGroundTruth:
    frame_number: int
    visibility: str
    box: tuple[float, float, float, float] | None = None
    occlusion: str = "none"
    review_status: str = "human_reviewed"

    def __post_init__(self) -> None:
        if self.visibility not in VALID_VISIBILITY:
            raise ValueError(f"Unknown ball visibility: {self.visibility}")
        if self.visibility in {"visible", "occluded"} and self.box is None:
            raise ValueError(f"A {self.visibility} ball requires a box")
        if self.visibility in {"not_visible", "unreviewed"} and self.box is not None:
            raise ValueError(f"{self.visibility} frames cannot have a ball box")


@dataclass(frozen=True)
class BallPrediction:
    frame_number: int
    box: tuple[float, float, float, float]
    confidence: float
    stage: str = "detector"


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def boxes_match(
    ground_truth: tuple[float, float, float, float],
    prediction: tuple[float, float, float, float],
    maximum_center_distance: float = 20.0,
    minimum_iou: float = 0.10,
) -> bool:
    """Match tiny balls by overlap or center distance.

    IoU alone is unstable for an object that may span fewer than ten pixels.
    Center distance keeps the metric useful while still enforcing localization.
    """

    distance = math.dist(_box_center(ground_truth), _box_center(prediction))
    return distance <= maximum_center_distance or (
        minimum_iou > 0.0
        and _intersection_over_union(ground_truth, prediction) >= minimum_iou
    )


def evaluate_ball_predictions(
    annotations: Iterable[BallGroundTruth],
    predictions: Iterable[BallPrediction],
    *,
    maximum_center_distance: float = 20.0,
    minimum_iou: float = 0.10,
) -> dict[str, object]:
    reviewed = [
        item
        for item in annotations
        if item.visibility != "unreviewed" and item.review_status == "human_reviewed"
    ]
    predictions_by_frame: dict[int, list[BallPrediction]] = {}
    for prediction in predictions:
        predictions_by_frame.setdefault(prediction.frame_number, []).append(prediction)

    true_positives = false_positives = false_negatives = 0
    errors = {
        "missed_visible_ball": 0,
        "missed_occluded_ball": 0,
        "localization_miss": 0,
        "false_positive_without_visible_ball": 0,
        "duplicate_false_positive": 0,
    }
    frame_results: list[dict[str, object]] = []

    for annotation in reviewed:
        frame_predictions = sorted(
            predictions_by_frame.get(annotation.frame_number, []),
            key=lambda item: item.confidence,
            reverse=True,
        )
        result = "true_negative"
        if annotation.visibility in {"visible", "occluded"} and annotation.box:
            match = next(
                (
                    item
                    for item in frame_predictions
                    if boxes_match(
                        annotation.box,
                        item.box,
                        maximum_center_distance,
                        minimum_iou,
                    )
                ),
                None,
            )
            if match is None:
                false_negatives += 1
                if frame_predictions:
                    errors["localization_miss"] += 1
                    false_positives += len(frame_predictions)
                    result = "localization_miss"
                else:
                    key = (
                        "missed_occluded_ball"
                        if annotation.visibility == "occluded"
                        else "missed_visible_ball"
                    )
                    errors[key] += 1
                    result = key
            else:
                true_positives += 1
                extra = len(frame_predictions) - 1
                false_positives += extra
                errors["duplicate_false_positive"] += extra
                result = "matched"
        elif frame_predictions:
            false_positives += len(frame_predictions)
            errors["false_positive_without_visible_ball"] += len(frame_predictions)
            result = "false_positive_without_visible_ball"
        frame_results.append(
            {
                "frame_number": annotation.frame_number,
                "visibility": annotation.visibility,
                "prediction_count": len(frame_predictions),
                "result": result,
            }
        )

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    return {
        "reviewed_frames": len(reviewed),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "errors": errors,
        "frames": frame_results,
    }


def load_ball_annotations(path: str | Path) -> tuple[dict[str, object], list[BallGroundTruth]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    annotations = []
    for item in payload.get("annotations", []):
        raw_box = item.get("ball_box")
        annotations.append(
            BallGroundTruth(
                frame_number=int(item["frame_number"]),
                visibility=str(item.get("visibility", "unreviewed")),
                box=None if raw_box is None else tuple(float(value) for value in raw_box),
                occlusion=str(item.get("occlusion", "none")),
                review_status=str(item.get("review_status", "unreviewed")),
            )
        )
    return payload, annotations


def load_ball_predictions(
    path: str | Path,
    *,
    stage: str = "tracker",
) -> list[BallPrediction]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        BallPrediction(
            frame_number=int(item["frame_number"]),
            box=tuple(float(value) for value in item["box"]),
            confidence=float(item["confidence"]),
            stage=str(item.get("stage", stage)),
        )
        for item in payload.get("observations", payload.get("predictions", []))
        if item.get("source", "detected") == "detected"
    ]
