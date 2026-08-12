from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco_ball_predictions(
    annotations_path: str | Path,
    predictions: Iterable[Mapping[str, object]],
    *,
    maximum_detections: int = 500,
) -> dict[str, float | int]:
    """Evaluate full-frame ball detections with COCO metrics."""

    if maximum_detections < 1:
        raise ValueError("maximum_detections must be positive")
    annotations_path = Path(annotations_path)
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    image_count = len(payload.get("images", []))
    annotation_count = len(payload.get("annotations", []))
    materialized = [dict(item) for item in predictions]
    if not materialized:
        return {
            "images": image_count,
            "ground_truth_boxes": annotation_count,
            "predictions": 0,
            "map_50_95": 0.0,
            "map_50": 0.0,
            "map_75": 0.0,
            "mar": 0.0,
        }

    with redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotations_path))
        detected = ground_truth.loadRes(materialized)
        evaluator = COCOeval(ground_truth, detected, "bbox")
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 100, maximum_detections]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "images": image_count,
        "ground_truth_boxes": annotation_count,
        "predictions": len(materialized),
        "map_50_95": float(evaluator.stats[0]),
        "map_50": float(evaluator.stats[1]),
        "map_75": float(evaluator.stats[2]),
        "mar": float(evaluator.stats[8]),
    }


def evaluate_score_thresholds(
    annotations_path: str | Path,
    predictions: Iterable[Mapping[str, object]],
    thresholds: Iterable[float],
) -> list[dict[str, float | int]]:
    """Evaluate multiple operating thresholds from one inference result."""

    materialized = [dict(item) for item in predictions]
    unique_thresholds = sorted({float(value) for value in thresholds})
    if any(not 0.0 <= value <= 1.0 for value in unique_thresholds):
        raise ValueError("score thresholds must be between zero and one")
    return [
        {
            "threshold": threshold,
            **evaluate_coco_ball_predictions(
                annotations_path,
                (
                    prediction
                    for prediction in materialized
                    if float(prediction.get("score", 0.0)) >= threshold
                ),
            ),
        }
        for threshold in unique_thresholds
    ]


def evaluate_tiny_ball_center_hits(
    annotations_path: str | Path,
    predictions: Iterable[Mapping[str, object]],
    *,
    minimum_radius_px: float = 6.0,
    ground_truth_diagonal_factor: float = 0.75,
    top_k_values: tuple[int, ...] = (1, 5, 10, 25, 50),
) -> dict[str, float | int]:
    """Evaluate approximate ball localization without requiring high tiny-box IoU."""
    if minimum_radius_px <= 0.0 or ground_truth_diagonal_factor <= 0.0:
        raise ValueError("Center-hit tolerances must be positive")
    if not top_k_values or any(value < 1 for value in top_k_values):
        raise ValueError("Top-k values must be positive")
    payload = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    truth_by_image: dict[int, list[tuple[float, float, float]]] = {}
    for item in payload.get("annotations", []):
        if int(item["category_id"]) != 1:
            continue
        x, y, width, height = (float(value) for value in item["bbox"])
        radius = max(
            minimum_radius_px,
            ground_truth_diagonal_factor * float(np.hypot(width, height)),
        )
        truth_by_image.setdefault(int(item["image_id"]), []).append(
            (x + width / 2.0, y + height / 2.0, radius)
        )
    predictions_by_image: dict[int, list[tuple[float, float, float]]] = {}
    materialized = [dict(item) for item in predictions]
    for item in materialized:
        x, y, width, height = (float(value) for value in item["bbox"])
        predictions_by_image.setdefault(int(item["image_id"]), []).append(
            (x + width / 2.0, y + height / 2.0, float(item.get("score", 0.0)))
        )
    hits = 0
    hit_scores = []
    hit_ranks = []
    for image_id, truths in truth_by_image.items():
        candidates = sorted(
            predictions_by_image.get(image_id, ()), key=lambda item: item[2], reverse=True
        )
        for center_x, center_y, radius in truths:
            matching = [
                (rank, score) for rank, (x, y, score) in enumerate(candidates, start=1)
                if float(np.hypot(x - center_x, y - center_y)) <= radius
            ]
            if matching:
                hits += 1
                best_rank, best_score = min(matching, key=lambda item: item[0])
                hit_ranks.append(best_rank)
                hit_scores.append(best_score)
    truth_count = sum(len(items) for items in truth_by_image.values())
    result = {
        "center_hit_ground_truth_boxes": truth_count,
        "center_hits": hits,
        "center_hit_recall": hits / max(1, truth_count),
        "center_hit_median_score": float(np.median(hit_scores)) if hit_scores else 0.0,
        "center_hit_median_rank": float(np.median(hit_ranks)) if hit_ranks else 0.0,
        "predictions_per_image": len(materialized) / max(1, len(payload.get("images", []))),
    }
    for top_k in sorted(set(top_k_values)):
        result[f"center_hit_recall_at_{top_k}"] = sum(rank <= top_k for rank in hit_ranks) / max(
            1, truth_count
        )
    return result


def non_maximum_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Return indices kept after class-agnostic NMS."""

    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError("boxes must have shape (N, 4)")
    if scores.shape != (len(boxes),):
        raise ValueError("scores must have shape (N,)")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between zero and one")
    if not len(boxes):
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    kept: list[int] = []
    while len(order):
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        others = order[1:]
        intersection_width = np.maximum(
            0.0, np.minimum(x2[current], x2[others]) - np.maximum(x1[current], x1[others])
        )
        intersection_height = np.maximum(
            0.0, np.minimum(y2[current], y2[others]) - np.maximum(y1[current], y1[others])
        )
        intersection = intersection_width * intersection_height
        union = areas[current] + areas[others] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        )
        order = others[iou <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)
