from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detection.ball_detector_evaluation import (
    evaluate_coco_ball_predictions,
    evaluate_score_thresholds,
    evaluate_tiny_ball_center_hits,
    non_maximum_suppression,
)
from football_ai.detection.ball_detector_dataset import (
    extract_square_tile,
    square_tile_origins,
)


def _sports_ball_mask(detections: object) -> np.ndarray:
    data = getattr(detections, "data", {})
    names = data.get("class_name") if isinstance(data, dict) else None
    if names is not None and len(names) == len(detections):
        return np.asarray([str(name) == "sports ball" for name in names])
    class_ids = getattr(detections, "class_id", None)
    if class_ids is None:
        raise ValueError("RF-DETR predictions contain neither class names nor IDs")
    unique = set(int(value) for value in class_ids)
    if unique <= {0}:
        return np.ones(len(detections), dtype=bool)
    raise ValueError(f"Cannot identify sports-ball detections from class IDs: {sorted(unique)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained or fine-tuned RF-DETR on one COCO split."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", default="valid", choices=("train", "valid"))
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint", type=Path)
    checkpoint_group.add_argument("--yolo-checkpoint", type=Path)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument(
        "--score-thresholds",
        type=float,
        nargs="+",
        help="Also score these thresholds from the same inference pass.",
    )
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inference_threshold = min(
        [args.threshold, *(args.score_thresholds or [])]
    )

    annotations_path = args.dataset / args.split / "_annotations.coco.json"
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    if args.yolo_checkpoint is not None:
        from ultralytics import YOLO

        model = YOLO(str(args.yolo_checkpoint.resolve()))
        model_name = str(args.yolo_checkpoint.resolve())
        backend = "yolo"
    elif args.checkpoint is None:
        from rfdetr import RFDETRMedium

        model = RFDETRMedium()
        model_name = "rfdetr_medium_pretrained_coco"
        backend = "rfdetr"
    else:
        from rfdetr import RFDETRMedium

        model = RFDETRMedium.from_checkpoint(str(args.checkpoint.resolve()))
        model_name = str(args.checkpoint.resolve())
        backend = "rfdetr"

    def predict_boxes(source: object) -> tuple[np.ndarray, np.ndarray]:
        if backend == "yolo":
            result = model.predict(
                source,
                conf=inference_threshold,
                imgsz=args.tile_size or 960,
                verbose=False,
            )[0]
            if result.boxes is None:
                return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            selected = class_ids == 0
            return (
                result.boxes.xyxy.detach().cpu().numpy()[selected].astype(np.float64),
                result.boxes.conf.detach().cpu().numpy()[selected].astype(np.float64),
            )
        detections = model.predict(
            source, threshold=inference_threshold, include_source_image=False
        )
        selected = _sports_ball_mask(detections)
        return (
            np.asarray(detections.xyxy)[selected].astype(np.float64),
            np.asarray(detections.confidence)[selected].astype(np.float64),
        )

    predictions: list[dict[str, object]] = []
    for image in payload.get("images", []):
        image_path = args.dataset / args.split / str(image["file_name"])
        if args.tile_size is None:
            boxes, confidences = predict_boxes(str(image_path))
        else:
            source = np.asarray(Image.open(image_path).convert("RGB"))
            if backend == "yolo":
                # Ultralytics treats numpy-array inputs as OpenCV-style BGR.
                source = source[:, :, ::-1].copy()
            tiled_boxes: list[np.ndarray] = []
            tiled_confidences: list[np.ndarray] = []
            for tile_x, tile_y in square_tile_origins(
                source.shape[1],
                source.shape[0],
                tile_size=args.tile_size,
                overlap=args.tile_overlap,
            ):
                tile = extract_square_tile(
                    source, (tile_x, tile_y), tile_size=args.tile_size
                )
                tile_boxes, tile_confidences = predict_boxes(tile)
                if len(tile_boxes):
                    tile_boxes[:, (0, 2)] += tile_x
                    tile_boxes[:, (1, 3)] += tile_y
                    tile_boxes[:, (0, 2)] = np.clip(
                        tile_boxes[:, (0, 2)], 0, source.shape[1]
                    )
                    tile_boxes[:, (1, 3)] = np.clip(
                        tile_boxes[:, (1, 3)], 0, source.shape[0]
                    )
                    tiled_boxes.append(tile_boxes)
                    tiled_confidences.append(tile_confidences)
            boxes = (
                np.concatenate(tiled_boxes)
                if tiled_boxes
                else np.empty((0, 4), dtype=np.float64)
            )
            confidences = (
                np.concatenate(tiled_confidences)
                if tiled_confidences
                else np.empty((0,), dtype=np.float64)
            )
            kept = non_maximum_suppression(
                boxes, confidences, iou_threshold=args.nms_iou
            )
            boxes = boxes[kept]
            confidences = confidences[kept]
        for box, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = (float(value) for value in box)
            predictions.append(
                {
                    "image_id": int(image["id"]),
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(confidence),
                }
            )
    report = {
        "schema_version": 1,
        "model": model_name,
        "backend": backend,
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "threshold": args.threshold,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap if args.tile_size is not None else None,
        "nms_iou": args.nms_iou if args.tile_size is not None else None,
        **evaluate_coco_ball_predictions(
            annotations_path,
            (
                prediction
                for prediction in predictions
                if float(prediction["score"]) >= args.threshold
            ),
        ),
        **evaluate_tiny_ball_center_hits(
            annotations_path,
            (
                prediction
                for prediction in predictions
                if float(prediction["score"]) >= args.threshold
            ),
        ),
    }
    if args.score_thresholds:
        report["threshold_sweep"] = evaluate_score_thresholds(
            annotations_path,
            predictions,
            args.score_thresholds,
        )
        report["center_hit_threshold_sweep"] = [
            {
                "threshold": threshold,
                **evaluate_tiny_ball_center_hits(
                    annotations_path,
                    (
                        prediction for prediction in predictions
                        if float(prediction["score"]) >= threshold
                    ),
                ),
            }
            for threshold in sorted(set(args.score_thresholds))
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
