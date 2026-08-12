from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detection.active_ball_classifier import (
    candidate_patch_features,
    candidate_temporal_features,
)


def _load_examples(dataset: Path) -> tuple[str, list[dict[str, object]]]:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    source = str(payload.get("source_video", ""))
    if payload.get("schema_version") != 1 or not source:
        raise ValueError(f"Unsupported dataset report: {dataset}")
    examples = list(payload.get("examples", []))
    return source, examples


def _example_stats(examples: list[dict[str, object]]) -> dict[str, object]:
    stats: dict[str, object] = {}
    for label in ("positive", "negative"):
        selected = [item for item in examples if item["label"] == label]
        widths = [float(item["box"][2]) - float(item["box"][0]) for item in selected]
        heights = [float(item["box"][3]) - float(item["box"][1]) for item in selected]
        confidences = [float(item["confidence"]) for item in selected]
        stats[label] = {
            "examples": len(selected),
            "mean_confidence": float(np.mean(confidences)) if selected else 0.0,
            "mean_box_width": float(np.mean(widths)) if selected else 0.0,
            "mean_box_height": float(np.mean(heights)) if selected else 0.0,
        }
    return stats


def _extract_source_features(
    source: str,
    examples: list[dict[str, object]],
    *,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray]:
    by_frame: dict[int, list[dict[str, object]]] = {}
    for example in examples:
        by_frame.setdefault(int(example["frame_number"]), []).append(example)
    capture = cv2.VideoCapture(str((PROJECT_ROOT / source).resolve()))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")
    features: list[np.ndarray] = []
    labels: list[int] = []
    try:
        for frame_number in sorted(by_frame):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read {source} frame {frame_number}")
            frame_height, frame_width = image.shape[:2]
            for example in by_frame[frame_number]:
                if example.get("label") not in {"positive", "negative"}:
                    continue
                box = tuple(float(value) for value in example["box"])
                width = max(1.0, box[2] - box[0])
                height = max(1.0, box[3] - box[1])
                visual = candidate_patch_features(image, box)
                metadata = np.asarray(
                    [
                        float(example["confidence"]),
                        np.log1p(width * height),
                        width / height,
                    ],
                    dtype=np.float32,
                )
                parts = [visual, metadata]
                if feature_set == "patch-temporal":
                    parts.append(
                        candidate_temporal_features(
                            example,
                            by_frame,
                            frame_width=frame_width,
                            frame_height=frame_height,
                        )
                    )
                features.append(np.concatenate(parts))
                labels.append(1 if example["label"] == "positive" else 0)
    finally:
        capture.release()
    return np.asarray(features), np.asarray(labels, dtype=np.int8)


def _model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=0,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and clip-holdout evaluate an active-ball patch classifier."
    )
    parser.add_argument("--datasets", nargs="+", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument(
        "--feature-set",
        choices=("patch", "patch-temporal"),
        default="patch",
        help="Add label-free nearby-frame continuity features to the patch baseline.",
    )
    args = parser.parse_args()

    sources: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    source_stats: dict[str, dict[str, object]] = {}
    for dataset in args.datasets:
        source, examples = _load_examples(dataset)
        if source in sources:
            raise ValueError(f"Duplicate source video: {source}")
        sources[source] = _extract_source_features(
            source, examples, feature_set=args.feature_set
        )
        source_stats[source] = _example_stats(examples)
    if len(sources) < 3:
        raise ValueError("Clip-holdout evaluation requires at least three videos")

    folds = []
    for held_out, (test_x, test_y) in sources.items():
        train_parts = [value for source, value in sources.items() if source != held_out]
        train_x = np.concatenate([part[0] for part in train_parts])
        train_y = np.concatenate([part[1] for part in train_parts])
        model = _model()
        model.fit(train_x, train_y)
        predicted = model.predict(test_x)
        true_positives = int(np.sum((test_y == 1) & (predicted == 1)))
        false_positives = int(np.sum((test_y == 0) & (predicted == 1)))
        false_negatives = int(np.sum((test_y == 1) & (predicted == 0)))
        true_negatives = int(np.sum((test_y == 0) & (predicted == 0)))
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y,
            predicted,
            average="binary",
            zero_division=0,
        )
        folds.append(
            {
                "held_out_source": held_out,
                "examples": int(len(test_y)),
                "positive_examples": int(test_y.sum()),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_negatives": true_negatives,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    all_x = np.concatenate([part[0] for part in sources.values()])
    all_y = np.concatenate([part[1] for part in sources.values()])
    final_model = _model()
    final_model.fit(all_x, all_y)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_bytes(pickle.dumps(final_model))
    report = {
        "schema_version": 1,
        "model": "scaled_logistic_regression_on_candidate_patches",
        "feature_set": args.feature_set,
        "source_videos": list(sources),
        "source_stats": source_stats,
        "examples": int(len(all_y)),
        "positive_examples": int(all_y.sum()),
        "negative_examples": int(len(all_y) - all_y.sum()),
        "clip_holdout_folds": folds,
        "mean_precision": float(np.mean([fold["precision"] for fold in folds])),
        "mean_recall": float(np.mean([fold["recall"] for fold in folds])),
        "mean_f1": float(np.mean([fold["f1"] for fold in folds])),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
