from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detection.ball_evaluation import (
    evaluate_ball_predictions,
    load_ball_annotations,
    load_ball_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure ball precision/recall on manually reviewed frames."
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--center-distance", type=float, default=20.0)
    parser.add_argument("--minimum-iou", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    _, annotations = load_ball_annotations(args.annotations)
    if not annotations:
        parser.error("The annotation manifest is empty")
    if all(item.review_status != "human_reviewed" for item in annotations):
        parser.error("Human-review at least one frame before evaluating predictions")
    predictions = load_ball_predictions(args.predictions)
    stages = sorted({item.stage for item in predictions}) or ["tracker"]
    stage_reports = {
        stage: evaluate_ball_predictions(
            annotations,
            (item for item in predictions if item.stage == stage),
            maximum_center_distance=args.center_distance,
            minimum_iou=args.minimum_iou,
        )
        for stage in stages
    }
    report = {
        "schema_version": 1,
        "annotations": str(args.annotations),
        "predictions": str(args.predictions),
        "maximum_center_distance": args.center_distance,
        "minimum_iou": args.minimum_iou,
        "stages": stage_reports,
    }
    output = args.output or args.annotations.with_name("evaluation.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for stage, metrics in stage_reports.items():
        print(
            f"{stage}: reviewed={metrics['reviewed_frames']} | "
            f"precision={metrics['precision']:.3f} | "
            f"recall={metrics['recall']:.3f} | F1={metrics['f1']:.3f}"
        )
    print(f"Evaluation report: {output}")


if __name__ == "__main__":
    main()
