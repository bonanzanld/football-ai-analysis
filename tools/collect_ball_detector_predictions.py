from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detection.ball_evaluation import load_ball_annotations
from football_ai.detection.ball_tracking import (
    candidates_from_detections,
    exclude_candidates_inside_people,
)
from football_ai.detector import FootballDetector


def _serialize(frame_number: int, stage: str, candidates: object) -> list[dict[str, object]]:
    return [
        {
            "frame_number": frame_number,
            "box": list(candidate.box),
            "confidence": candidate.confidence,
            "stage": stage,
        }
        for candidate in candidates
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect raw and person-filtered ball candidates on annotated frames."
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest, annotations = load_ball_annotations(args.annotations)
    source_video = manifest.get("source_video")
    if not source_video:
        parser.error("Annotation manifest has no source_video")
    video_path = Path(str(source_video))
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    detector = FootballDetector(player_threshold=0.20, ball_threshold=args.threshold)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    predictions: list[dict[str, object]] = []
    try:
        for annotation in annotations:
            capture.set(cv2.CAP_PROP_POS_FRAMES, annotation.frame_number)
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Cannot read frame {annotation.frame_number}")
            _, people, detections = detector.detect(frame)
            raw = candidates_from_detections(detections)
            filtered = exclude_candidates_inside_people(raw, people.xyxy)
            predictions.extend(_serialize(annotation.frame_number, "raw_detector", raw))
            predictions.extend(
                _serialize(annotation.frame_number, "person_filtered", filtered)
            )
    finally:
        capture.release()

    output = args.output or args.annotations.with_name("detector_predictions.json")
    payload = {
        "schema_version": 1,
        "source_video": str(source_video),
        "threshold": args.threshold,
        "stages": ["raw_detector", "person_filtered"],
        "predictions": predictions,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Collected {len(predictions)} staged candidates: {output}")


if __name__ == "__main__":
    main()
