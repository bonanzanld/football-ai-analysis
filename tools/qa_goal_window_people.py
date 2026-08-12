from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.classification.goalkeeper_classifier import (
    GoalLineReference,
    goal_line_proximity_score,
    is_on_pitch_side_of_goal_line,
)
from football_ai.detector import FootballDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecteer personen uitsluitend in bevestigde doelvensters.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", default="8v8")
    parser.add_argument("--player-threshold", type=float, default=0.20)
    parser.add_argument("--maximum-distance-ratio", type=float, default=0.12)
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    tracking_path = output / f"{prefix}_anchored_goal_tracking_qa.json"
    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))["records"]

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    diagonal = float(np.hypot(capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    maximum_distance = args.maximum_distance_ratio * diagonal
    detector = FootballDetector(player_threshold=args.player_threshold, ball_threshold=1.0)
    records = []
    try:
        for index, goal in enumerate(tracking, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(goal["frame_number"]))
            success, frame = capture.read()
            if not success:
                continue
            _all, people, _balls = detector.detect(frame)
            line = GoalLineReference(
                str(goal["goal"]), tuple(goal["ground_points"][0]), tuple(goal["ground_points"][1])
            )
            pitch_reference = (frame.shape[1] / 2.0, float(frame.shape[0] - 1))
            candidates = []
            confidences = people.confidence if people.confidence is not None else np.ones(len(people))
            for box, confidence in zip(people.xyxy, confidences):
                footpoint = (float((box[0] + box[2]) / 2.0), float(box[3]))
                if not is_on_pitch_side_of_goal_line(footpoint, line, pitch_reference):
                    continue
                score = goal_line_proximity_score(footpoint, line, maximum_distance)
                if score <= 0.0:
                    continue
                candidates.append({
                    "box": tuple(map(float, box)),
                    "footpoint": footpoint,
                    "confidence": float(confidence),
                    "goal_proximity_score": score,
                })
            candidates.sort(key=lambda item: item["goal_proximity_score"], reverse=True)
            records.append({
                "frame_number": int(goal["frame_number"]),
                "time_seconds": float(goal["time_seconds"]),
                "goal": str(goal["goal"]),
                "detected_people": int(len(people)),
                "candidates": candidates,
            })
            if index % 10 == 0:
                print(f"Doelframes verwerkt: {index}/{len(tracking)}", flush=True)
    finally:
        capture.release()
    payload = {
        "schema_version": 1,
        "video_name": video.name,
        "diagnostic_only": True,
        "maximum_distance_pixels": maximum_distance,
        "records": records,
        "summary": {
            "frames": len(records),
            "frames_with_candidates": sum(bool(item["candidates"]) for item in records),
            "candidate_detections": sum(len(item["candidates"]) for item in records),
        },
    }
    path = output / f"{prefix}_goal_window_people_qa.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Doelvenster-personen-QA: {path}")


if __name__ == "__main__":
    main()
