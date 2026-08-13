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
    parser.add_argument("--maximum-windows-per-goal", type=int, default=None)
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    tracking_path = output / f"{prefix}_anchored_goal_tracking_qa.json"
    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))["records"]
    if args.maximum_windows_per_goal is not None:
        if args.maximum_windows_per_goal < 1:
            parser.error("Maximum aantal vensters moet positief zijn.")
        tracking = list(_select_spread_windows(tracking, args.maximum_windows_per_goal))

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
                    "goal_relative_position": _goal_relative_position(footpoint, line),
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


def _goal_relative_position(point, line):
    first = np.asarray(line.first_post, dtype=np.float64)
    second = np.asarray(line.second_post, dtype=np.float64)
    vector = second - first
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return (0.0, 0.0)
    direction = vector / length
    offset = np.asarray(point, dtype=np.float64) - first
    along = float(np.dot(offset, direction) / length)
    perpendicular = float((direction[0] * offset[1] - direction[1] * offset[0]) / length)
    return (along, perpendicular)


def _select_spread_windows(records, maximum_per_goal, maximum_gap=0.75):
    selected = []
    for goal in ("A", "B"):
        items = sorted(
            (item for item in records if item["goal"] == goal),
            key=lambda item: float(item["time_seconds"]),
        )
        windows = []
        for item in items:
            if not windows or float(item["time_seconds"]) - float(windows[-1][-1]["time_seconds"]) > maximum_gap:
                windows.append([])
            windows[-1].append(item)
        eligible = [window for window in windows if len(window) >= 3]
        if len(eligible) > maximum_per_goal:
            indices = np.linspace(0, len(eligible) - 1, maximum_per_goal).round().astype(int)
            eligible = [eligible[index] for index in indices]
        selected.extend(item for window in eligible for item in window)
    return tuple(sorted(selected, key=lambda item: (item["goal"], item["time_seconds"])))


if __name__ == "__main__":
    main()
