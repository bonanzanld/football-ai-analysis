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

from football_ai.classification.goal_window_candidates import (
    GoalWindowPerson,
    appearance_stability_classification,
    evaluate_goal_person_path,
    select_continuous_goal_person,
)
from football_ai.classification.color_features import extract_shirt_feature


def main() -> None:
    parser = argparse.ArgumentParser(description="Bouw conservatieve keeper-reviewtrajecten uit doelvensters.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", default="8v8")
    parser.add_argument("--maximum-gap", type=float, default=0.75)
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    source = json.loads((output / f"{prefix}_goal_window_people_qa.json").read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    diagonal = float(np.hypot(capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    windows = []
    for goal in ("A", "B"):
        records = sorted((item for item in source["records"] if item["goal"] == goal), key=lambda item: item["time_seconds"])
        for group in _groups(records, args.maximum_gap):
            frames = tuple(
                tuple(
                    GoalWindowPerson(
                        int(record["frame_number"]), tuple(candidate["footpoint"]),
                        float(candidate["goal_proximity_score"]), tuple(candidate["box"]),
                    )
                    for candidate in record["candidates"]
                )
                for record in group
            )
            path = select_continuous_goal_person(frames, frame_diagonal=diagonal)
            quality = evaluate_goal_person_path(path, frame_diagonal=diagonal)
            appearance_distances = _appearance_distances(capture, path)
            appearance = appearance_stability_classification(appearance_distances)
            classification = quality.classification
            if classification == "consistent_review_candidate" and appearance == "unstable":
                classification = "ambiguous"
            windows.append({
                "goal": goal,
                "start_seconds": float(group[0]["time_seconds"]),
                "end_seconds": float(group[-1]["time_seconds"]),
                "quality": {
                    "sample_count": quality.sample_count,
                    "mean_goal_proximity": quality.mean_goal_proximity,
                    "maximum_step_ratio": quality.maximum_step_ratio,
                    "mean_step_ratio": quality.mean_step_ratio,
                    "classification": classification,
                    "motion_classification": quality.classification,
                    "appearance_classification": appearance,
                    "appearance_median_distance": float(np.median(appearance_distances)) if appearance_distances else None,
                },
                "path": [
                    {"frame_number": item.frame_number, "footpoint": item.footpoint,
                     "goal_proximity_score": item.goal_proximity_score, "box": item.box}
                    for item in path
                ],
            })
    capture.release()
    payload = {
        "schema_version": 1,
        "video_name": video.name,
        "diagnostic_only": True,
        "role_assignment": False,
        "windows": windows,
        "summary": {
            "windows": len(windows),
            "consistent_review_candidates": sum(item["quality"]["classification"] == "consistent_review_candidate" for item in windows),
            "ambiguous": sum(item["quality"]["classification"] == "ambiguous" for item in windows),
            "insufficient_evidence": sum(item["quality"]["classification"] == "insufficient_evidence" for item in windows),
        },
    }
    path = output / f"{prefix}_goalkeeper_review_candidates.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Keeper-reviewtrajecten: {path}")


def _groups(records: list[dict], maximum_gap: float) -> tuple[tuple[dict, ...], ...]:
    if not records:
        return ()
    groups, current = [], [records[0]]
    for item in records[1:]:
        if float(item["time_seconds"]) - float(current[-1]["time_seconds"]) > maximum_gap:
            groups.append(tuple(current))
            current = []
        current.append(item)
    groups.append(tuple(current))
    return tuple(groups)


def _appearance_distances(
    capture: cv2.VideoCapture,
    path: tuple[GoalWindowPerson, ...],
) -> tuple[float, ...]:
    features = []
    for item in path:
        capture.set(cv2.CAP_PROP_POS_FRAMES, item.frame_number)
        success, frame = capture.read()
        if not success:
            continue
        feature = extract_shirt_feature(frame, np.asarray(item.box, dtype=np.float64))
        if feature is None:
            continue
        norm = float(np.linalg.norm(feature))
        if norm > 0:
            features.append(feature / norm)
    if not features:
        return ()
    median = np.median(np.stack(features), axis=0)
    median /= max(float(np.linalg.norm(median)), 1e-9)
    return tuple(float(np.linalg.norm(item - median)) for item in features)


if __name__ == "__main__":
    main()
