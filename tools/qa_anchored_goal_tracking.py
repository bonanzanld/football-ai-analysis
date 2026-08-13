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

from football_ai.calibration.anchored_goal_tracking import (
    contiguous_goal_windows,
    project_anchored_goal,
    project_anchored_goal_line,
)
from football_ai.calibration.bootstrap.goal_seed import load_goal_seeds
from football_ai.calibration.global_frame_graph import (
    estimate_frame_edge,
    estimate_ground_frame_edge,
    homography_local_scale_ratio,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Volg bevestigde doelstructuren naar terugkerende camerastanden.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", default="8v8")
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--local-radius", type=float, default=2.0)
    parser.add_argument("--local-interval", type=float, default=0.5)
    parser.add_argument("--maximum-scale-ratio", type=float, default=1.12)
    parser.add_argument("--maximum-model-disagreement", type=float, default=12.0)
    parser.add_argument("--ground-only", action="store_true", help="Gebruik bevestigde doelvoeten zonder latpunten.")
    args = parser.parse_args()
    if args.sample_interval <= 0 or args.local_interval <= 0 or args.local_radius < 0:
        parser.error("Intervallen moeten positief zijn en de lokale radius mag niet negatief zijn.")

    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    duration = float(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / fps

    if args.ground_only:
        anchors = [
            (seed.goal_id, seed.time_seconds, (seed.first_ground, seed.second_ground), ())
            for seed in load_goal_seeds(output / f"{prefix}_goal_seeds.json")
        ]
    else:
        anchors = []
        for goal in ("A", "B"):
            data = json.loads((output / f"{prefix}_view_{goal}_3d.json").read_text(encoding="utf-8"))["view"]
            points = {item["landmark_id"]: tuple(item["image_point"]) for item in data["observations"]}
            key = goal.lower()
            anchors.append(
                (
                    goal,
                    float(data["frame_number"]) / fps,
                    (points[f"goal_{key}_rear_bottom"], points[f"goal_{key}_front_bottom"]),
                    (points[f"goal_{key}_rear_top"], points[f"goal_{key}_front_top"]),
                )
            )

    records = []
    try:
        for goal, anchor_time, ground, top in anchors:
            anchor = _read(capture, anchor_time, fps)
            coarse_times = np.arange(0.0, duration, args.sample_interval)
            accepted_coarse = []
            for time_seconds in coarse_times:
                if abs(float(time_seconds) - anchor_time) < args.local_radius + 0.1:
                    continue
                record = _evaluate(
                    capture, fps, anchor, goal, anchor_time, float(time_seconds), ground, top,
                    args.maximum_scale_ratio, args.maximum_model_disagreement, args.ground_only,
                )
                if record is not None:
                    accepted_coarse.append(float(time_seconds))
                    records.append(record | {"source": "coarse_recurrent_view"})
            local_times = {
                round(center + offset, 6)
                for center in accepted_coarse + [anchor_time]
                for offset in np.arange(-args.local_radius, args.local_radius + 1e-6, args.local_interval)
                if 0.0 <= center + offset < duration
            }
            existing = {round(item["time_seconds"], 6) for item in records if item["goal"] == goal}
            for time_seconds in sorted(local_times - existing):
                record = _evaluate(
                    capture, fps, anchor, goal, anchor_time, time_seconds, ground, top,
                    args.maximum_scale_ratio, args.maximum_model_disagreement, args.ground_only,
                )
                if record is not None:
                    records.append(record | {"source": "local_fill"})
    finally:
        capture.release()

    records.sort(key=lambda item: (item["goal"], item["time_seconds"]))
    payload = {
        "schema_version": 1,
        "video_name": video.name,
        "diagnostic_only": True,
        "records": records,
        "summary": {
            goal: {
                "accepted_frames": sum(item["goal"] == goal for item in records),
                "recurrent_view_centres": sum(item["goal"] == goal and item["source"] == "coarse_recurrent_view" for item in records),
                "tracking_windows": [
                    {"start_seconds": start, "end_seconds": end, "sample_count": count}
                    for start, end, count in contiguous_goal_windows(
                        tuple(item["time_seconds"] for item in records if item["goal"] == goal),
                        maximum_gap_seconds=args.local_interval * 1.5,
                    )
                ],
            }
            for goal in ("A", "B")
        },
    }
    path = output / f"{prefix}_anchored_goal_tracking_qa.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Doeltracking-QA: {path}")


def _read(capture: cv2.VideoCapture, time_seconds: float, fps: float) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, round(time_seconds * fps))
    success, frame = capture.read()
    if not success:
        raise ValueError(f"Frame op {time_seconds:.2f}s niet leesbaar.")
    return frame


def _evaluate(capture, fps, anchor, goal, anchor_time, time_seconds, ground, top, maximum_scale, maximum_disagreement, ground_only):
    try:
        target = _read(capture, time_seconds, fps)
        full = estimate_frame_edge("anchor", "target", anchor, target)
        plane = estimate_ground_frame_edge("anchor", "target", anchor, target)
        scale = homography_local_scale_ratio(plane.source_to_target)
        scale_ratio = max(scale, 1.0 / scale)
        result = (
            project_anchored_goal_line(
                ground, full, plane,
                maximum_model_disagreement_px=min(maximum_disagreement, 8.0),
            )
            if ground_only
            else project_anchored_goal(
                ground, top, full, plane, maximum_model_disagreement_px=maximum_disagreement
            )
        )
        if not result.valid or scale_ratio > maximum_scale:
            return None
        return {
            "goal": goal,
            "anchor_time_seconds": anchor_time,
            "frame_number": int(round(time_seconds * fps)),
            "time_seconds": time_seconds,
            "ground_points": result.ground_points,
            "top_points": result.top_points,
            "model_disagreement_px": result.model_disagreement_px,
            "full_frame_inliers": full.inliers,
            "ground_frame_inliers": plane.inliers,
            "scale_ratio": scale_ratio,
        }
    except (ValueError, cv2.error):
        return None


if __name__ == "__main__":
    main()
