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

from football_ai.calibration.global_frame_graph import (
    estimate_frame_edge,
    estimate_ground_frame_edge,
)


def _read(capture: cv2.VideoCapture, frame_number: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Frame {frame_number} kon niet worden gelezen.")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Breng twee apart aangeklikte 8v8-hoeken naar één ankerframe."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--rear", type=Path, required=True)
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--target-time", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    observations = {
        "rear": json.loads(args.rear.read_text(encoding="utf-8")),
        "front": json.loads(args.front.read_text(encoding="utf-8")),
    }
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(args.video)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    target_frame_number = int(round(args.target_time * fps))
    target = _read(capture, target_frame_number)
    mapped = {}
    diagnostics = {}
    try:
        for role, observation in observations.items():
            source_number = int(observation["frame_number"])
            source = _read(capture, source_number)
            if source_number == target_frame_number:
                matrix = np.eye(3, dtype=np.float64)
                diagnostics[role] = {
                    "matches": 0, "inliers": 0, "inlier_ratio": 1.0,
                    "source_coverage": 1.0, "target_coverage": 1.0,
                    "median_error_px": 0.0, "estimator": "identity",
                }
            else:
                estimator = "ground"
                try:
                    edge = estimate_ground_frame_edge(role, "target", source, target)
                except ValueError:
                    estimator = "full_frame"
                    edge = estimate_frame_edge(role, "target", source, target)
                matrix = edge.source_to_target
                diagnostics[role] = {
                    "matches": edge.matches, "inliers": edge.inliers,
                    "inlier_ratio": edge.inlier_ratio,
                    "source_coverage": edge.source_coverage,
                    "target_coverage": edge.target_coverage,
                    "median_error_px": edge.median_error_px,
                    "estimator": estimator,
                }
            point = np.asarray(observation["point"], dtype=np.float32).reshape(1, 1, 2)
            mapped[role] = cv2.perspectiveTransform(point, matrix).reshape(2).tolist()
    finally:
        capture.release()
    result = {
        "schema_version": 2,
        "video_name": args.video.name,
        "frame_number": target_frame_number,
        "time_seconds": args.target_time,
        "role": "8v8_right_end_line_corners",
        "field_x_m": 64.0,
        "rear_corner": mapped["rear"],
        "front_corner": mapped["front"],
        "source_observations": observations,
        "tracking_diagnostics": diagnostics,
        "provenance": "human_reviewed_split_frames_tracked_to_anchor",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.preview is not None:
        for role, color in (("rear", (0, 140, 255)), ("front", (255, 0, 255))):
            point = tuple(np.rint(mapped[role]).astype(int))
            cv2.circle(target, point, 12, color, -1, cv2.LINE_AA)
            cv2.putText(target, role, (point[0] + 14, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2, cv2.LINE_AA)
        cv2.line(
            target, tuple(np.rint(mapped["rear"]).astype(int)),
            tuple(np.rint(mapped["front"]).astype(int)), (0, 255, 255), 5, cv2.LINE_AA,
        )
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), target)
    print(f"Gecombineerde hoeken: {args.output}")
    for role, item in diagnostics.items():
        print(
            f"{role}: {item['estimator']} | {item['inliers']} inliers | "
            f"{item['inlier_ratio']:.0%} | dekking "
            f"{min(item['source_coverage'], item['target_coverage']):.1%} | "
            f"mediaan {item['median_error_px']:.2f}px"
        )


if __name__ == "__main__":
    main()
