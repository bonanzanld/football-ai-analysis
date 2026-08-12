from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.global_ground_registration import load_global_ground_registration


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    mapped = (matrix @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--rejected", action="store_true")
    args = parser.parse_args()
    video = args.video if args.video.is_absolute() else PROJECT_ROOT / args.video
    registration_path = args.registration if args.registration.is_absolute() else PROJECT_ROOT / args.registration
    registration = load_global_ground_registration(registration_path)
    profile = create_detection_profile(registration.match_format)
    selected = [registration.frames[int(index)] for index in np.linspace(0, len(registration.frames) - 1, args.samples).round()]
    capture = cv2.VideoCapture(str(video))
    tile_width, tile_height = 640, 360
    sheet = np.full((tile_height, tile_width * len(selected), 3), 24, np.uint8)
    field = np.asarray(((0, 0), (profile.pitch_length_m, 0), (profile.pitch_length_m, profile.pitch_width_m), (0, profile.pitch_width_m)), dtype=float)
    for index, item in enumerate(selected):
        capture.set(cv2.CAP_PROP_POS_FRAMES, item.frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        polygon = np.round(project(item.ground_to_image, field)).astype(np.int32)
        overlay = frame.copy(); cv2.fillPoly(overlay, [polygon], (40, 180, 60)); frame = cv2.addWeighted(overlay, .14, frame, .86, 0)
        cv2.polylines(frame, [polygon], True, (0, 255, 255), 5, cv2.LINE_AA)
        frame = cv2.resize(frame, (tile_width, tile_height))
        cv2.rectangle(frame, (0, 0), (tile_width, 52), (18, 18, 18), -1)
        label = "AFGEKEURD" if args.rejected else "FRAMEGRAPH DIAGNOSE"
        color = (0, 0, 255) if args.rejected else (0, 180, 255)
        cv2.putText(frame, f"{label} | {item.time_seconds:.1f}s", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
        sheet[:, index * tile_width:(index + 1) * tile_width] = frame
    capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    print(f"Diagnostische tijdlijn: {args.output}")


if __name__ == "__main__":
    main()
