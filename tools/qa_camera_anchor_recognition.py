from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from football_ai.calibration.camera_anchor_bank_3d import load_camera_anchor_bank
from football_ai.calibration.camera_anchor_recognition import CameraAnchorRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Test conservatieve herkenning van 3D-camera-ankers.")
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4", help="Bestandsnaam in videos-map.")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--start", type=float, default=900.0, help="Starttijd in seconden.")
    parser.add_argument("--duration", type=float, default=110.0, help="Testduur in seconden.")
    parser.add_argument("--interval", type=float, default=2.0, help="Afstand tussen testframes in seconden.")
    args = parser.parse_args()

    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    bank = load_camera_anchor_bank(output_dir / f"{prefix}_camera_anchors_3d.json")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video kon niet worden geopend: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    anchor_frames: dict[str, np.ndarray] = {}
    for anchor in bank.anchors:
        capture.set(cv2.CAP_PROP_POS_FRAMES, anchor.frame_number)
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Ankerframe {anchor.frame_number} kon niet worden gelezen.")
        anchor_frames[anchor.anchor_id] = frame
    recognizer = CameraAnchorRecognizer.from_frames(anchor_frames)

    records = []
    tiles = []
    time_seconds = args.start
    while time_seconds <= args.start + args.duration + 1e-6:
        frame_number = int(round(time_seconds * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        if not success:
            break
        recognition = recognizer.recognize(frame)
        records.append({
            "time_seconds": time_seconds,
            "frame_number": frame_number,
            "status": recognition.status.value,
            "anchor_id": recognition.anchor_id,
            "reason": recognition.reason,
            "scores": [asdict(item) for item in recognition.scores],
        })
        color = (0, 200, 0) if recognition.anchor_id else (0, 165, 255)
        label = f"{time_seconds:.1f}s | {recognition.anchor_id or recognition.status.value.upper()}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (20, 20, 20), -1)
        cv2.putText(frame, label, (15, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        tiles.append(cv2.resize(frame, (320, 180)))
        time_seconds += args.interval
    capture.release()

    report_path = output_dir / f"{prefix}_anchor_recognition_qa.json"
    report_path.write_text(json.dumps({"schema_version": 1, "samples": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    if tiles:
        columns = 4
        rows = [np.hstack(tiles[index:index + columns]) for index in range(0, len(tiles), columns) if len(tiles[index:index + columns]) == columns]
        if rows:
            preview_path = output_dir / f"{prefix}_anchor_recognition_qa.jpg"
            cv2.imwrite(str(preview_path), np.vstack(rows))
            print(f"QA-overzicht: {preview_path}")
    counts: dict[str, int] = {}
    for item in records:
        key = item["anchor_id"] or item["status"]
        counts[key] = counts.get(key, 0) + 1
    print(f"Herkenningsrapport: {report_path}")
    print("Resultaten: " + " | ".join(f"{key} {value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()
