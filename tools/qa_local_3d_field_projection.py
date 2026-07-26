from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.camera_anchor_bank_3d import load_camera_anchor_bank
from football_ai.calibration.camera_anchor_recognition import CameraAnchorRecognizer
from football_ai.calibration.local_anchor_projection import estimate_local_anchor_projection
from football_ai.calibration.reference_3d import create_field_reference_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="QA voor directe lokale 3D-veldprojecties vanaf vaste ankers.")
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--start", type=float, default=900.0)
    parser.add_argument("--duration", type=float, default=110.0)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--expanded", action="store_true", help="Gebruik de uitgebreide ankerbank met tussenankers.")
    args = parser.parse_args()

    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    suffix = "_camera_anchors_3d_expanded.json" if args.expanded else "_camera_anchors_3d.json"
    bank = load_camera_anchor_bank(output_dir / f"{prefix}{suffix}")
    reference = create_field_reference_3d(create_detection_profile(args.format))
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    anchor_frames = {}
    for anchor in bank.anchors:
        capture.set(cv2.CAP_PROP_POS_FRAMES, anchor.frame_number)
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Ankerframe {anchor.frame_number} kon niet worden gelezen.")
        anchor_frames[anchor.anchor_id] = frame
    recognizer = CameraAnchorRecognizer.from_frames(anchor_frames)
    anchor_by_id = {item.anchor_id: item for item in bank.anchors}
    records, tiles = [], []
    time_seconds = args.start
    while time_seconds <= args.start + args.duration + 1e-6:
        frame_number = int(round(time_seconds * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        if not success:
            break
        recognition = recognizer.recognize(frame)
        result = None
        if recognition.anchor_id is not None:
            anchor = anchor_by_id[recognition.anchor_id]
            result = estimate_local_anchor_projection(
                anchor_frames[recognition.anchor_id], frame, anchor.projection, reference
            )
        valid = result is not None and result.valid and result.projection is not None
        if valid:
            ids = ("corner_a_rear", "corner_b_rear", "corner_b_front", "corner_a_front")
            polygon = np.asarray([result.projection.project(reference.landmark(item).point) for item in ids], dtype=np.int32)
            cv2.polylines(frame, [polygon], True, (0, 255, 255), 4, cv2.LINE_AA)
        status = "VALID" if valid else "UNKNOWN"
        anchor_id = recognition.anchor_id if valid else None
        reason = result.reason if result is not None else recognition.reason
        color = (0, 220, 0) if valid else (0, 165, 255)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (20, 20, 20), -1)
        cv2.putText(frame, f"{time_seconds:.1f}s | {status} | {anchor_id or '-'}", (14, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
        records.append({
            "time_seconds": time_seconds,
            "frame_number": frame_number,
            "status": status.lower(),
            "anchor_id": anchor_id,
            "reason": reason,
            "local": None if result is None else {
                "good_matches": result.good_matches,
                "inliers": result.inliers,
                "inlier_ratio": result.inlier_ratio,
                "anchor_coverage": result.anchor_coverage,
                "frame_coverage": result.frame_coverage,
            },
        })
        tiles.append(cv2.resize(frame, (320, 180)))
        time_seconds += args.interval
    capture.release()
    report = output_dir / f"{prefix}_local_3d_projection_qa.json"
    report.write_text(json.dumps({"schema_version": 1, "samples": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = 4
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    overview = np.vstack([np.hstack(tiles[index:index + columns]) for index in range(0, len(tiles), columns)])
    preview = output_dir / f"{prefix}_local_3d_projection_qa.jpg"
    cv2.imwrite(str(preview), overview)
    valid_count = sum(item["status"] == "valid" for item in records)
    print(f"Geldig: {valid_count}/{len(records)} | UNKNOWN: {len(records) - valid_count}/{len(records)}")
    print(f"QA-overzicht: {preview}")
    print(f"QA-rapport: {report}")


if __name__ == "__main__":
    main()
