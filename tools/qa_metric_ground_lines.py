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
from football_ai.calibration.ground_line_evidence import detect_metric_ground_lines, draw_ground_line_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="QA van lange witte lijnen op het 2D-grondvlak.")
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--minimum-length", type=float, default=3.0)
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    bank = load_camera_anchor_bank(output_dir / f"{prefix}_camera_anchors_3d.json")
    profile = create_detection_profile(args.format)
    capture = cv2.VideoCapture(str(video))
    tiles, records = [], []
    for anchor in bank.anchors:
        capture.set(cv2.CAP_PROP_POS_FRAMES, anchor.frame_number)
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Ankerframe {anchor.frame_number} kon niet worden gelezen.")
        detection = detect_metric_ground_lines(
            frame,
            profile,
            anchor.projection.ground_homography(),
            minimum_length_m=args.minimum_length,
        )
        preview = draw_ground_line_evidence(frame, detection)
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 64), (20, 20, 20), -1)
        cv2.putText(
            preview,
            f"{anchor.anchor_id} | >= {args.minimum_length:g}m | cyaan=lengte | magenta=dwars | {len(detection.lines)} lijnen",
            (14, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )
        tiles.append(cv2.resize(preview, (640, 360)))
        records.append({
            "anchor_id": anchor.anchor_id,
            "frame_number": anchor.frame_number,
            "raw_candidates": detection.raw_candidates,
            "merged_candidates": detection.merged_candidates,
            "rejected_short": detection.rejected_short,
            "rejected_direction": detection.rejected_direction,
            "rejected_implausible": detection.rejected_implausible,
            "lines": [item.to_dict() for item in detection.lines],
        })
        print(
            f"{anchor.anchor_id}: {len(detection.lines)} >= {args.minimum_length:g}m | "
            f"rauw {detection.raw_candidates} | kort verworpen {detection.rejected_short} | "
            f"richting verworpen {detection.rejected_direction} | "
            f"onwaarschijnlijk verworpen {detection.rejected_implausible}"
        )
    capture.release()
    preview_path = output_dir / f"{prefix}_metric_ground_lines_qa.jpg"
    report_path = output_dir / f"{prefix}_metric_ground_lines_qa.json"
    if not cv2.imwrite(str(preview_path), np.vstack(tiles)):
        raise RuntimeError(f"QA-preview kon niet worden opgeslagen: {preview_path}")
    report_path.write_text(json.dumps({"schema_version": 1, "minimum_length_m": args.minimum_length, "anchors": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"QA-preview: {preview_path}")
    print(f"QA-rapport: {report_path}")


if __name__ == "__main__":
    main()
