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
from football_ai.calibration.ground_circle_evidence import (
    detect_metric_center_circle,
    project_ground_circle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="QA voor de metrische 11v11-middencirkel.")
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    expanded_bank = output_dir / f"{prefix}_camera_anchors_3d_expanded.json"
    bank_path = expanded_bank if expanded_bank.exists() else output_dir / f"{prefix}_camera_anchors_3d.json"
    bank = load_camera_anchor_bank(bank_path)
    profile = create_detection_profile(args.format)
    capture = cv2.VideoCapture(str(video))
    results, previews = {}, []
    ordered_anchors = sorted(bank.anchors, key=lambda item: item.time_seconds)
    for anchor in ordered_anchors:
        capture.set(cv2.CAP_PROP_POS_FRAMES, anchor.frame_number)
        success, frame = capture.read()
        if not success:
            continue
        homography = anchor.projection.ground_homography()
        evidence = detect_metric_center_circle(frame, profile, homography)
        preview = frame.copy()
        if evidence is not None:
            polygon = project_ground_circle(evidence, homography)
            finite = np.all(np.isfinite(polygon), axis=1)
            if np.count_nonzero(finite) >= 3:
                cv2.polylines(preview, [np.round(polygon[finite]).astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)
            label = (
                f"CIRKELBOOG | boog {evidence.radial_support:.0%} | "
                f"dekking {evidence.angular_coverage:.0%} | middenlijn {evidence.halfway_line_support:.0%}"
            )
            results[anchor.anchor_id] = evidence.to_dict()
            color = (0, 255, 255)
        else:
            label = "GEEN BETROUWBARE CIRKELBOOG"
            results[anchor.anchor_id] = None
            color = (0, 80, 255)
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 58), (20, 20, 20), -1)
        role = "DOELSTAND" if anchor.anchor_type == "primary" else "TUSSEN-/MIDDENSTAND"
        cv2.putText(
            preview,
            f"{anchor.anchor_id} | {role} | {anchor.time_seconds:.1f}s | {label}",
            (14, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        previews.append(preview)
    capture.release()
    if previews:
        height = min(item.shape[0] for item in previews)
        previews = [cv2.resize(item, (round(item.shape[1] * height / item.shape[0]), height)) for item in previews]
        preview_path = output_dir / f"{prefix}_metric_center_circle_qa.jpg"
        cv2.imwrite(str(preview_path), np.hstack(previews))
        print(f"QA-preview: {preview_path}")
    report_path = output_dir / f"{prefix}_metric_center_circle_qa.json"
    report_path.write_text(json.dumps({"schema_version": 1, "anchors": results}, indent=2), encoding="utf-8")
    for anchor_id, evidence in results.items():
        print(f"{anchor_id}: {'GEVONDEN' if evidence is not None else 'NIET GEVONDEN'}")
    print(f"QA-rapport: {report_path}")


if __name__ == "__main__":
    main()
