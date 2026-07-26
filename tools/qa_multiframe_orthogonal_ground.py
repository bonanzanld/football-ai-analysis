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
from football_ai.calibration.ground_line_evidence import detect_metric_ground_lines
from football_ai.calibration.local_anchor_projection import estimate_local_anchor_projection
from football_ai.calibration.orthogonal_ground_orientation import (
    ImageLineObservation,
    estimate_orthogonal_ground_orientation,
    transform_line_observation,
)
from football_ai.calibration.reference_3d import create_field_reference_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Verzamel lange witte lijnen uit meerdere frames in één referentiebeeld.")
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--window", type=float, default=8.0, help="Seconden voor en na ieder primair anker.")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--minimum-length", type=float, default=3.0)
    args = parser.parse_args()

    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    bank = load_camera_anchor_bank(output_dir / f"{prefix}_camera_anchors_3d.json")
    primaries = tuple(item for item in bank.anchors if item.anchor_type == "primary")
    profile = create_detection_profile(args.format)
    reference = create_field_reference_3d(profile)
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reports, previews = [], []
    for anchor in primaries:
        capture.set(cv2.CAP_PROP_POS_FRAMES, anchor.frame_number)
        success, anchor_frame = capture.read()
        if not success:
            raise RuntimeError(f"Ankerframe {anchor.frame_number} kon niet worden gelezen.")
        observations = []
        accepted_frames = 0
        time_seconds = anchor.time_seconds - args.window
        while time_seconds <= anchor.time_seconds + args.window + 1e-6:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(time_seconds * fps)))
            success, frame = capture.read()
            if success:
                local = estimate_local_anchor_projection(
                    anchor_frame, frame, anchor.projection, reference
                )
                if local.valid and local.projection is not None and local.image_transform is not None:
                    detection = detect_metric_ground_lines(
                        frame,
                        profile,
                        local.projection.ground_homography(),
                        minimum_length_m=args.minimum_length,
                    )
                    frame_to_anchor = np.linalg.inv(local.image_transform)
                    for line in detection.lines:
                        observations.append(
                            transform_line_observation(
                                ImageLineObservation(
                                    line.family,
                                    line.image_start,
                                    line.image_end,
                                    max(line.metric_length * line.confidence, 0.1),
                                    float(np.mean((line.ground_start, line.ground_end), axis=0)[1 if line.family.value == "longitudinal" else 0]),
                                ),
                                frame_to_anchor,
                            )
                        )
                    accepted_frames += 1
            time_seconds += args.interval
        result = None
        failure = None
        try:
            result = estimate_orthogonal_ground_orientation(
                tuple(observations), (anchor_frame.shape[1], anchor_frame.shape[0])
            )
        except ValueError as error:
            failure = str(error)
        preview = anchor_frame.copy()
        colors = {"longitudinal": (255, 255, 0), "transverse": (255, 0, 255)}
        for item in observations:
            cv2.line(
                preview,
                tuple(np.round(item.start).astype(int)),
                tuple(np.round(item.end).astype(int)),
                colors[item.family.value],
                2,
                cv2.LINE_AA,
            )
        status = "OPGELOST" if result is not None else f"ONVOLDOENDE: {failure}"
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 68), (20, 20, 20), -1)
        cv2.putText(preview, f"{anchor.anchor_id} | frames {accepted_frames} | lijnen {len(observations)} | {status}", (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        previews.append(cv2.resize(preview, (640, 360)))
        reports.append({
            "anchor_id": anchor.anchor_id,
            "accepted_frames": accepted_frames,
            "line_observations": len(observations),
            "solved": result is not None,
            "failure_reason": failure,
            "orientation": result.to_dict() if result is not None else None,
        })
        print(f"{anchor.anchor_id}: frames {accepted_frames} | lijnen {len(observations)} | {status}")
    capture.release()
    preview_path = output_dir / f"{prefix}_multiframe_orthogonal_ground_qa.jpg"
    report_path = output_dir / f"{prefix}_multiframe_orthogonal_ground_qa.json"
    cv2.imwrite(str(preview_path), np.vstack(previews))
    report_path.write_text(json.dumps({"schema_version": 1, "minimum_length_m": args.minimum_length, "anchors": reports}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"QA-preview: {preview_path}")
    print(f"QA-rapport: {report_path}")


if __name__ == "__main__":
    main()
