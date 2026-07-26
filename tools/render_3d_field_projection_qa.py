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
from football_ai.calibration.camera_anchor_runtime import CameraAnchorRuntime
from football_ai.calibration.reference_3d import create_field_reference_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Render een temporele QA-video van de 3D-veldprojectie.")
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--start", type=float, default=900.0)
    parser.add_argument("--duration", type=float, default=110.0)
    parser.add_argument("--fps", type=float, default=5.0, help="Aantal geanalyseerde beelden per seconde.")
    parser.add_argument("--expanded", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0.0:
        raise ValueError("QA-fps moet positief zijn.")

    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    suffix = "_camera_anchors_3d_expanded.json" if args.expanded else "_camera_anchors_3d.json"
    bank = load_camera_anchor_bank(output_dir / f"{prefix}{suffix}")
    reference = create_field_reference_3d(create_detection_profile(args.format))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video kon niet worden geopend: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    anchor_frames = {}
    for anchor in bank.anchors:
        capture.set(cv2.CAP_PROP_POS_FRAMES, anchor.frame_number)
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Ankerframe {anchor.frame_number} kon niet worden gelezen.")
        anchor_frames[anchor.anchor_id] = frame
    runtime = CameraAnchorRuntime(bank, reference, anchor_frames)

    label = "expanded" if args.expanded else "primary"
    output_path = output_dir / f"{prefix}_field_projection_{label}_qa.mp4"
    timeline_path = output_dir / f"{prefix}_field_projection_{label}_qa.json"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"QA-video kon niet worden aangemaakt: {output_path}")

    field_ids = ("corner_a_rear", "corner_b_rear", "corner_b_front", "corner_a_front")
    total = int(np.floor(args.duration * args.fps)) + 1
    records = []
    previous_anchor: str | None = None
    switches = []
    valid_count = 0
    for index in range(total):
        time_seconds = args.start + index / args.fps
        frame_number = int(round(time_seconds * source_fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        if not success:
            break
        result = runtime.project(frame)
        if result.valid and result.projection is not None:
            polygon = np.asarray(
                [result.projection.project(reference.landmark(item).point) for item in field_ids],
                dtype=np.int32,
            )
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], (0, 180, 180), cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, 0.10, frame, 0.90, 0.0)
            cv2.polylines(frame, [polygon], True, (0, 255, 255), 4, cv2.LINE_AA)
            valid_count += 1
        if result.anchor_id is not None and previous_anchor is not None and result.anchor_id != previous_anchor:
            switches.append({"time_seconds": time_seconds, "from": previous_anchor, "to": result.anchor_id})
        if result.anchor_id is not None:
            previous_anchor = result.anchor_id
        status = "VALID" if result.valid else "UNKNOWN"
        color = (0, 220, 0) if result.valid else (0, 165, 255)
        local_text = ""
        if result.local is not None:
            local_text = (
                f" | inliers {result.local.inliers} ({result.local.inlier_ratio:.0%})"
                f" | dekking {min(result.local.anchor_coverage, result.local.frame_coverage):.0%}"
            )
        cv2.rectangle(frame, (0, 0), (width, 74), (18, 18, 18), -1)
        cv2.putText(
            frame,
            f"3D VELD QA | {time_seconds:.1f}s | {status} | {result.anchor_id or '-'}{local_text}",
            (16, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA,
        )
        writer.write(frame)
        records.append({
            "time_seconds": time_seconds,
            "frame_number": frame_number,
            "status": status.lower(),
            "anchor_id": result.anchor_id,
            "reason": result.reason,
        })
        if index and index % max(int(args.fps * 10), 1) == 0:
            print(f"Verwerkt: {index / args.fps:.0f}s / {args.duration:.0f}s")
    capture.release()
    writer.release()
    timeline_path.write_text(
        json.dumps({"schema_version": 1, "fps": args.fps, "samples": records, "anchor_switches": switches}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Geldig: {valid_count}/{len(records)} | UNKNOWN: {len(records) - valid_count}/{len(records)}")
    print(f"Ankerwissels: {len(switches)}")
    print(f"QA-video: {output_path}")
    print(f"QA-tijdlijn: {timeline_path}")


if __name__ == "__main__":
    main()
