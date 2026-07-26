from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.camera_anchor_bank_3d import (
    CameraAnchor3D,
    CameraAnchorBank3D,
    load_camera_anchor_bank,
    save_camera_anchor_bank,
)
from football_ai.calibration.camera_anchor_recognition import CameraAnchorRecognizer
from football_ai.calibration.local_anchor_projection import estimate_local_anchor_projection
from football_ai.calibration.reference_3d import create_field_reference_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Promoveer gecontroleerde lokale projecties tot zelfstandige tussenankers.")
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--times", default="932,946,970,1010", help="Kommagetallen in seconden, gescheiden door komma's.")
    args = parser.parse_args()
    times = tuple(float(item.strip()) for item in args.times.split(",") if item.strip())
    if not times:
        raise ValueError("Geef minimaal één tijdstip op.")

    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    source_path = output_dir / f"{prefix}_camera_anchors_3d.json"
    bank = load_camera_anchor_bank(source_path)
    primaries = tuple(item for item in bank.anchors if item.anchor_type == "primary")
    reference = create_field_reference_3d(create_detection_profile(args.format))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video kon niet worden geopend: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    primary_frames = {}
    for anchor in primaries:
        capture.set(cv2.CAP_PROP_POS_FRAMES, anchor.frame_number)
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Primair ankerframe {anchor.frame_number} kon niet worden gelezen.")
        primary_frames[anchor.anchor_id] = frame
    recognizer = CameraAnchorRecognizer.from_frames(primary_frames)
    by_id = {item.anchor_id: item for item in primaries}
    intermediates = []
    for time_seconds in times:
        frame_number = int(round(time_seconds * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        if not success:
            print(f"OVERGESLAGEN {time_seconds:.1f}s: frame kon niet worden gelezen.")
            continue
        recognition = recognizer.recognize(frame)
        if recognition.anchor_id is None:
            print(f"OVERGESLAGEN {time_seconds:.1f}s: {recognition.reason}")
            continue
        parent = by_id[recognition.anchor_id]
        local = estimate_local_anchor_projection(primary_frames[parent.anchor_id], frame, parent.projection, reference)
        if not local.valid or local.projection is None:
            print(f"OVERGESLAGEN {time_seconds:.1f}s: {local.reason}")
            continue
        intermediates.append(
            CameraAnchor3D(
                anchor_id=f"local-{frame_number}",
                goal_id=parent.goal_id,
                frame_number=frame_number,
                time_seconds=time_seconds,
                camera_state=parent.camera_state,
                view_position=None,
                projection=local.projection,
                rms_error_px=parent.rms_error_px,
                maximum_error_px=parent.maximum_error_px,
                anchor_type="intermediate",
                parent_anchor_id=parent.anchor_id,
                local_inliers=local.inliers,
                local_inlier_ratio=local.inlier_ratio,
                local_coverage=min(local.anchor_coverage, local.frame_coverage),
            )
        )
        print(
            f"TUSSENANKER {time_seconds:.1f}s -> {parent.anchor_id} | "
            f"inliers {local.inliers} | dekking {min(local.anchor_coverage, local.frame_coverage):.1%}"
        )
    capture.release()
    if not intermediates:
        raise RuntimeError("Geen enkel voorgesteld tussenanker voldeed aan de QA.")
    expanded = CameraAnchorBank3D(
        bank.match_format,
        bank.video_name,
        bank.pitch_length_m,
        bank.pitch_width_m,
        primaries + tuple(intermediates),
    )
    output = output_dir / f"{prefix}_camera_anchors_3d_expanded.json"
    save_camera_anchor_bank(expanded, output)
    print(f"Uitgebreide 3D-camera-ankerbank: {output}")
    print(f"Primaire ankers: {len(primaries)} | tussenankers: {len(intermediates)}")


if __name__ == "__main__":
    main()
