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

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.ground_circle_evidence import (
    detect_metric_center_circle,
    project_ground_circle,
)
from football_ai.calibration.lens_intrinsics_io import load_lens_intrinsics
from football_ai.calibration.local_field_atlas import load_local_field_atlas
from qa_local_field_atlas_overlap import _connect_anchors


def main() -> None:
    parser = argparse.ArgumentParser(description="Valideer de handmatige middencirkel tegen het cameramodel.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    review = json.loads((output / f"{prefix}_manual_midfield_circle.json").read_text())
    atlas = load_local_field_atlas(output / f"{prefix}_local_field_atlas.json")
    lens, lens_source = load_lens_intrinsics(output / f"{prefix}_lens_geometry_qa.json")
    profile = create_detection_profile(args.format)
    goal_b = next(item for item in atlas.patches if item.patch_id == "goal-b")
    target = int(review["frame_number"])
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    transforms, graph = _connect_anchors(
        capture, lens, goal_b.anchor_frame, target, goal_b.anchor_frame, fps
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, raw = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Reviewframe kon niet worden gelezen.")
    frame = cv2.undistort(raw, lens.camera_matrix, lens.distortion_coefficients)
    target_to_anchor = transforms[f"f{target}"]
    ground_to_target = np.linalg.inv(target_to_anchor) @ goal_b.ground_to_anchor
    ground_to_target /= ground_to_target[2, 2]
    evidence = detect_metric_center_circle(frame, profile, ground_to_target, 9.15)
    clicked = np.asarray(review["points"], dtype=np.float64)
    result = {
        "schema_version": 1,
        "video": video.name,
        "frame_number": target,
        "time_seconds": target / fps,
        "lens_source": lens_source,
        "frame_graph": graph,
        "camera_constrained_circle": None,
        "accepted": False,
    }
    if evidence is not None:
        projected = project_ground_circle(evidence, ground_to_target, samples=720)
        distances = np.min(
            np.linalg.norm(clicked[:, None, :] - projected[None, :, :], axis=2), axis=1
        )
        maximum = float(np.max(distances))
        rms = float(np.sqrt(np.mean(np.square(distances))))
        accepted = rms <= 5.0 and maximum <= 9.0
        result["camera_constrained_circle"] = {
            **evidence.to_dict(),
            "click_rms_px": rms,
            "click_maximum_px": maximum,
        }
        result["accepted"] = accepted
        cv2.polylines(frame, [np.rint(projected).astype(np.int32)], True, (0, 255, 255), 3, cv2.LINE_AA)
    for point in clicked:
        cv2.circle(frame, tuple(np.rint(point).astype(int)), 7, (255, 0, 255), -1, cv2.LINE_AA)
    status = "GEACCEPTEERD" if result["accepted"] else "AFGEKEURD"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (15, 15, 15), -1)
    cv2.putText(
        frame, f"MIDDENCIRKEL CAMERAFIT: {status}", (18, 48),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0) if result["accepted"] else (0, 0, 255),
        2, cv2.LINE_AA,
    )
    preview = output / f"{prefix}_manual_midfield_circle_qa.jpg"
    report = output / f"{prefix}_manual_midfield_circle_qa.json"
    cv2.imwrite(str(preview), frame)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Middencirkel-camerafit: {status}")
    if result["camera_constrained_circle"] is not None:
        item = result["camera_constrained_circle"]
        print(f"Klikfout: RMS {item['click_rms_px']:.1f}px | max {item['click_maximum_px']:.1f}px")
    print(f"Preview: {preview}")
    print(f"Rapport: {report}")


if __name__ == "__main__":
    main()
