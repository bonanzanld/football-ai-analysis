from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export video frames and create an empty ball ground-truth manifest."
    )
    parser.add_argument("--video", required=True, help="Video in videos/ or an absolute path.")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.start_frame < 0 or args.end_frame < args.start_frame or args.step < 1:
        parser.error("Require 0 <= start-frame <= end-frame and step >= 1")
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / "videos" / video_path
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    output = args.output or (
        PROJECT_ROOT / "data" / "ball_ground_truth" / video_path.stem
    )
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "frames"
    frames_dir.mkdir(exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    annotations = []
    try:
        for frame_number in range(args.start_frame, args.end_frame + 1, args.step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Cannot read frame {frame_number}")
            image_name = f"frame_{frame_number:06d}.jpg"
            if not cv2.imwrite(str(frames_dir / image_name), frame):
                raise RuntimeError(f"Cannot write {image_name}")
            annotations.append(
                {
                    "frame_number": frame_number,
                    "image": f"frames/{image_name}",
                    "visibility": "unreviewed",
                    "ball_box": None,
                    "occlusion": "none",
                    "review_status": "unreviewed",
                    "notes": "",
                }
            )
    finally:
        capture.release()

    manifest = {
        "schema_version": 1,
        "source_video": str(video_path.relative_to(PROJECT_ROOT)),
        "fps": fps,
        "human_review_required": True,
        "annotation_rules": {
            "visibility": ["visible", "occluded", "not_visible", "unreviewed"],
            "ball_box": "[x1, y1, x2, y2] in original image pixels; required for visible/occluded",
            "occlusion": ["none", "player", "shadow", "other"],
        },
        "annotations": annotations,
    }
    manifest_path = output / "annotations.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(annotations)} frames to {frames_dir}")
    print(f"Ground-truth manifest: {manifest_path}")


if __name__ == "__main__":
    main()
