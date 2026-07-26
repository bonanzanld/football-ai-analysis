from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.bootstrap.goal_seed import load_goal_seeds
from football_ai.calibration.camera_anchor_bank_3d import (
    CameraAnchorBank3D,
    build_camera_anchor,
    save_camera_anchor_bank,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bouw een gevalideerde 3D-camera-ankerbank.")
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4", help="Bestandsnaam in videos-map.")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    args = parser.parse_args()

    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    seeds = load_goal_seeds(output_dir / f"{prefix}_goal_seeds.json")
    anchors = tuple(
        build_camera_anchor(seed, output_dir / f"{prefix}_view_{seed.goal_id}_3d.json")
        for seed in seeds
    )
    profile = create_detection_profile(args.format)
    bank = CameraAnchorBank3D(
        match_format=profile.match_format.value,
        video_name=video.name,
        pitch_length_m=profile.pitch_length_m,
        pitch_width_m=profile.pitch_width_m,
        anchors=anchors,
    )
    output = output_dir / f"{prefix}_camera_anchors_3d.json"
    save_camera_anchor_bank(bank, output)
    print(f"3D-camera-ankerbank opgeslagen: {output}")
    for anchor in bank.anchors:
        print(
            f"{anchor.anchor_id}: frame {anchor.frame_number} | positie {anchor.view_position:.3f} | "
            f"RMS {anchor.rms_error_px:.2f}px | max {anchor.maximum_error_px:.2f}px"
        )


if __name__ == "__main__":
    main()
