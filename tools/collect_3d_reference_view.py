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
from football_ai.calibration.reference_3d import create_field_reference_3d
from football_ai.calibration.reference_observation_app import (
    ReferenceObservationApp,
    create_projection_preview,
    load_goal_seed,
    save_observation_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Schat één 3D-naar-2D camerareferentie vanuit een bekend doelvlak.")
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4", help="Bestandsnaam in videos-map.")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--goal", choices=("A", "B", "a", "b"), default="B")
    args = parser.parse_args()

    video = PROJECT_ROOT / "videos" / args.video
    seeds_path = PROJECT_ROOT / "output" / "pitch_bootstrap" / f"{video.stem}_{args.format}_goal_seeds.json"
    if not video.exists():
        raise FileNotFoundError(f"Video niet gevonden: {video}")
    if not seeds_path.exists():
        raise FileNotFoundError(f"Doel-seeds niet gevonden: {seeds_path}")
    profile = create_detection_profile(args.format)
    reference = create_field_reference_3d(profile)
    seed = load_goal_seed(seeds_path, args.goal)
    result = ReferenceObservationApp(video, seed, reference).run()

    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    output_path = output_dir / f"{video.stem}_{args.format}_view_{args.goal.upper()}_3d.json"
    preview_path = output_dir / f"{video.stem}_{args.format}_view_{args.goal.upper()}_3d.jpg"
    save_observation_result(result, output_path)

    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, seed.frame_number)
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"Frame {seed.frame_number} kon niet voor QA worden gelezen.")
    preview = create_projection_preview(frame, reference, result)
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"QA-preview kon niet worden opgeslagen: {preview_path}")

    print(f"3D-observaties opgeslagen: {output_path}")
    print(f"QA-preview: {preview_path}")
    if result.estimate is None:
        print(f"GEEN GELDIGE PROJECTIE: {result.failure_reason}")
    else:
        print(f"Projectie geldig | RMS {result.estimate.rms_error_px:.2f}px | max {result.estimate.maximum_error_px:.2f}px")


if __name__ == "__main__":
    main()
