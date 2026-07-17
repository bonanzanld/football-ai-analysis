from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.pitch import (
    MultiFramePitchCalibrator,
    create_half_pitch_profile,
)


def main() -> None:
    print("=" * 66)
    print("Football AI - OpenCV multi-frame veldkalibratie")
    print("=" * 66)

    video_path = PROJECT_ROOT / "videos" / "brandevoortbrab.mov"

    output_dir = PROJECT_ROOT / "output" / "pitch"
    calibration_path = output_dir / "brandevoortbrab_half_pitch.json"
    preview_path = output_dir / "brandevoortbrab_half_pitch_preview.jpg"

    if not video_path.exists():
        raise FileNotFoundError(f"Video niet gevonden: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    profile = create_half_pitch_profile()
    calibrator = MultiFramePitchCalibrator(profile=profile)

    calibration = calibrator.calibrate_video(video_path=video_path)
    calibration.save(calibration_path)

    preview = calibrator.create_preview(
        calibration=calibration,
        grid_interval_m=5.0,
    )

    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(
            f"Preview kon niet worden opgeslagen: {preview_path}"
        )

    print()
    print("Kalibratie gereed.")
    print(f"Veldprofiel: {profile.name}")
    print(
        f"Afmetingen: {profile.length_m:.1f} x "
        f"{profile.width_m:.1f} meter"
    )
    print(f"Kalibratie: {calibration_path}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
