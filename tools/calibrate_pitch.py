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
    failed_calibration_path = (
        output_dir / "brandevoortbrab_half_pitch_failed.json"
    )
    failed_preview_path = (
        output_dir / "brandevoortbrab_half_pitch_failed_preview.jpg"
    )

    if not video_path.exists():
        raise FileNotFoundError(f"Video niet gevonden: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    profile = create_half_pitch_profile()
    calibrator = MultiFramePitchCalibrator(profile=profile)

    try:
        calibration = calibrator.calibrate_video(video_path=video_path)
    except RuntimeError as error:
        print()
        print("KALIBRATIE AFGEBROKEN")
        print(str(error))
        print(
            "Selecteer extra tussenframes wanneer opeenvolgende beelden "
            "onvoldoende visuele overlap hebben."
        )
        raise SystemExit(2) from None
    preview = calibrator.create_preview(
        calibration=calibration,
        grid_interval_m=5.0,
    )

    if calibration.is_usable:
        target_calibration_path = calibration_path
        target_preview_path = preview_path
    else:
        target_calibration_path = failed_calibration_path
        target_preview_path = failed_preview_path

    calibration.save(target_calibration_path)

    if not cv2.imwrite(str(target_preview_path), preview):
        raise RuntimeError(
            f"Preview kon niet worden opgeslagen: {target_preview_path}"
        )

    if not calibration.is_usable:
        print()
        print("KALIBRATIE AFGEKEURD - actieve kalibratie niet overschreven.")
        print(f"Diagnostiek: {target_calibration_path}")
        print(f"Preview: {target_preview_path}")
        raise SystemExit(2)

    print()
    print("Kalibratie gereed.")
    print(f"Veldprofiel: {profile.name}")
    print(
        f"Afmetingen: {profile.length_m:.1f} x "
        f"{profile.width_m:.1f} meter"
    )
    print(f"Kalibratie: {target_calibration_path}")
    print(f"Preview: {target_preview_path}")


if __name__ == "__main__":
    main()
