from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detector import FootballDetector
from football_ai.pitch.calibration_model import PitchCalibration
from football_ai.video_processor import VideoProcessor


def main() -> None:
    print("=" * 50)
    print("⚽ Football AI")
    print("=" * 50)

    video_path = (
        PROJECT_ROOT
        / "videos"
        / "brandevoortbrab.mov"
    )

    calibration_path = (
        PROJECT_ROOT
        / "output"
        / "pitch"
        / "brandevoortbrab_half_pitch.json"
    )

    output_path = (
        PROJECT_ROOT
        / "output"
        / "brandevoortbrab_pitch_filter_test.mp4"
    )

    max_seconds = 20

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video niet gevonden: {video_path}"
        )

    if not calibration_path.exists():
        raise FileNotFoundError(
            "Kalibratiebestand niet gevonden.\n"
            "Voer eerst uit:\n"
            "python tools/calibrate_pitch.py\n\n"
            f"Verwacht bestand:\n{calibration_path}"
        )

    calibration = PitchCalibration.load(
        calibration_path
    )

    detector = FootballDetector(
        player_threshold=0.20,
        ball_threshold=0.05,
    )

    processor = VideoProcessor(
        detector=detector,
        pitch_calibration=calibration,
    )

    frames_processed = processor.process(
        video_path=video_path,
        output_path=output_path,
        max_seconds=max_seconds,
    )

    print()
    print("✅ PitchFilter-test gereed")
    print(
        f"✅ Frames verwerkt: "
        f"{frames_processed}"
    )
    print(
        f"✅ Output opgeslagen: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()