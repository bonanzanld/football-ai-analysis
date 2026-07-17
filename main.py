from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detector import FootballDetector
from football_ai.video_processor import VideoProcessor


def main() -> None:
    print("=" * 50)
    print("⚽ Football AI")
    print("=" * 50)

    video_path = Path(
        "videos/brandevoortbrab.mov"
    )

    output_path = Path(
        "output/"
        "brandevoortbrab_player_filter_test.mp4"
    )

    max_seconds = 20

    detector = FootballDetector(
        player_threshold=0.20,
        ball_threshold=0.05,
    )

    processor = VideoProcessor(
        detector=detector,
    )

    frames_processed = processor.process(
        video_path=video_path,
        output_path=output_path,
        max_seconds=max_seconds,
    )

    print()
    print("✅ PlayerFilter-test gereed")
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