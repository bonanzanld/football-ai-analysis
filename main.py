from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detector import FootballDetector


def main() -> None:
    print("=" * 50)
    print("⚽ Football AI")
    print("=" * 50)

    video_path = Path("videos/Test2_4k_opbouw_keeper.mp4")

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video niet gevonden: {video_path}"
        )

    print(f"✅ Video gevonden: {video_path}")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Video kon niet worden geopend: {video_path}"
        )

    # Test een duidelijker moment in de video.
    capture.set(cv2.CAP_PROP_POS_MSEC, 5000)

    success, frame = capture.read()
    capture.release()

    if not success:
        raise RuntimeError(
            "Het frame op vijf seconden kon niet worden gelezen."
        )

    print(f"✅ Frame gelezen: {frame.shape}")

    detector = FootballDetector(
        player_threshold=0.25,
        ball_threshold=0.05,
    )

    (
        all_detections,
        player_detections,
        ball_detections,
    ) = detector.detect(frame)

    print()
    print(f"Alle detecties: {len(all_detections)}")

    if len(all_detections) > 0:
        print("Class-ID's:")
        print(all_detections.class_id)

        print("Confidences:")
        print(all_detections.confidence)

        if "class_name" in all_detections.data:
            print("Class names:")
            print(all_detections.data["class_name"])

    print()
    print(f"✅ Spelers gedetecteerd: {len(player_detections)}")
    print(f"✅ Ballen gedetecteerd: {len(ball_detections)}")

    output_path = Path("output/test_frame_5_seconds.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)

    print(f"✅ Testframe opgeslagen: {output_path}")


if __name__ == "__main__":
    main()