from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detector import FootballDetector
from football_ai.tracker import FootballTracker


def main() -> None:
    print("=" * 50)
    print("⚽ Football AI")
    print("=" * 50)

    video_path = Path(
        "videos/Test2_4k_opbouw_keeper.mp4"
    )

    output_path = Path(
        "output/Test2_4k_opbouw_keeper_tracking_test.mp4"
    )

    max_seconds = 5

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video niet gevonden: {video_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Video kon niet worden geopend: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if fps <= 0:
        fps = 30.0

    frames_to_process = int(
        fps * max_seconds
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Outputvideo kon niet worden gemaakt: "
            f"{output_path}"
        )

    detector = FootballDetector(
        player_threshold=0.20,
        ball_threshold=0.05,
    )

    tracker = FootballTracker(
        frame_rate=fps,
    )

    frame_number = 0

    try:
        while frame_number < frames_to_process:
            success, frame = capture.read()

            if not success:
                break

            (
                all_detections,
                player_detections,
                ball_detections,
            ) = detector.detect(frame)

            tracked_players = tracker.update(
                player_detections
            )

            annotated_frame = frame.copy()

            for index in range(
                len(tracked_players)
            ):
                x1, y1, x2, y2 = (
                    tracked_players.xyxy[index]
                    .astype(int)
                )

                confidence = float(
                    tracked_players.confidence[index]
                )

                tracker_id = None

                if (
                    tracked_players.tracker_id
                    is not None
                ):
                    tracker_id = int(
                        tracked_players.tracker_id[index]
                    )

                if tracker_id is None:
                    label = (
                        f"Player {confidence:.2f}"
                    )
                else:
                    label = (
                        f"Player {tracker_id} "
                        f"{confidence:.2f}"
                    )

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (
                        x1,
                        max(y1 - 8, 20),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(annotated_frame)

            frame_number += 1

            if frame_number % 30 == 0:
                print(
                    f"Frame {frame_number}/"
                    f"{frames_to_process} verwerkt"
                )

    finally:
        capture.release()
        writer.release()

    print()
    print("✅ Trackingtest gereed")
    print(
        f"✅ Frames verwerkt: {frame_number}"
    )
    print(
        f"✅ Output opgeslagen: {output_path}"
    )


if __name__ == "__main__":
    main()