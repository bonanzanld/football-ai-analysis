from __future__ import annotations

from pathlib import Path

import cv2

from football_ai.classification.team_classifier import (
    TeamClassifier,
)
from football_ai.detector import FootballDetector
from football_ai.player_filter import PlayerFilter
from football_ai.tracker import FootballTracker
from football_ai.visualizer import draw_tracked_players


class VideoProcessor:
    def __init__(
        self,
        detector: FootballDetector,
    ) -> None:
        self.detector = detector

        self.player_filter = PlayerFilter(
            minimum_box_height=24,
            minimum_aspect_ratio=1.15,
            maximum_aspect_ratio=6.0,
            minimum_foot_y_ratio=0.15,
            minimum_green_ratio=0.18,
        )

        self.team_classifier = TeamClassifier(
            samples_per_player=30,
            minimum_players=4,
            refit_interval=30,
        )

    def process(
        self,
        video_path: Path,
        output_path: Path,
        max_seconds: float | None = None,
    ) -> int:
        if not video_path.exists():
            raise FileNotFoundError(
                f"Video niet gevonden: {video_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Video kon niet worden geopend: "
                f"{video_path}"
            )

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        if fps <= 0:
            fps = 30.0

        frames_to_process = None

        if max_seconds is not None:
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
                "Outputvideo kon niet worden "
                f"gemaakt: {output_path}"
            )

        tracker = FootballTracker(
            frame_rate=fps,
        )

        frame_number = 0

        try:
            while True:
                if (
                    frames_to_process is not None
                    and frame_number >= frames_to_process
                ):
                    break

                success, frame = capture.read()

                if not success:
                    break

                (
                    _all_detections,
                    player_detections,
                    _ball_detections,
                ) = self.detector.detect(frame)

                filtered_player_detections = (
                    self.player_filter.filter(
                        frame=frame,
                        detections=player_detections,
                    )
                )

                tracked_players = tracker.update(
                    filtered_player_detections
                )

                team_by_tracker_id = (
                    self.team_classifier.update(
                        frame=frame,
                        tracked_players=tracked_players,
                    )
                )

                annotated_frame = draw_tracked_players(
                    frame=frame,
                    tracked_players=tracked_players,
                    team_by_tracker_id=team_by_tracker_id,
                )

                writer.write(
                    annotated_frame
                )

                frame_number += 1

                if frame_number % 30 == 0:
                    if frames_to_process is None:
                        print(
                            f"Frame {frame_number} "
                            "verwerkt"
                        )
                    else:
                        print(
                            f"Frame {frame_number}/"
                            f"{frames_to_process} "
                            "verwerkt"
                        )

        finally:
            capture.release()
            writer.release()

        return frame_number