from __future__ import annotations

import cv2
import numpy as np
import supervision as sv


def draw_tracked_players(
    frame: np.ndarray,
    tracked_players: sv.Detections,
) -> np.ndarray:
    annotated_frame = frame.copy()

    for index in range(len(tracked_players)):
        x1, y1, x2, y2 = (
            tracked_players.xyxy[index].astype(int)
        )

        confidence = float(
            tracked_players.confidence[index]
        )

        tracker_id = None

        if tracked_players.tracker_id is not None:
            tracker_id = int(
                tracked_players.tracker_id[index]
            )

        if tracker_id is None:
            label = f"Player {confidence:.2f}"
        else:
            label = (
                f"Player {tracker_id} "
                f"{confidence:.2f}"
            )

        color = (0, 255, 0)

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            color,
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
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated_frame