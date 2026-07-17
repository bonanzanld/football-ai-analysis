from __future__ import annotations

import cv2
import numpy as np
import supervision as sv


TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 100, 0),
    1: (0, 0, 255),
}

UNKNOWN_COLOR = (0, 255, 255)


def draw_tracked_players(
    frame: np.ndarray,
    tracked_players: sv.Detections,
    team_by_tracker_id: dict[int, int] | None = None,
) -> np.ndarray:
    annotated_frame = frame.copy()

    if team_by_tracker_id is None:
        team_by_tracker_id = {}

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

        if tracked_players.tracker_id is not None:
            tracker_id = int(
                tracked_players.tracker_id[index]
            )

        team_id = None

        if tracker_id is not None:
            team_id = team_by_tracker_id.get(
                tracker_id
            )

        if team_id is None:
            color = UNKNOWN_COLOR
            team_label = "Team ?"
        else:
            color = TEAM_COLORS.get(
                team_id,
                UNKNOWN_COLOR,
            )
            team_label = f"Team {team_id + 1}"

        if tracker_id is None:
            label = (
                f"{team_label} "
                f"{confidence:.2f}"
            )
        else:
            label = (
                f"ID {tracker_id} "
                f"{team_label} "
                f"{confidence:.2f}"
            )

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