from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from football_ai.tracking.entity_corrections import (
    EntityRole,
    TeamAssignment,
)
from football_ai.tracking.entity_resolver import ResolvedEntity
from football_ai.detection.ball_tracking import BallObservation


TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 100, 0),
    1: (0, 0, 255),
}

UNKNOWN_COLOR = (0, 255, 255)
GOALKEEPER_COLOR = (255, 0, 255)
FOOTPOINT_COLOR = (0, 255, 0)
FOOTPOINT_OUTLINE_COLOR = (0, 64, 0)
BALL_COLOR = (0, 255, 255)
BALL_PREDICTED_COLOR = (0, 165, 255)


def draw_ball_observation(
    frame: np.ndarray,
    observation: BallObservation | None,
) -> np.ndarray:
    """Draw a detected or temporarily predicted ball without hiding the frame."""

    annotated = frame.copy()
    if observation is None:
        return annotated
    frame_height, frame_width = annotated.shape[:2]
    center = np.asarray(observation.center, dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        return annotated
    x = int(np.clip(center[0], 0, frame_width - 1))
    y = int(np.clip(center[1], 0, frame_height - 1))
    color = BALL_COLOR if observation.source == "detected" else BALL_PREDICTED_COLOR
    radius = 10 if observation.source == "detected" else 8
    cv2.circle(annotated, (x, y), radius, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(annotated, (x, y), radius, color, 2, cv2.LINE_AA)
    label = f"BAL {observation.confidence:.0%}"
    if observation.source == "predicted":
        label += " voorspeld"
    cv2.putText(
        annotated,
        label,
        (x + 14, max(22, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated


def draw_footpoint(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> None:
    """Teken het grondreferentiepunt midden tussen de voeten van een persoon."""

    x1, _y1, x2, y2 = box
    frame_height, frame_width = frame.shape[:2]
    foot_x = int(np.clip(round((x1 + x2) / 2.0), 0, frame_width - 1))
    foot_y = int(np.clip(y2, 0, frame_height - 1))
    cv2.circle(
        frame,
        (foot_x, foot_y),
        6,
        FOOTPOINT_OUTLINE_COLOR,
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        (foot_x, foot_y),
        4,
        FOOTPOINT_COLOR,
        -1,
        cv2.LINE_AA,
    )


def draw_tracked_players(
    frame: np.ndarray,
    tracked_players: sv.Detections,
    team_by_tracker_id: dict[int, int] | None = None,
    resolved_entities: dict[int, ResolvedEntity] | None = None,
    show_excluded: bool = False,
) -> np.ndarray:
    annotated_frame = frame.copy()

    if team_by_tracker_id is None:
        team_by_tracker_id = {}

    if resolved_entities is None:
        resolved_entities = {}

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

        entity = (
            resolved_entities.get(tracker_id)
            if tracker_id is not None
            else None
        )

        if entity is not None and entity.excluded and not show_excluded:
            continue

        if entity is not None:
            color, team_label = _entity_style(entity)
        elif team_id is None:
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
        draw_footpoint(annotated_frame, (x1, y1, x2, y2))

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


def _entity_style(entity: ResolvedEntity) -> tuple[tuple[int, int, int], str]:
    if entity.excluded:
        return (128, 128, 128), "UITGESLOTEN"

    team_colors = {
        TeamAssignment.TEAM_A: TEAM_COLORS[0],
        TeamAssignment.TEAM_B: TEAM_COLORS[1],
    }
    color = (
        GOALKEEPER_COLOR
        if entity.role is EntityRole.GOALKEEPER
        else team_colors.get(entity.team, UNKNOWN_COLOR)
    )

    role_labels = {
        EntityRole.PLAYER: "Speler",
        EntityRole.GOALKEEPER: "Keeper",
        EntityRole.REFEREE: "Scheidsrechter",
        EntityRole.STAFF: "Staf",
        EntityRole.SPECTATOR: "Toeschouwer",
        EntityRole.UNKNOWN: "Persoon ?",
    }
    team_labels = {
        TeamAssignment.TEAM_A: "A",
        TeamAssignment.TEAM_B: "B",
        TeamAssignment.OFFICIAL: "",
        TeamAssignment.NONE: "",
        TeamAssignment.UNKNOWN: "?",
    }
    role = role_labels[entity.role]
    team = team_labels[entity.team]
    if entity.role is EntityRole.UNKNOWN and entity.team is TeamAssignment.UNKNOWN:
        return color, role
    return color, f"{role} {team}".strip()


def draw_resolved_track_boxes(
    frame: np.ndarray,
    boxes_by_tracker_id: dict[int, tuple[float, float, float, float]],
    resolved_entities: dict[int, ResolvedEntity],
    agreement_by_tracker_id: dict[int, float] | None = None,
    label_by_tracker_id: dict[int, str] | None = None,
) -> np.ndarray:
    """Teken een tweede pass met één definitief label per volledige track."""

    annotated_frame = frame.copy()
    agreements = agreement_by_tracker_id or {}
    identity_labels = label_by_tracker_id or {}
    for tracker_id, box in boxes_by_tracker_id.items():
        entity = resolved_entities[tracker_id]
        if entity.excluded:
            continue
        color, entity_label = _entity_style(entity)
        x1, y1, x2, y2 = (int(value) for value in box)
        agreement = agreements.get(tracker_id, 0.0)
        certainty = f" {agreement:.0%}" if agreement > 0 else ""
        identity_label = identity_labels.get(tracker_id)
        label = (
            f"{identity_label} | {entity_label}{certainty}"
            if identity_label is not None
            else f"ID {tracker_id} {entity_label}{certainty}"
        )
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        draw_footpoint(annotated_frame, (x1, y1, x2, y2))
        cv2.putText(
            annotated_frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated_frame
