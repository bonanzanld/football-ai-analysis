from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import json
import math
from pathlib import Path

import numpy as np

from football_ai.analysis.entity_timeline import TimelineEntity
from football_ai.detection.ball_tracking import BallObservation


class PossessionState(StrEnum):
    CONTROLLED = "controlled"
    INFERRED = "inferred"
    CONTESTED = "contested"
    LOOSE = "loose"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PossessionObservation:
    frame_number: int
    state: PossessionState
    identity_id: int | None
    track_id: int | None
    label: str | None
    team: str | None
    confidence: float

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "state": self.state.value,
            "identity_id": self.identity_id,
            "track_id": self.track_id,
            "label": self.label,
            "team": self.team,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class PassEvent:
    start_frame: int
    end_frame: int
    from_identity_id: int | None
    to_identity_id: int | None
    from_label: str
    to_label: str
    team: str
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TurnoverEvent:
    start_frame: int
    end_frame: int
    from_identity_id: int | None
    to_identity_id: int | None
    from_label: str
    to_label: str
    from_team: str
    to_team: str
    event_type: str
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


class PossessionTracker:
    def __init__(
        self,
        confirmation_frames: int = 3,
        opponent_confirmation_frames: int = 12,
        teammate_control_radius: float = 0.85,
        opponent_control_radius: float = 0.45,
        interception_contact_radius: float = 0.65,
        interception_ground_zone_ratio: float = 0.25,
        dwell_control_frames: int = 6,
        motion_evidence_memory_frames: int = 6,
        low_speed_threshold: float = 4.0,
        deceleration_ratio: float = 0.55,
        direction_change_degrees: float = 35.0,
        minimum_pass_confidence: float = 0.12,
        inferred_confidence_decay: float = 0.985,
        minimum_inferred_confidence: float = 0.12,
    ) -> None:
        self.confirmation_frames = confirmation_frames
        self.opponent_confirmation_frames = max(
            confirmation_frames,
            opponent_confirmation_frames,
        )
        self.teammate_control_radius = teammate_control_radius
        self.opponent_control_radius = opponent_control_radius
        self.interception_contact_radius = interception_contact_radius
        self.interception_ground_zone_ratio = interception_ground_zone_ratio
        self.dwell_control_frames = max(confirmation_frames, dwell_control_frames)
        self.motion_evidence_memory_frames = motion_evidence_memory_frames
        self.low_speed_threshold = low_speed_threshold
        self.deceleration_ratio = deceleration_ratio
        self.direction_change_degrees = direction_change_degrees
        self.minimum_pass_confidence = minimum_pass_confidence
        self.inferred_confidence_decay = inferred_confidence_decay
        self.minimum_inferred_confidence = minimum_inferred_confidence
        self._owner_key: tuple[int | None, int] | None = None
        self._owner: TimelineEntity | None = None
        self._candidate_key: tuple[int | None, int] | None = None
        self._candidate: TimelineEntity | None = None
        self._candidate_count = 0
        self._candidate_has_motion_evidence = False
        self._missing = 0
        self._last_change_frame: int | None = None
        self._ball_centers: list[np.ndarray] = []
        self._motion_evidence_age: int | None = None
        self._current_motion_evidence = False
        self._current_interception_evidence = False
        self.passes: list[PassEvent] = []
        self.turnovers: list[TurnoverEvent] = []

    def update(
        self,
        frame_number: int,
        ball: BallObservation | None,
        entities: list[TimelineEntity],
    ) -> PossessionObservation:
        self._update_ball_motion(ball)
        candidate, state, confidence, normalized_distance = _nearest_candidate(
            ball,
            entities,
        )
        key = _entity_key(candidate)
        if (
            candidate is not None
            and self._owner is not None
            and candidate.team != self._owner.team
            and normalized_distance <= self.interception_contact_radius
            and self._current_interception_evidence
            and ball is not None
            and ball.source == "detected"
            and ball.confidence >= 0.15
            and _is_in_ground_contact_zone(
                ball,
                candidate,
                self.interception_ground_zone_ratio,
            )
        ):
            previous = self._owner
            start_frame = (
                self._last_change_frame
                if self._last_change_frame is not None
                else frame_number
            )
            self.turnovers.append(
                TurnoverEvent(
                    start_frame=start_frame,
                    end_frame=frame_number,
                    from_identity_id=previous.identity_id,
                    to_identity_id=candidate.identity_id,
                    from_label=previous.label,
                    to_label=candidate.label,
                    from_team=previous.team.value,
                    to_team=candidate.team.value,
                    event_type="intercepted_pass",
                    confidence=confidence,
                )
            )
            self._owner = None
            self._owner_key = None
            self._candidate = None
            self._candidate_key = None
            self._candidate_count = 0
            self._candidate_has_motion_evidence = False
            self._missing = 0
            self._last_change_frame = frame_number
            return _observation(
                frame_number,
                PossessionState.CONTESTED,
                None,
                confidence,
            )
        if (
            candidate is not None
            and self._owner is not None
            and key != self._owner_key
        ):
            allowed_radius = (
                self.teammate_control_radius
                if candidate.team == self._owner.team
                else self.opponent_control_radius
            )
            if normalized_distance > allowed_radius:
                candidate = None
                key = None
                state = PossessionState.LOOSE
                confidence = 0.0
        if candidate is None or state is not PossessionState.CONTROLLED:
            self._missing += 1
            self._candidate_key = None
            self._candidate = None
            self._candidate_count = 0
            self._candidate_has_motion_evidence = False
            if self._owner is not None:
                inferred_confidence = max(
                    self.minimum_inferred_confidence,
                    self.inferred_confidence_decay ** self._missing,
                )
                return _observation(
                    frame_number,
                    PossessionState.INFERRED,
                    self._owner,
                    inferred_confidence,
                )
            return _observation(frame_number, state, None, confidence)

        self._missing = 0
        if key == self._owner_key:
            self._candidate_key = None
            self._candidate_count = 0
            self._candidate_has_motion_evidence = False
            return _observation(frame_number, PossessionState.CONTROLLED, self._owner, confidence)
        if key != self._candidate_key:
            self._candidate_key = key
            self._candidate = candidate
            self._candidate_count = 1
            self._candidate_has_motion_evidence = self._has_recent_motion_evidence()
        else:
            self._candidate_count += 1
            self._candidate_has_motion_evidence |= self._has_recent_motion_evidence()
        required_frames = self.confirmation_frames
        if (
            self._owner is not None
            and self._candidate is not None
            and self._candidate.team != self._owner.team
        ):
            required_frames = self.opponent_confirmation_frames
        has_control_evidence = (
            self._candidate_has_motion_evidence
            or self._candidate_count >= self.dwell_control_frames
        )
        if self._candidate_count < required_frames or not has_control_evidence:
            if self._owner is not None:
                return _observation(
                    frame_number,
                    PossessionState.INFERRED,
                    self._owner,
                    max(self.minimum_inferred_confidence, confidence * 0.6),
                )
            return _observation(frame_number, PossessionState.CONTESTED, None, confidence * 0.6)

        previous = self._owner
        previous_change = self._last_change_frame
        self._owner = self._candidate
        self._owner_key = self._candidate_key
        self._candidate = None
        self._candidate_key = None
        self._candidate_count = 0
        self._candidate_has_motion_evidence = False
        self._last_change_frame = frame_number
        if (
            previous is not None
            and self._owner is not None
            and _entity_key(previous) != _entity_key(self._owner)
        ):
            start_frame = previous_change if previous_change is not None else frame_number
            if (
                previous.team == self._owner.team
                and confidence >= self.minimum_pass_confidence
            ):
                self.passes.append(
                    PassEvent(
                        start_frame=start_frame,
                        end_frame=frame_number,
                        from_identity_id=previous.identity_id,
                        to_identity_id=self._owner.identity_id,
                        from_label=previous.label,
                        to_label=self._owner.label,
                        team=self._owner.team.value,
                        confidence=confidence,
                    )
                )
            elif previous.team != self._owner.team:
                self.turnovers.append(
                    TurnoverEvent(
                        start_frame=start_frame,
                        end_frame=frame_number,
                        from_identity_id=previous.identity_id,
                        to_identity_id=self._owner.identity_id,
                        from_label=previous.label,
                        to_label=self._owner.label,
                        from_team=previous.team.value,
                        to_team=self._owner.team.value,
                        event_type="possession_change",
                        confidence=confidence,
                    )
                )
        return _observation(frame_number, PossessionState.CONTROLLED, self._owner, confidence)

    def _update_ball_motion(self, ball: BallObservation | None) -> None:
        self._current_motion_evidence = False
        self._current_interception_evidence = False
        if self._motion_evidence_age is not None:
            self._motion_evidence_age += 1
            if self._motion_evidence_age > self.motion_evidence_memory_frames:
                self._motion_evidence_age = None
        if ball is None or ball.source not in {"detected", "interpolated", "predicted"}:
            return

        self._ball_centers.append(np.asarray(ball.center, dtype=np.float64))
        self._ball_centers = self._ball_centers[-3:]
        if len(self._ball_centers) < 2:
            return

        current_vector = self._ball_centers[-1] - self._ball_centers[-2]
        current_speed = float(np.linalg.norm(current_vector))
        low_speed = current_speed <= self.low_speed_threshold
        strong_deceleration = False
        sharp_turn = False
        if len(self._ball_centers) == 3:
            previous_vector = self._ball_centers[-2] - self._ball_centers[-3]
            previous_speed = float(np.linalg.norm(previous_vector))
            strong_deceleration = (
                previous_speed >= self.low_speed_threshold
                and current_speed <= previous_speed * self.deceleration_ratio
            )
            if previous_speed >= 2.0 and current_speed >= 2.0:
                cosine = float(np.dot(previous_vector, current_vector)) / (
                    previous_speed * current_speed
                )
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                sharp_turn = angle >= self.direction_change_degrees
        if low_speed or strong_deceleration or sharp_turn:
            self._motion_evidence_age = 0
            self._current_motion_evidence = True
        # Een directe onderschepping vereist een zichtbare koerswijziging.
        # Alleen vertragen is onvoldoende: de bal kan ook gewoon langs een
        # tegenstander rollen of daar kort stilvallen.
        if sharp_turn:
            self._current_interception_evidence = True

    def _has_recent_motion_evidence(self) -> bool:
        return self._motion_evidence_age is not None


def should_render_inferred_ball(
    possession: PossessionObservation,
    ball: BallObservation | None,
    reliable_ball_confidence: float = 0.15,
) -> bool:
    """Show a guessed ball marker only when no reliable real ball is visible."""

    if possession.state is not PossessionState.INFERRED:
        return False
    if ball is None:
        return True
    if ball.source != "detected":
        return True
    return float(ball.confidence) < reliable_ball_confidence


def _nearest_candidate(
    ball: BallObservation | None,
    entities: list[TimelineEntity],
) -> tuple[TimelineEntity | None, PossessionState, float, float]:
    if ball is None or ball.source not in {"detected", "interpolated", "predicted"}:
        return None, PossessionState.UNKNOWN, 0.0, float("inf")
    if ball.source == "predicted" and ball.confidence < 0.15:
        return None, PossessionState.UNKNOWN, 0.0, float("inf")
    candidates = []
    ball_point = np.asarray(ball.center, dtype=np.float64)
    for entity in entities:
        height = max(entity.box[3] - entity.box[1], 1.0)
        distance = float(np.linalg.norm(ball_point - np.asarray(entity.footpoint)))
        normalized = distance / max(30.0, 0.55 * height)
        if normalized <= 1.0:
            candidates.append((normalized, entity))
    if not candidates:
        return None, PossessionState.LOOSE, 0.0, float("inf")
    candidates.sort(key=lambda item: item[0])
    best_distance, best = candidates[0]
    confidence = max(0.0, 1.0 - best_distance) * float(ball.confidence)
    if len(candidates) > 1:
        second_distance, second = candidates[1]
        if second.team != best.team and second_distance - best_distance < 0.22:
            return None, PossessionState.CONTESTED, confidence, best_distance
    return best, PossessionState.CONTROLLED, confidence, best_distance


def _is_in_ground_contact_zone(
    ball: BallObservation,
    entity: TimelineEntity,
    maximum_height_ratio: float,
) -> bool:
    """Return whether the ball is close enough to the player's ground plane.

    Image coordinates grow downward. The footpoint is therefore the relevant
    vertical reference; a trajectory near the torso or head cannot count as a
    ground interception even when its 2D distance happens to be small.
    """

    height = max(entity.box[3] - entity.box[1], 1.0)
    vertical_gap = abs(float(ball.center[1]) - float(entity.footpoint[1]))
    return vertical_gap <= maximum_height_ratio * height


def _entity_key(entity: TimelineEntity | None) -> tuple[int | None, int] | None:
    if entity is None:
        return None
    return (entity.identity_id, entity.track_id if entity.identity_id is None else -1)


def _observation(frame: int, state: PossessionState, entity: TimelineEntity | None, confidence: float) -> PossessionObservation:
    return PossessionObservation(
        frame_number=frame,
        state=state,
        identity_id=entity.identity_id if entity else None,
        track_id=entity.track_id if entity else None,
        label=entity.label if entity else None,
        team=entity.team.value if entity else None,
        confidence=float(max(0.0, min(1.0, confidence))),
    )


def save_possession_report(
    path: Path,
    source_video: str,
    fps: float,
    observations: list[PossessionObservation],
    passes: list[PassEvent],
    turnovers: list[TurnoverEvent] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 2,
        "source_video": source_video,
        "fps": fps,
        "statistics": build_possession_statistics(observations, fps),
        "observations": [item.to_dict() for item in observations],
        "passes": [item.to_dict() for item in passes],
        "turnovers": [item.to_dict() for item in (turnovers or [])],
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def build_possession_statistics(
    observations: list[PossessionObservation],
    fps: float,
) -> dict:
    """Count confirmed and inferred possession without conflating certainty.

    Inferred frames contribute to total possession time for the last confirmed
    player and team. They remain separately visible so downstream statistics
    can report both the practical estimate and its evidential quality.
    """

    safe_fps = max(float(fps), 1e-6)
    teams: dict[str, dict] = {}
    players: dict[str, dict] = {}
    for item in observations:
        if item.state not in {PossessionState.CONTROLLED, PossessionState.INFERRED}:
            continue
        if item.team is None or item.label is None:
            continue
        certainty = (
            "confirmed_frames"
            if item.state is PossessionState.CONTROLLED
            else "inferred_frames"
        )
        team_stats = teams.setdefault(
            item.team,
            {"confirmed_frames": 0, "inferred_frames": 0},
        )
        team_stats[certainty] += 1
        player_key = (
            str(item.identity_id)
            if item.identity_id is not None
            else f"track:{item.track_id}"
        )
        player_stats = players.setdefault(
            player_key,
            {
                "label": item.label,
                "team": item.team,
                "confirmed_frames": 0,
                "inferred_frames": 0,
            },
        )
        player_stats[certainty] += 1

    for collection in (teams, players):
        for stats in collection.values():
            total = stats["confirmed_frames"] + stats["inferred_frames"]
            stats["total_possession_frames"] = total
            stats["total_possession_seconds"] = total / safe_fps
            stats["inferred_share"] = (
                stats["inferred_frames"] / total if total else 0.0
            )
    return {"teams": teams, "players": players}
