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
    evidence: str | None = None

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "state": self.state.value,
            "identity_id": self.identity_id,
            "track_id": self.track_id,
            "label": self.label,
            "team": self.team,
            "confidence": self.confidence,
            "evidence": self.evidence,
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
    from_track_id: int | None = None
    to_track_id: int | None = None

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
    from_track_id: int | None = None
    to_track_id: int | None = None

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
        teammate_reception_ground_zone_ratio: float = 0.60,
        dwell_control_frames: int = 6,
        motion_evidence_memory_frames: int = 6,
        low_speed_threshold: float = 4.0,
        deceleration_ratio: float = 0.55,
        direction_change_degrees: float = 35.0,
        minimum_pass_confidence: float = 0.12,
        inferred_confidence_decay: float = 0.985,
        minimum_inferred_confidence: float = 0.12,
        owner_attachment_radius: float = 1.15,
        same_player_handover_radius: float = 1.35,
        synthetic_ball_override_confidence: float = 0.50,
        long_travel_reception_frames: int = 10,
        maximum_unseen_possession_frames: int = 12,
        maximum_team_magnet_frames: int = 90,
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
        self.teammate_reception_ground_zone_ratio = teammate_reception_ground_zone_ratio
        self.dwell_control_frames = max(confirmation_frames, dwell_control_frames)
        self.motion_evidence_memory_frames = motion_evidence_memory_frames
        self.low_speed_threshold = low_speed_threshold
        self.deceleration_ratio = deceleration_ratio
        self.direction_change_degrees = direction_change_degrees
        self.minimum_pass_confidence = minimum_pass_confidence
        self.inferred_confidence_decay = inferred_confidence_decay
        self.minimum_inferred_confidence = minimum_inferred_confidence
        self.owner_attachment_radius = owner_attachment_radius
        self.same_player_handover_radius = same_player_handover_radius
        self.synthetic_ball_override_confidence = synthetic_ball_override_confidence
        self.long_travel_reception_frames = long_travel_reception_frames
        self.maximum_unseen_possession_frames = max(
            0,
            int(maximum_unseen_possession_frames),
        )
        self.maximum_team_magnet_frames = max(
            self.maximum_unseen_possession_frames,
            int(maximum_team_magnet_frames),
        )
        self._owner_key: tuple[int | None, int, str] | None = None
        self._owner: TimelineEntity | None = None
        self._candidate_key: tuple[int | None, int, str] | None = None
        self._candidate: TimelineEntity | None = None
        self._candidate_count = 0
        self._candidate_has_motion_evidence = False
        self._missing = 0
        self._ball_unseen = 0
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
        missing_before_candidate = self._missing
        # Een voorspeld/geinterpoleerd punt is hulpmateriaal, geen hard bewijs.
        # Wanneer zo'n punt duidelijk los ligt van de actuele bezitter, krijgt
        # de stabiele bezitshypothese voorrang. Een echte modeldetectie blijft
        # altijd leidend en kan dus wel een pass of balverlies bevestigen.
        current_owner = _current_entity(self._owner_key, entities)
        if (
            ball is not None
            and ball.source in {"interpolated", "predicted", "stationary_hold"}
            and ball.confidence < self.synthetic_ball_override_confidence
            and current_owner is not None
            and _normalized_ball_distance(ball, current_owner)
            > self.owner_attachment_radius
        ):
            ball = None
        if ball is None:
            self._ball_unseen += 1
        else:
            self._ball_unseen = 0
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
                    from_track_id=previous.track_id,
                    to_track_id=candidate.track_id,
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
        # Logische "magneet" rond het voetpunt. Zolang de bal nog duidelijk in
        # de controlezone van de huidige bezitter ligt, mag een andere speler
        # het bezit niet door één nipt kleinere beeldafstand overnemen. Een
        # echte onderschepping is hierboven al apart en strenger beoordeeld.
        current_owner = _current_entity(self._owner_key, entities)
        owner_distance = _normalized_ball_distance(ball, current_owner)
        if (
            current_owner is not None
            and owner_distance <= self.owner_attachment_radius
            and key != self._owner_key
        ):
            self._owner = current_owner
            candidate = current_owner
            key = self._owner_key
            state = PossessionState.CONTROLLED
            confidence = max(
                confidence,
                max(0.0, 1.0 - owner_distance / self.owner_attachment_radius)
                * (float(ball.confidence) if ball is not None else 0.0),
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
            team_magnet_owner = _current_entity(self._owner_key, entities)
            if team_magnet_owner is None and self._owner is not None:
                team_magnet_owner = _nearby_same_team_handover(
                    self._owner,
                    entities,
                    self.same_player_handover_radius,
                )
                if team_magnet_owner is not None:
                    self._owner = team_magnet_owner
                    self._owner_key = _entity_key(team_magnet_owner)
            if (
                self._owner is not None
                and self._ball_unseen <= self.maximum_unseen_possession_frames
            ):
                inferred_confidence = max(
                    self.minimum_inferred_confidence,
                    self.inferred_confidence_decay ** self._missing,
                )
                return _observation(
                    frame_number,
                    PossessionState.INFERRED,
                    self._owner,
                    inferred_confidence,
                    evidence="team_magnet" if team_magnet_owner is not None else "short_occlusion",
                )
            if (
                self._owner is not None
                and team_magnet_owner is not None
                and self._ball_unseen <= self.maximum_team_magnet_frames
                and team_magnet_owner.team.value in {"team_a", "team_b"}
            ):
                # De fysieke bal is niet zichtbaar, maar de laatst bevestigde
                # bezitter wordt nog als dezelfde speler gevolgd. Drukte of
                # een speler die ervoor langs loopt is op zichzelf geen bewijs
                # van balverlies. Bewaar alleen de bezitshypothese; dit maakt
                # geen balobservatie, pass of turnover aan.
                self._owner = team_magnet_owner
                inferred_confidence = max(
                    self.minimum_inferred_confidence,
                    self.inferred_confidence_decay ** self._missing,
                )
                return _observation(
                    frame_number,
                    PossessionState.INFERRED,
                    self._owner,
                    inferred_confidence,
                    evidence="team_magnet",
                )
            if (
                self._owner is not None
                and self._ball_unseen > self.maximum_unseen_possession_frames
            ):
                # De bal is werkelijk buiten beeld in plaats van kort
                # afgedekt. Pauzeer de bezitsteller en begin bij de volgende
                # zichtbare controle opnieuw, zodat ontbrekend beeld nooit
                # fictief teambezit, een pass of balverlies oplevert.
                self._owner = None
                self._owner_key = None
                self._candidate_key = None
                self._candidate = None
                self._candidate_count = 0
                self._candidate_has_motion_evidence = False
                self._last_change_frame = None
            return _observation(frame_number, state, None, confidence)

        self._missing = 0
        if key == self._owner_key:
            # Bewaar steeds de actuele box en het actuele voetpunt. Anders
            # vergelijkt een latere trackwissel met de plek waar het bezit
            # ooit begon in plaats van met de laatste zichtbare positie.
            self._owner = candidate
            self._candidate_key = None
            self._candidate_count = 0
            self._candidate_has_motion_evidence = False
            return _observation(frame_number, PossessionState.CONTROLLED, self._owner, confidence)
        if key != self._candidate_key:
            self._candidate_key = key
            self._candidate = candidate
            self._candidate_count = 1
            self._candidate_has_motion_evidence = (
                self._current_motion_evidence
                if self._owner is not None
                and candidate.team != self._owner.team
                else self._has_recent_motion_evidence()
            )
        else:
            self._candidate_count += 1
            self._candidate_has_motion_evidence |= (
                self._current_motion_evidence
                if self._owner is not None
                and self._candidate is not None
                and self._candidate.team != self._owner.team
                else self._has_recent_motion_evidence()
            )
        required_frames = self.confirmation_frames
        if (
            self._owner is not None
            and self._candidate is not None
            and self._candidate.team != self._owner.team
        ):
            required_frames = self.opponent_confirmation_frames
        is_opponent_transfer = (
            self._owner is not None
            and self._candidate is not None
            and self._candidate.team != self._owner.team
        )
        one_touch_teammate_reception = (
            self._owner is not None
            and self._candidate is not None
            and self._candidate.team == self._owner.team
            and ball is not None
            and ball.source == "detected"
            and ball.confidence >= 0.15
            and normalized_distance <= self.teammate_control_radius
            and (
                self._current_motion_evidence
                or missing_before_candidate >= self.long_travel_reception_frames
            )
            and _is_in_ground_contact_zone(
                ball,
                self._candidate,
                self.teammate_reception_ground_zone_ratio,
            )
        )
        if one_touch_teammate_reception:
            required_frames = 1
        # Een bal die langs een tegenstander rolt mag niet uitsluitend door
        # nabijheid en verstreken tijd bezit worden. Voor een teamwissel is
        # werkelijk balgedrag nodig dat op controle wijst. Een ploeggenoot kan
        # daarnaast via stabiele nabijheid een pass ontvangen.
        has_control_evidence = self._candidate_has_motion_evidence or (
            not is_opponent_transfer
            and self._candidate_count >= self.dwell_control_frames
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
            same_player_handover = (
                previous.team == self._owner.team
                and _normalized_entity_distance(previous, self._owner)
                <= self.same_player_handover_radius
            )
            if (
                previous.team == self._owner.team
                and not same_player_handover
                and (
                    self._owner.identity_id is not None
                    or one_touch_teammate_reception
                )
                and (
                    confidence >= self.minimum_pass_confidence
                    or one_touch_teammate_reception
                )
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
                        from_track_id=previous.track_id,
                        to_track_id=self._owner.track_id,
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
                        from_track_id=previous.track_id,
                        to_track_id=self._owner.track_id,
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
        if ball is None or ball.source not in {"detected", "interpolated", "predicted", "stationary_hold"}:
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
    """Show the owner magnet only when no reliable ball path is available."""

    if possession.state is not PossessionState.INFERRED:
        return False
    if ball is None:
        return True
    return float(ball.confidence) < reliable_ball_confidence


def _nearest_candidate(
    ball: BallObservation | None,
    entities: list[TimelineEntity],
) -> tuple[TimelineEntity | None, PossessionState, float, float]:
    if ball is None or ball.source not in {"detected", "interpolated", "predicted", "stationary_hold"}:
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


def _entity_key(
    entity: TimelineEntity | None,
) -> tuple[int | None, int, str] | None:
    if entity is None:
        return None
    return (
        entity.identity_id,
        entity.track_id if entity.identity_id is None else -1,
        entity.team.value,
    )


def _current_entity(
    key: tuple[int | None, int, str] | None,
    entities: list[TimelineEntity],
) -> TimelineEntity | None:
    if key is None:
        return None
    return next((entity for entity in entities if _entity_key(entity) == key), None)


def _normalized_ball_distance(
    ball: BallObservation | None,
    entity: TimelineEntity | None,
) -> float:
    if ball is None or entity is None:
        return float("inf")
    height = max(entity.box[3] - entity.box[1], 1.0)
    distance = float(
        np.linalg.norm(
            np.asarray(ball.center, dtype=np.float64)
            - np.asarray(entity.footpoint, dtype=np.float64)
        )
    )
    return distance / max(30.0, 0.55 * height)


def _normalized_entity_distance(
    first: TimelineEntity,
    second: TimelineEntity,
) -> float:
    """Compare two player observations using their ground contact points.

    A tracker can assign a new technical ID after overlap or occlusion. When
    both observations occupy effectively the same physical position, the
    change is an identity handover and cannot represent a football pass.
    """

    first_height = max(first.box[3] - first.box[1], 1.0)
    second_height = max(second.box[3] - second.box[1], 1.0)
    scale = max(30.0, 0.55 * max(first_height, second_height))
    distance = float(
        np.linalg.norm(
            np.asarray(first.footpoint, dtype=np.float64)
            - np.asarray(second.footpoint, dtype=np.float64)
        )
    )
    return distance / scale


def _nearby_same_team_handover(
    previous: TimelineEntity,
    entities: list[TimelineEntity],
    maximum_distance: float,
) -> TimelineEntity | None:
    """Continue an occluded owner across a nearby technical track-ID change."""

    candidates = [
        (_normalized_entity_distance(previous, entity), entity)
        for entity in entities
        if entity.team == previous.team
        and entity.track_id != previous.track_id
    ]
    if not candidates:
        return None
    distance, candidate = min(candidates, key=lambda item: item[0])
    if distance > maximum_distance:
        return None
    opponent_is_also_local = any(
        entity.team.value in {"team_a", "team_b"}
        and entity.team != previous.team
        and _normalized_entity_distance(previous, entity) <= maximum_distance
        for entity in entities
    )
    return None if opponent_is_also_local else candidate


def _observation(
    frame: int,
    state: PossessionState,
    entity: TimelineEntity | None,
    confidence: float,
    evidence: str | None = None,
) -> PossessionObservation:
    return PossessionObservation(
        frame_number=frame,
        state=state,
        identity_id=entity.identity_id if entity else None,
        track_id=entity.track_id if entity else None,
        label=entity.label if entity else None,
        team=entity.team.value if entity else None,
        confidence=float(max(0.0, min(1.0, confidence))),
        evidence=evidence,
    )


def save_possession_report(
    path: Path,
    source_video: str,
    fps: float,
    observations: list[PossessionObservation],
    passes: list[PassEvent],
    turnovers: list[TurnoverEvent] | None = None,
    timeline_metadata: dict | None = None,
    timeline_events: list[dict] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 4,
        "source_video": source_video,
        "fps": fps,
        "timeline_engine": timeline_metadata or {"name": "frame_tracker"},
        "statistics": {
            **build_possession_statistics(observations, fps),
            "events": build_event_statistics(passes, turnovers or []),
        },
        "observations": [item.to_dict() for item in observations],
        "passes": [item.to_dict() for item in passes],
        "turnovers": [item.to_dict() for item in (turnovers or [])],
        "timeline_events": timeline_events or [],
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


def build_event_statistics(
    passes: list[PassEvent],
    turnovers: list[TurnoverEvent],
) -> dict:
    """Summarise match events and credit interceptions to the defending team."""

    teams: dict[str, dict[str, int]] = {}
    players: dict[str, dict[str, int | str]] = {}

    def team_stats(team: str) -> dict[str, int]:
        return teams.setdefault(
            team,
            {
                "successful_passes": 0,
                "failed_passes": 0,
                "possession_losses": 0,
                "interceptions": 0,
            },
        )

    def player_stats(identity_id: int | None, label: str, team: str) -> dict[str, int | str]:
        key = str(identity_id) if identity_id is not None else f"label:{label}"
        return players.setdefault(
            key,
            {
                "label": label,
                "team": team,
                "passes_completed": 0,
                "passes_received": 0,
                "failed_passes": 0,
                "possession_losses": 0,
                "interceptions": 0,
            },
        )

    for event in passes:
        team_stats(event.team)["successful_passes"] += 1
        player_stats(event.from_identity_id, event.from_label, event.team)[
            "passes_completed"
        ] += 1
        player_stats(event.to_identity_id, event.to_label, event.team)[
            "passes_received"
        ] += 1

    for event in turnovers:
        losing_team = team_stats(event.from_team)
        losing_team["possession_losses"] += 1
        player_stats(event.from_identity_id, event.from_label, event.from_team)[
            "possession_losses"
        ] += 1
        if event.event_type != "intercepted_pass":
            continue
        losing_team["failed_passes"] += 1
        team_stats(event.to_team)["interceptions"] += 1
        player_stats(event.from_identity_id, event.from_label, event.from_team)[
            "failed_passes"
        ] += 1
        player_stats(event.to_identity_id, event.to_label, event.to_team)[
            "interceptions"
        ] += 1

    return {
        "successful_passes": len(passes),
        "possession_losses": len(turnovers),
        "interceptions": sum(
            event.event_type == "intercepted_pass" for event in turnovers
        ),
        "teams": teams,
        "players": players,
    }
