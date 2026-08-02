from __future__ import annotations

from dataclasses import dataclass

from football_ai.analysis.possession import (
    PassEvent,
    PossessionObservation,
    PossessionState,
    TurnoverEvent,
)


@dataclass(frozen=True, slots=True)
class MatchTimelineResult:
    """Canonical possession timeline and the events derived from it."""

    observations: tuple[PossessionObservation, ...]
    passes: tuple[PassEvent, ...]
    turnovers: tuple[TurnoverEvent, ...]
    events: tuple[MatchTimelineEvent, ...]
    suppressed_team_switches: int


@dataclass(frozen=True, slots=True)
class MatchTimelineEvent:
    """One public, chronologically sortable match event."""

    event_type: str
    start_frame: int
    end_frame: int
    from_identity_id: int | None
    to_identity_id: int | None
    from_label: str
    to_label: str
    from_team: str
    to_team: str
    confidence: float
    from_track_id: int | None = None
    to_track_id: int | None = None

    def to_dict(
        self,
        fps: float,
        team_names: dict[str, str] | None = None,
    ) -> dict:
        safe_fps = max(float(fps), 1.0)
        names = team_names or {}
        return {
            "event_type": self.event_type,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "confirmed_at_frame": self.end_frame,
            "start_seconds": self.start_frame / safe_fps,
            "end_seconds": self.end_frame / safe_fps,
            "confirmed_at_seconds": self.end_frame / safe_fps,
            "duration_seconds": max(0, self.end_frame - self.start_frame) / safe_fps,
            "from_identity_id": self.from_identity_id,
            "to_identity_id": self.to_identity_id,
            "from_track_id": self.from_track_id,
            "to_track_id": self.to_track_id,
            "from_label": self.from_label,
            "to_label": self.to_label,
            "from_team": self.from_team,
            "to_team": self.to_team,
            "from_club": _club_name(
                self.from_label,
                names.get(self.from_team, self.from_team),
            ),
            "to_club": _club_name(
                self.to_label,
                names.get(self.to_team, self.to_team),
            ),
            "confidence": self.confidence,
        }


class MatchTimelineEngine:
    """Resolve noisy frame evidence into one temporally consistent timeline.

    The possession tracker deliberately works frame by frame. This second,
    offline layer is allowed to inspect nearby future frames before accepting
    a team switch. Unknown intervals remain unknown (and therefore do not
    change possession percentages), while the last confirmed team remains
    available as context for a later pass or turnover.
    """

    def __init__(
        self,
        fps: float,
        opponent_confirmation_frames: int = 6,
        minimum_opponent_confidence: float = 0.25,
        player_confirmation_frames: int = 2,
        minimum_pass_confidence: float = 0.18,
        maximum_relay_seconds: float = 1.0,
        maximum_event_gap_seconds: float = 4.0,
        minimum_interception_travel_seconds: float = 0.15,
    ) -> None:
        self.fps = max(float(fps), 1.0)
        self.opponent_confirmation_frames = max(2, opponent_confirmation_frames)
        self.minimum_opponent_confidence = max(0.0, minimum_opponent_confidence)
        self.player_confirmation_frames = max(1, player_confirmation_frames)
        self.minimum_pass_confidence = max(0.0, minimum_pass_confidence)
        self.maximum_relay_frames = max(
            1,
            int(round(maximum_relay_seconds * self.fps)),
        )
        self.maximum_event_gap_frames = max(
            1,
            int(round(maximum_event_gap_seconds * self.fps)),
        )
        self.minimum_interception_travel_frames = max(
            2,
            int(round(minimum_interception_travel_seconds * self.fps)),
        )

    def resolve(
        self,
        observations: list[PossessionObservation],
        evidence_passes: list[PassEvent] | None = None,
        evidence_turnovers: list[TurnoverEvent] | None = None,
    ) -> MatchTimelineResult:
        if not observations:
            return MatchTimelineResult((), (), (), (), 0)

        evidence_passes = evidence_passes or []
        evidence_turnovers = evidence_turnovers or []
        accepted = list(observations)
        last_team: str | None = None
        pending_team: str | None = None
        pending_start: int | None = None
        suppressed = 0

        for index, item in enumerate(observations):
            if item.team is None:
                continue
            if item.state not in {PossessionState.CONTROLLED, PossessionState.INFERRED}:
                continue
            if (
                item.state is PossessionState.INFERRED
                and item.evidence == "team_magnet"
            ):
                # Toon de expliciete bezitshypothese, maar gebruik een
                # onzichtbare bal nooit om het laatst bevestigde team te
                # vervangen. Event-afleiding gebruikt alleen CONTROLLED.
                continue
            if last_team is None or item.team == last_team:
                last_team = item.team
                pending_team = None
                pending_start = None
                continue
            if pending_team != item.team:
                pending_team = item.team
                pending_start = index
            assert pending_start is not None
            if self._team_switch_is_confirmed(
                observations,
                pending_start,
                item.team,
            ):
                last_team = item.team
                pending_team = None
                pending_start = None
                continue
            accepted[index] = _without_possession(item)
            suppressed += 1

        passes, turnovers = self._derive_events(
            accepted,
            evidence_passes,
            evidence_turnovers,
        )
        return MatchTimelineResult(
            observations=tuple(accepted),
            passes=tuple(passes),
            turnovers=tuple(turnovers),
            events=tuple(_build_public_events(passes, turnovers)),
            suppressed_team_switches=suppressed,
        )

    def _team_switch_is_confirmed(
        self,
        observations: list[PossessionObservation],
        start: int,
        team: str,
    ) -> bool:
        controlled: list[PossessionObservation] = []
        window_end = min(
            len(observations),
            start + self.opponent_confirmation_frames * 3,
        )
        for item in observations[start:window_end]:
            if item.state is PossessionState.CONTROLLED and item.team == team:
                controlled.append(item)
                if len(controlled) >= self.opponent_confirmation_frames:
                    break
            elif item.state is PossessionState.CONTROLLED and item.team != team:
                break
        if len(controlled) < self.opponent_confirmation_frames:
            return False
        mean_confidence = sum(item.confidence for item in controlled) / len(controlled)
        return mean_confidence >= self.minimum_opponent_confidence

    def _derive_events(
        self,
        observations: list[PossessionObservation],
        evidence_passes: list[PassEvent],
        evidence_turnovers: list[TurnoverEvent],
    ) -> tuple[list[PassEvent], list[TurnoverEvent]]:
        passes: list[PassEvent] = []
        turnovers: list[TurnoverEvent] = []
        last_owner: PossessionObservation | None = None

        for index, item in enumerate(observations):
            if item.state is not PossessionState.CONTROLLED or item.team is None:
                continue
            if last_owner is None:
                last_owner = item
                continue
            if _same_owner(last_owner, item):
                last_owner = item
                continue

            gap = item.frame_number - last_owner.frame_number
            if gap > self.maximum_event_gap_frames:
                last_owner = item
                continue

            if item.team == last_owner.team:
                if _looks_like_immediate_track_fragment(last_owner, item, gap):
                    # A detector can replace a stable identity with a temporary
                    # track during an overlap or skill move. With no temporal
                    # room for ball travel this is continuity, not a pass.
                    last_owner = item
                    continue
                if not self._player_change_is_confirmed(observations, index, item):
                    continue
                evidence = _matching_pass(evidence_passes, last_owner, item)
                passes.append(
                    PassEvent(
                        start_frame=last_owner.frame_number,
                        end_frame=item.frame_number,
                        from_identity_id=last_owner.identity_id,
                        to_identity_id=item.identity_id,
                        from_label=last_owner.label or "Onbekende speler",
                        to_label=item.label or "Onbekende speler",
                        team=item.team,
                        confidence=(
                            evidence.confidence
                            if evidence is not None
                            else min(last_owner.confidence, item.confidence)
                        ),
                        from_track_id=last_owner.track_id,
                        to_track_id=item.track_id,
                    )
                )
            else:
                evidence = _matching_turnover(evidence_turnovers, last_owner, item)
                event_type = (
                    evidence.event_type
                    if evidence is not None
                    else "possession_change"
                )
                if event_type == "possession_change" and self._is_intercepted_pass(
                    observations,
                    last_owner,
                    item,
                ):
                    event_type = "intercepted_pass"
                turnovers.append(
                    TurnoverEvent(
                        start_frame=last_owner.frame_number,
                        end_frame=item.frame_number,
                        from_identity_id=last_owner.identity_id,
                        to_identity_id=item.identity_id,
                        from_label=last_owner.label or "Onbekende speler",
                        to_label=item.label or "Onbekende speler",
                        from_team=last_owner.team,
                        to_team=item.team,
                        event_type=event_type,
                        confidence=(
                            evidence.confidence
                            if evidence is not None
                            else min(last_owner.confidence, item.confidence)
                        ),
                        from_track_id=last_owner.track_id,
                        to_track_id=item.track_id,
                    )
                )
            last_owner = item

        deduplicated = _deduplicate_passes(passes)
        return self._collapse_quick_relays(deduplicated), _deduplicate_turnovers(turnovers)

    def _is_intercepted_pass(
        self,
        observations: list[PossessionObservation],
        previous_owner: PossessionObservation,
        new_owner: PossessionObservation,
    ) -> bool:
        """Distinguish an intercepted travelling ball from a direct tackle.

        During a pass the old owner may remain as inferred context while the
        ball is travelling. An interception is accepted only when that travel
        lasts long enough and is followed by at least one genuinely loose or
        contested frame before the opponent establishes control. A direct
        adjacent owner switch therefore remains an ordinary possession change.
        """

        between = [
            item
            for item in observations
            if previous_owner.frame_number < item.frame_number < new_owner.frame_number
        ]
        inferred_old_owner = sum(
            item.state is PossessionState.INFERRED
            and item.team == previous_owner.team
            and _same_owner(previous_owner, item)
            for item in between
        )
        free_or_contested = sum(
            item.state
            in {
                PossessionState.LOOSE,
                PossessionState.CONTESTED,
                PossessionState.UNKNOWN,
            }
            for item in between
        )
        travel_frames = new_owner.frame_number - previous_owner.frame_number
        return (
            travel_frames >= self.minimum_interception_travel_frames
            and inferred_old_owner >= 2
            and free_or_contested >= 1
        )

    def _collapse_quick_relays(self, events: list[PassEvent]) -> list[PassEvent]:
        """Count a brief same-team relay as one continuous successful pass.

        A short touch by an intermediate teammate may alter the ball trajectory,
        but should not automatically inflate the pass count. Longer, established
        possession by that teammate still separates two genuine passes.
        """

        collapsed: list[PassEvent] = []
        for item in events:
            if not collapsed:
                collapsed.append(item)
                continue
            previous = collapsed[-1]
            relay_duration = item.start_frame - previous.end_frame
            same_intermediate = _pass_recipient_key(previous) == _pass_sender_key(item)
            if (
                previous.team == item.team
                and same_intermediate
                and 0 <= relay_duration <= self.maximum_relay_frames
            ):
                combined = PassEvent(
                    start_frame=previous.start_frame,
                    end_frame=item.end_frame,
                    from_identity_id=previous.from_identity_id,
                    to_identity_id=item.to_identity_id,
                    from_label=previous.from_label,
                    to_label=item.to_label,
                    team=item.team,
                    confidence=max(previous.confidence, item.confidence),
                    from_track_id=previous.from_track_id,
                    to_track_id=item.to_track_id,
                )
                if _pass_sender_key(combined) == _pass_recipient_key(combined):
                    collapsed.pop()
                else:
                    collapsed[-1] = combined
                continue
            collapsed.append(item)
        return collapsed

    def _player_change_is_confirmed(
        self,
        observations: list[PossessionObservation],
        start: int,
        candidate: PossessionObservation,
    ) -> bool:
        confirmations = 0
        window_end = min(
            len(observations),
            start + self.player_confirmation_frames * 3,
        )
        for item in observations[start:window_end]:
            if item.state is not PossessionState.CONTROLLED:
                continue
            if _same_owner(candidate, item):
                confirmations += 1
                if confirmations >= self.player_confirmation_frames:
                    return True
            elif item.team != candidate.team:
                break
        return confirmations >= self.player_confirmation_frames


def _without_possession(item: PossessionObservation) -> PossessionObservation:
    return PossessionObservation(
        frame_number=item.frame_number,
        state=(
            PossessionState.UNKNOWN
            if item.state is PossessionState.INFERRED
            else PossessionState.CONTESTED
        ),
        identity_id=None,
        track_id=None,
        label=None,
        team=None,
        confidence=item.confidence,
        evidence=item.evidence,
    )


def _same_owner(first: PossessionObservation, second: PossessionObservation) -> bool:
    if first.team != second.team:
        return False
    if first.identity_id is not None and second.identity_id is not None:
        return first.identity_id == second.identity_id
    return first.track_id == second.track_id


def _looks_like_immediate_track_fragment(
    first: PossessionObservation,
    second: PossessionObservation,
    frame_gap: int,
) -> bool:
    """Recognise an instantaneous stable-ID/temporary-track handover."""

    return (
        first.team == second.team
        and frame_gap <= 1
        and (first.identity_id is None or second.identity_id is None)
    )


def _matching_pass(
    evidence: list[PassEvent],
    first: PossessionObservation,
    second: PossessionObservation,
) -> PassEvent | None:
    return next(
        (
            item
            for item in evidence
            if item.team == second.team
            and _pass_sender_key(item) == _observation_owner_key(first)
            and _pass_recipient_key(item) == _observation_owner_key(second)
            and abs(item.end_frame - second.frame_number) <= 2
        ),
        None,
    )


def _matching_turnover(
    evidence: list[TurnoverEvent],
    first: PossessionObservation,
    second: PossessionObservation,
) -> TurnoverEvent | None:
    return next(
        (
            item
            for item in evidence
            if item.from_team == first.team
            and item.to_team == second.team
            and abs(item.end_frame - second.frame_number) <= 2
            and (
                _turnover_recipient_key(item) == _observation_owner_key(second)
                or _turnover_recipient_key(item) is None
                or _observation_owner_key(second) is None
            )
        ),
        None,
    )


def _deduplicate_passes(events: list[PassEvent]) -> list[PassEvent]:
    result: list[PassEvent] = []
    seen: set[tuple[int, int, tuple | None, tuple | None]] = set()
    for item in events:
        sender = _pass_sender_key(item)
        recipient = _pass_recipient_key(item)
        key = (
            item.start_frame,
            item.end_frame,
            sender,
            recipient,
        )
        if key not in seen and sender is not None and sender != recipient:
            result.append(item)
            seen.add(key)
    return result


def _deduplicate_turnovers(events: list[TurnoverEvent]) -> list[TurnoverEvent]:
    result: list[TurnoverEvent] = []
    seen: set[tuple[int, int, str, str]] = set()
    for item in events:
        key = (item.start_frame, item.end_frame, item.from_team, item.to_team)
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _build_public_events(
    passes: list[PassEvent],
    turnovers: list[TurnoverEvent],
) -> list[MatchTimelineEvent]:
    events = [
        MatchTimelineEvent(
            event_type="successful_pass",
            start_frame=item.start_frame,
            end_frame=item.end_frame,
            from_identity_id=item.from_identity_id,
            to_identity_id=item.to_identity_id,
            from_label=item.from_label,
            to_label=item.to_label,
            from_team=item.team,
            to_team=item.team,
            confidence=item.confidence,
            from_track_id=item.from_track_id,
            to_track_id=item.to_track_id,
        )
        for item in passes
    ]
    events.extend(
        MatchTimelineEvent(
            event_type=item.event_type,
            start_frame=item.start_frame,
            end_frame=item.end_frame,
            from_identity_id=item.from_identity_id,
            to_identity_id=item.to_identity_id,
            from_label=item.from_label,
            to_label=item.to_label,
            from_team=item.from_team,
            to_team=item.to_team,
            confidence=item.confidence,
            from_track_id=item.from_track_id,
            to_track_id=item.to_track_id,
        )
        for item in turnovers
    )
    return sorted(events, key=lambda item: (item.end_frame, item.start_frame, item.event_type))


def _owner_key(
    identity_id: int | None,
    track_id: int | None,
    label: str | None,
) -> tuple[str, int | str] | None:
    if identity_id is not None:
        return ("identity", identity_id)
    if track_id is not None:
        return ("track", track_id)
    if label:
        return ("label", label)
    return None


def _observation_owner_key(item: PossessionObservation) -> tuple[str, int | str] | None:
    return _owner_key(item.identity_id, item.track_id, item.label)


def _pass_sender_key(item: PassEvent) -> tuple[str, int | str] | None:
    return _owner_key(item.from_identity_id, item.from_track_id, item.from_label)


def _pass_recipient_key(item: PassEvent) -> tuple[str, int | str] | None:
    return _owner_key(item.to_identity_id, item.to_track_id, item.to_label)


def _turnover_recipient_key(item: TurnoverEvent) -> tuple[str, int | str] | None:
    return _owner_key(item.to_identity_id, item.to_track_id, item.to_label)


def _club_name(label: str, fallback: str) -> str:
    if " - " not in label:
        return fallback
    club = label.split(" - ", 1)[0].strip()
    if not club or club.lower() == "onbekend" or club.upper().startswith("ID "):
        return fallback
    return club
