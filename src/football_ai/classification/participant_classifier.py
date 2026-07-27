from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ParticipantDecision(StrEnum):
    PLAYER = "player"
    REFEREE_REVIEW = "referee_review"
    OUTSIDER_REVIEW = "outsider_review"
    CONFIRMED_REFEREE = "confirmed_referee"
    CONFIRMED_EXCLUDED = "confirmed_excluded"


@dataclass(frozen=True, slots=True)
class ParticipantEvidence:
    track_id: int
    segment_index: int | None
    team_reliability: float
    team_uniform_distance: float
    player_group_proximity: float
    relative_activity: float
    track_stability: float

    def __post_init__(self) -> None:
        for name in (
            "team_reliability",
            "team_uniform_distance",
            "player_group_proximity",
            "relative_activity",
            "track_stability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} moet tussen 0 en 1 liggen.")


@dataclass(frozen=True, slots=True)
class ParticipantAssessment:
    track_id: int
    segment_index: int | None
    score: float
    decision: ParticipantDecision
    evidence: ParticipantEvidence
    reasons: tuple[str, ...]


class ParticipantClassifier:
    """Conservatieve kandidaatselectie; verwijdert nooit automatisch personen."""

    def assess(self, evidence: ParticipantEvidence) -> ParticipantAssessment:
        referee_score = (
            0.30 * evidence.team_uniform_distance
            + 0.30 * evidence.player_group_proximity
            + 0.22 * evidence.relative_activity
            + 0.18 * evidence.track_stability
        )
        outsider_score = (
            0.38 * (1.0 - evidence.player_group_proximity)
            + 0.24 * (1.0 - evidence.relative_activity)
            + 0.20 * evidence.team_uniform_distance
            + 0.18 * evidence.track_stability
        )
        reasons = []
        if evidence.team_uniform_distance >= 0.55:
            reasons.append("kleding wijkt af van beide teams")
        if evidence.player_group_proximity >= 0.65:
            reasons.append("track blijft tussen of vlak bij de spelers")
        elif evidence.player_group_proximity <= 0.30:
            reasons.append("track blijft meestal buiten de spelersgroep")
        if evidence.relative_activity >= 0.55:
            reasons.append("track beweegt actief mee met het spel")
        elif evidence.relative_activity <= 0.25:
            reasons.append("track verplaatst weinig ten opzichte van de spelers")
        if evidence.track_stability >= 0.60:
            reasons.append("track is lang genoeg gevolgd")

        # Een betrouwbaar teamlid blijft speler. Dit voorkomt dat een afwijkend
        # belicht shirt of een korte occlusie iemand tot official maakt.
        if evidence.team_reliability >= 0.72:
            decision = ParticipantDecision.PLAYER
            score = 1.0 - evidence.team_reliability
        elif (
            evidence.team_uniform_distance >= 0.50
            and evidence.player_group_proximity >= 0.58
            and evidence.relative_activity >= 0.35
            and evidence.track_stability >= 0.30
            and referee_score >= 0.56
        ):
            decision = ParticipantDecision.REFEREE_REVIEW
            score = referee_score
        elif (
            evidence.player_group_proximity <= 0.32
            and evidence.relative_activity <= 0.35
            and evidence.track_stability >= 0.35
            and outsider_score >= 0.56
        ):
            decision = ParticipantDecision.OUTSIDER_REVIEW
            score = outsider_score
        else:
            decision = ParticipantDecision.PLAYER
            score = max(referee_score, outsider_score)

        return ParticipantAssessment(
            track_id=evidence.track_id,
            segment_index=evidence.segment_index,
            score=float(score),
            decision=decision,
            evidence=evidence,
            reasons=tuple(reasons),
        )
