from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TeamConsensusResult:
    track_id: int
    team_id: int | None
    votes_team_a: int
    votes_team_b: int
    total_votes: int
    agreement_ratio: float
    is_reliable: bool


class TeamConsensus:
    """Bepaalt na de detectiepass één stabiel teamlabel per track."""

    def __init__(
        self,
        minimum_votes: int = 15,
        minimum_agreement_ratio: float = 0.80,
    ) -> None:
        if minimum_votes < 1:
            raise ValueError("minimum_votes moet minimaal 1 zijn.")
        if not 0.5 <= minimum_agreement_ratio <= 1.0:
            raise ValueError("minimum_agreement_ratio moet tussen 0.5 en 1.0 liggen.")
        self.minimum_votes = minimum_votes
        self.minimum_agreement_ratio = minimum_agreement_ratio
        self._votes: dict[int, Counter[int]] = defaultdict(Counter)

    def record(
        self,
        visible_track_ids: list[int],
        team_by_tracker_id: dict[int, int],
    ) -> None:
        for track_id in visible_track_ids:
            team_id = team_by_tracker_id.get(track_id)
            if team_id in (0, 1):
                self._votes[track_id][team_id] += 1

    def finalize(self, track_ids: list[int]) -> dict[int, TeamConsensusResult]:
        return {
            track_id: self._finalize_track(track_id)
            for track_id in sorted(set(track_ids))
        }

    def _finalize_track(self, track_id: int) -> TeamConsensusResult:
        votes = self._votes.get(track_id, Counter())
        votes_a = int(votes.get(0, 0))
        votes_b = int(votes.get(1, 0))
        total = votes_a + votes_b
        winning_votes = max(votes_a, votes_b)
        ratio = winning_votes / total if total else 0.0
        reliable = total >= self.minimum_votes and ratio >= self.minimum_agreement_ratio
        team_id = (0 if votes_a >= votes_b else 1) if reliable else None
        return TeamConsensusResult(
            track_id=track_id,
            team_id=team_id,
            votes_team_a=votes_a,
            votes_team_b=votes_b,
            total_votes=total,
            agreement_ratio=ratio,
            is_reliable=reliable,
        )
