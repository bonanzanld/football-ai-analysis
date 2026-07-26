from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .entity_corrections import (
    EntityCorrectionSet,
    EntityRole,
    TeamAssignment,
)


class EntityDecisionSource(StrEnum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True, slots=True)
class ResolvedEntity:
    track_id: int
    role: EntityRole
    team: TeamAssignment
    excluded: bool
    source: EntityDecisionSource
    note: str = ""

    @property
    def included_in_football_analysis(self) -> bool:
        return not self.excluded and self.role in (
            EntityRole.PLAYER,
            EntityRole.GOALKEEPER,
            EntityRole.REFEREE,
        )


class EntityResolver:
    """Combines automatic team clustering with authoritative manual review."""

    def __init__(
        self,
        corrections: EntityCorrectionSet | None = None,
    ) -> None:
        self.corrections = corrections

    def resolve(
        self,
        track_id: int,
        automatic_team_id: int | None,
        prefer_current_team: bool = False,
        segment_index: int | None = None,
    ) -> ResolvedEntity:
        correction = (
            self.corrections.get(track_id, segment_index)
            if self.corrections is not None
            else None
        )
        if correction is not None:
            team = correction.team
            source = EntityDecisionSource.MANUAL
            if prefer_current_team and automatic_team_id in (0, 1):
                team = (
                    TeamAssignment.TEAM_A
                    if automatic_team_id == 0
                    else TeamAssignment.TEAM_B
                )
                source = EntityDecisionSource.AUTOMATIC
            return ResolvedEntity(
                track_id=track_id,
                role=correction.role,
                team=team,
                excluded=correction.excluded,
                source=source,
                note=correction.note,
            )

        automatic_team = self._map_team(automatic_team_id)
        if automatic_team is TeamAssignment.UNKNOWN:
            return ResolvedEntity(
                track_id=track_id,
                role=EntityRole.UNKNOWN,
                team=TeamAssignment.UNKNOWN,
                excluded=False,
                source=EntityDecisionSource.UNCLASSIFIED,
            )

        return ResolvedEntity(
            track_id=track_id,
            role=EntityRole.PLAYER,
            team=automatic_team,
            excluded=False,
            source=EntityDecisionSource.AUTOMATIC,
        )

    def resolve_many(
        self,
        track_ids: list[int],
        automatic_teams: dict[int, int],
    ) -> dict[int, ResolvedEntity]:
        return {
            track_id: self.resolve(
                track_id=track_id,
                automatic_team_id=automatic_teams.get(track_id),
            )
            for track_id in track_ids
        }

    @staticmethod
    def _map_team(automatic_team_id: int | None) -> TeamAssignment:
        if automatic_team_id == 0:
            return TeamAssignment.TEAM_A
        if automatic_team_id == 1:
            return TeamAssignment.TEAM_B
        return TeamAssignment.UNKNOWN
