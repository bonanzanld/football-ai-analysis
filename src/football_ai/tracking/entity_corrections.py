from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class EntityRole(StrEnum):
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    STAFF = "staff"
    SPECTATOR = "spectator"
    UNKNOWN = "unknown"


class TeamAssignment(StrEnum):
    TEAM_A = "team_a"
    TEAM_B = "team_b"
    OFFICIAL = "official"
    NONE = "none"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TrackCorrection:
    """Human decision for one complete tracker ID, independent of pitch data."""

    track_id: int
    segment_index: int | None = None
    role: EntityRole = EntityRole.UNKNOWN
    team: TeamAssignment = TeamAssignment.UNKNOWN
    excluded: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError("track_id mag niet negatief zijn.")
        if self.segment_index is not None and self.segment_index < 1:
            raise ValueError("segment_index moet minimaal 1 zijn.")
        if self.role in (EntityRole.STAFF, EntityRole.SPECTATOR) and not self.excluded:
            raise ValueError("Staff en toeschouwers moeten van voetbalanalyse worden uitgesloten.")
        if self.role == EntityRole.REFEREE and self.team not in (
            TeamAssignment.OFFICIAL,
            TeamAssignment.UNKNOWN,
        ):
            raise ValueError("Een scheidsrechter kan niet aan team A of B worden toegewezen.")

    @property
    def included_in_football_analysis(self) -> bool:
        return not self.excluded and self.role in (
            EntityRole.PLAYER,
            EntityRole.GOALKEEPER,
            EntityRole.REFEREE,
        )

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "segment_index": self.segment_index,
            "role": self.role.value,
            "team": self.team.value,
            "excluded": self.excluded,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrackCorrection":
        return cls(
            track_id=int(data["track_id"]),
            segment_index=(
                int(data["segment_index"])
                if data.get("segment_index") is not None
                else None
            ),
            role=EntityRole(data.get("role", EntityRole.UNKNOWN.value)),
            team=TeamAssignment(data.get("team", TeamAssignment.UNKNOWN.value)),
            excluded=bool(data.get("excluded", False)),
            note=str(data.get("note", "")),
        )


@dataclass(frozen=True, slots=True)
class EntityCorrectionSet:
    """Versioned manual review result for one deterministic tracking run."""

    source_video: str
    corrections: tuple[TrackCorrection, ...] = ()
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.source_video.strip():
            raise ValueError("source_video mag niet leeg zijn.")
        keys = [(item.track_id, item.segment_index) for item in self.corrections]
        if len(keys) != len(set(keys)):
            raise ValueError("Een tracksegment mag maar één handmatige correctie hebben.")

    def get(
        self,
        track_id: int,
        segment_index: int | None = None,
    ) -> TrackCorrection | None:
        exact = next(
            (
                item for item in self.corrections
                if item.track_id == track_id and item.segment_index == segment_index
            ),
            None,
        )
        if exact is not None or segment_index is None:
            return exact
        return next(
            (
                item for item in self.corrections
                if item.track_id == track_id and item.segment_index is None
            ),
            None,
        )

    def with_correction(self, correction: TrackCorrection) -> "EntityCorrectionSet":
        values = [
            item for item in self.corrections
            if (item.track_id, item.segment_index)
            != (correction.track_id, correction.segment_index)
        ]
        values.append(correction)
        return EntityCorrectionSet(
            source_video=self.source_video,
            corrections=tuple(
                sorted(
                    values,
                    key=lambda item: (
                        item.track_id,
                        item.segment_index if item.segment_index is not None else 0,
                    ),
                )
            ),
            schema_version=self.schema_version,
        )

    def included_track_ids(self) -> frozenset[int]:
        return frozenset(
            item.track_id for item in self.corrections
            if item.included_in_football_analysis
        )

    def excluded_track_ids(self) -> frozenset[int]:
        return frozenset(item.track_id for item in self.corrections if item.excluded)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_video": self.source_video,
            "corrections": [item.to_dict() for item in self.corrections],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EntityCorrectionSet":
        version = int(data.get("schema_version", 1))
        if version not in (1, 2):
            raise ValueError(f"Niet-ondersteunde entity-correctieversie: {version}")
        return cls(
            source_video=str(data["source_video"]),
            corrections=tuple(
                TrackCorrection.from_dict(item)
                for item in data.get("corrections", [])
            ),
            schema_version=2,
        )


def save_entity_corrections(corrections: EntityCorrectionSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(corrections.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_entity_corrections(path: Path) -> EntityCorrectionSet:
    return EntityCorrectionSet.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
