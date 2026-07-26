from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PositionPeriod:
    """A manually entered tactical role starting at a match minute.

    Its end is implied by the next period for the same player (or full time).
    This is intentionally not inferred from movement data.
    """

    position: str
    start_minute: float

    def __post_init__(self) -> None:
        if not self.position.strip():
            raise ValueError("Een positie mag niet leeg zijn.")
        if self.start_minute < 0:
            raise ValueError("De startminuut van een positie mag niet negatief zijn.")

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "start_minute": self.start_minute,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PositionPeriod":
        return cls(
            position=str(data["position"]),
            start_minute=float(data["start_minute"]),
        )


@dataclass(frozen=True, slots=True)
class PlayerProfile:
    """Human-friendly metadata kept separate from tracking and detection."""

    identity_id: int
    display_name: str
    squad_number: str = ""
    position_periods: tuple[PositionPeriod, ...] = ()

    def __post_init__(self) -> None:
        if self.identity_id < 1:
            raise ValueError("identity_id moet positief zijn.")
        if not self.display_name.strip():
            raise ValueError("Een spelersnaam mag niet leeg zijn.")

    def to_dict(self) -> dict:
        return {
            "identity_id": self.identity_id,
            "display_name": self.display_name,
            "squad_number": self.squad_number,
            "position_periods": [item.to_dict() for item in self.position_periods],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlayerProfile":
        return cls(
            identity_id=int(data["identity_id"]),
            display_name=str(data["display_name"]),
            squad_number=str(data.get("squad_number", "")),
            position_periods=tuple(
                PositionPeriod.from_dict(item)
                for item in data.get("position_periods", [])
            ),
        )


@dataclass(frozen=True, slots=True)
class TeamRoster:
    source_video: str
    own_team_name: str
    players: tuple[PlayerProfile, ...] = ()
    schema_version: int = 1

    def display_label(self, identity_id: int, fallback: str) -> str:
        profile = next((item for item in self.players if item.identity_id == identity_id), None)
        if profile is None:
            return fallback
        suffix = f" (#{profile.squad_number})" if profile.squad_number else ""
        return f"{profile.display_name}{suffix}"

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_video": self.source_video,
            "own_team_name": self.own_team_name,
            "players": [item.to_dict() for item in self.players],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TeamRoster":
        version = int(data.get("schema_version", 1))
        if version != 1:
            raise ValueError(f"Niet-ondersteunde teamselectieversie: {version}")
        return cls(
            source_video=str(data["source_video"]),
            own_team_name=str(data["own_team_name"]),
            players=tuple(PlayerProfile.from_dict(item) for item in data.get("players", [])),
            schema_version=version,
        )


def save_team_roster(roster: TeamRoster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(roster.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_team_roster(path: Path) -> TeamRoster:
    return TeamRoster.from_dict(json.loads(path.read_text(encoding="utf-8")))
