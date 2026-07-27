from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GoalkeeperGoalReference:
    frame_number: int
    time_seconds: float
    defending_team_id: int
    first_post: tuple[float, float]
    second_post: tuple[float, float]

    def __post_init__(self) -> None:
        if self.frame_number < 0 or self.time_seconds < 0.0:
            raise ValueError("Frame en tijd van een doelreferentie mogen niet negatief zijn.")
        if self.defending_team_id not in (0, 1):
            raise ValueError("Het verdedigende team moet team 0 of team 1 zijn.")
        if self.first_post == self.second_post:
            raise ValueError("Beide doelpalen mogen niet hetzelfde beeldpunt zijn.")

    @property
    def goal_id(self) -> str:
        return f"team-{self.defending_team_id + 1}"

    @property
    def first_ground(self) -> tuple[float, float]:
        return self.first_post

    @property
    def second_ground(self) -> tuple[float, float]:
        return self.second_post

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "time_seconds": self.time_seconds,
            "defending_team_id": self.defending_team_id,
            "first_post": list(self.first_post),
            "second_post": list(self.second_post),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoalkeeperGoalReference":
        return cls(
            frame_number=int(data["frame_number"]),
            time_seconds=float(data["time_seconds"]),
            defending_team_id=int(data["defending_team_id"]),
            first_post=tuple(map(float, data["first_post"])),
            second_post=tuple(map(float, data["second_post"])),
        )


def save_goalkeeper_goal_references(
    source_video: str,
    references: tuple[GoalkeeperGoalReference, ...],
    path: Path,
) -> None:
    if not source_video.strip():
        raise ValueError("source_video mag niet leeg zijn.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_video": source_video,
                "references": [item.to_dict() for item in references],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_goalkeeper_goal_references(
    path: Path,
) -> tuple[GoalkeeperGoalReference, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 1)) != 1:
        raise ValueError("Niet-ondersteunde doelreferentieversie.")
    return tuple(
        GoalkeeperGoalReference.from_dict(item)
        for item in data.get("references", [])
    )
