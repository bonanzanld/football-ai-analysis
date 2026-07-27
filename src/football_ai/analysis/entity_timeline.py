from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from football_ai.tracking.entity_corrections import EntityRole, TeamAssignment
from football_ai.tracking.entity_identity import EntityIdentitySet
from football_ai.tracking.entity_resolver import EntityResolver
from football_ai.tracking.track_segmentation import TrackSegmentation


@dataclass(frozen=True, slots=True)
class TimelineEntity:
    frame_number: int
    track_id: int
    identity_id: int | None
    label: str
    role: EntityRole
    team: TeamAssignment
    box: tuple[float, float, float, float]
    footpoint: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "track_id": self.track_id,
            "identity_id": self.identity_id,
            "label": self.label,
            "role": self.role.value,
            "team": self.team.value,
            "box": list(self.box),
            "footpoint": list(self.footpoint),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TimelineEntity":
        return cls(
            frame_number=int(data["frame_number"]),
            track_id=int(data["track_id"]),
            identity_id=(int(data["identity_id"]) if data.get("identity_id") is not None else None),
            label=str(data["label"]),
            role=EntityRole(data["role"]),
            team=TeamAssignment(data["team"]),
            box=tuple(float(value) for value in data["box"]),
            footpoint=tuple(float(value) for value in data["footpoint"]),
        )


@dataclass(frozen=True, slots=True)
class EntityTimeline:
    source_video: str
    fps: float
    observations: tuple[TimelineEntity, ...]
    schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_video": self.source_video,
            "fps": self.fps,
            "observations": [item.to_dict() for item in self.observations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EntityTimeline":
        if int(data.get("schema_version", 1)) != 1:
            raise ValueError("Niet-ondersteunde entiteitentijdlijnversie.")
        return cls(
            source_video=str(data["source_video"]),
            fps=float(data["fps"]),
            observations=tuple(TimelineEntity.from_dict(item) for item in data.get("observations", [])),
        )


def build_entity_timeline(
    source_video: str,
    fps: float,
    tracks: list[Any],
    segmentations: dict[int, TrackSegmentation],
    resolver: EntityResolver,
    identities: EntityIdentitySet | None,
    final_teams: dict[int, int],
) -> EntityTimeline:
    identity_by_track = {}
    if identities is not None:
        for identity in identities.identities:
            for track_id in identity.track_ids:
                identity_by_track[track_id] = identity
    observations = []
    for track in tracks:
        identity = identity_by_track.get(track.track_id)
        for frame_number, box in zip(track.observation_frames, track.boxes, strict=True):
            segment = segmentations.get(track.track_id)
            active = segment.segment_at(frame_number) if segment is not None else None
            team_id = active.team_id if active is not None else final_teams.get(track.track_id)
            entity = resolver.resolve(
                track.track_id,
                team_id,
                segment_index=active.index if active is not None else None,
            )
            role = (
                identity.role
                if identity is not None and identity.role is not EntityRole.UNKNOWN
                else entity.role
            )
            team = (
                identity.team
                if identity is not None and identity.team is not TeamAssignment.UNKNOWN
                else entity.team
            )
            if entity.excluded or role not in (EntityRole.PLAYER, EntityRole.GOALKEEPER):
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            observations.append(
                TimelineEntity(
                    frame_number=int(frame_number),
                    track_id=track.track_id,
                    identity_id=identity.identity_id if identity is not None else None,
                    label=identity.label if identity is not None else f"ID {track.track_id}",
                    role=role,
                    team=team,
                    box=(x1, y1, x2, y2),
                    footpoint=((x1 + x2) / 2.0, y2),
                )
            )
    observations.sort(key=lambda item: (item.frame_number, item.track_id))
    return EntityTimeline(source_video, fps, tuple(observations))


def save_entity_timeline(timeline: EntityTimeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timeline.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_entity_timeline(path: Path) -> EntityTimeline:
    return EntityTimeline.from_dict(json.loads(path.read_text(encoding="utf-8")))
