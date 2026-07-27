from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TeamEvidence:
    frame_number: int
    team_id: int | None
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class TrackSegment:
    index: int
    first_frame: int
    last_frame: int
    team_id: int | None

    def contains(self, frame_number: int) -> bool:
        return self.first_frame <= frame_number <= self.last_frame

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "segment_id": f"{self.index}",
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "team_id": self.team_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrackSegment":
        team_id = data.get("team_id")
        return cls(
            index=int(data["index"]),
            first_frame=int(data["first_frame"]),
            last_frame=int(data["last_frame"]),
            team_id=int(team_id) if team_id is not None else None,
        )


@dataclass(frozen=True, slots=True)
class TrackSegmentation:
    track_id: int
    segments: tuple[TrackSegment, ...]

    def segment_at(self, frame_number: int) -> TrackSegment | None:
        return next(
            (segment for segment in self.segments if segment.contains(frame_number)),
            None,
        )

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "segments": [
                {
                    **item.to_dict(),
                    "segment_id": f"{self.track_id}.{item.index}",
                }
                for item in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrackSegmentation":
        return cls(
            track_id=int(data["track_id"]),
            segments=tuple(
                TrackSegment.from_dict(item) for item in data.get("segments", [])
            ),
        )


@dataclass(frozen=True, slots=True)
class TrackSegmentationSet:
    source_video: str
    fps: float
    tracks: tuple[TrackSegmentation, ...]
    schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_video": self.source_video,
            "fps": self.fps,
            "tracks": [item.to_dict() for item in self.tracks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrackSegmentationSet":
        version = int(data.get("schema_version", 1))
        if version != 1:
            raise ValueError(f"Niet-ondersteunde tracksegmentatieversie: {version}")
        return cls(
            source_video=str(data["source_video"]),
            fps=float(data["fps"]),
            tracks=tuple(
                TrackSegmentation.from_dict(item) for item in data.get("tracks", [])
            ),
            schema_version=version,
        )


def save_track_segmentations(data: TrackSegmentationSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_track_segmentations(path: Path) -> TrackSegmentationSet:
    return TrackSegmentationSet.from_dict(json.loads(path.read_text(encoding="utf-8")))


def segment_track_by_team_switches(
    track_id: int,
    evidence: list[TeamEvidence],
    initial_team_id: int | None,
    minimum_confidence: float = 0.12,
    confirmation_observations: int = 12,
) -> TrackSegmentation:
    """Split a technical track only after a sustained team-colour change.

    Missing or weak observations do not create a boundary. A confirmed change
    is backdated to the first reliable observation in its confirmation run.
    """

    if confirmation_observations < 2:
        raise ValueError("confirmation_observations moet minimaal 2 zijn.")
    if not evidence:
        return TrackSegmentation(track_id=track_id, segments=())

    ordered = sorted(evidence, key=lambda item: item.frame_number)
    reliable = [
        item
        for item in ordered
        if item.team_id in (0, 1) and item.confidence >= minimum_confidence
    ]
    current_team = initial_team_id
    if current_team not in (0, 1) and reliable:
        current_team = reliable[0].team_id

    boundaries: list[tuple[int, int | None]] = []
    candidate_team: int | None = None
    candidate_frames: list[int] = []

    for item in reliable:
        if item.team_id == current_team:
            candidate_team = None
            candidate_frames.clear()
            continue
        if item.team_id != candidate_team:
            candidate_team = item.team_id
            candidate_frames = [item.frame_number]
        else:
            candidate_frames.append(item.frame_number)
        if len(candidate_frames) < confirmation_observations:
            continue
        boundary_frame = candidate_frames[0]
        boundaries.append((boundary_frame, candidate_team))
        current_team = candidate_team
        candidate_team = None
        candidate_frames.clear()

    first_frame = ordered[0].frame_number
    last_frame = ordered[-1].frame_number
    segment_start = first_frame
    segment_team = initial_team_id if initial_team_id in (0, 1) else (
        reliable[0].team_id if reliable else None
    )
    segments: list[TrackSegment] = []
    for boundary_frame, next_team in boundaries:
        if boundary_frame > segment_start:
            segments.append(
                TrackSegment(
                    index=len(segments) + 1,
                    first_frame=segment_start,
                    last_frame=boundary_frame - 1,
                    team_id=segment_team,
                )
            )
        segment_start = boundary_frame
        segment_team = next_team
    segments.append(
        TrackSegment(
            index=len(segments) + 1,
            first_frame=segment_start,
            last_frame=last_frame,
            team_id=segment_team,
        )
    )
    return TrackSegmentation(track_id=track_id, segments=tuple(segments))
