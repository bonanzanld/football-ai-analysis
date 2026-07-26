from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from football_ai.classification.team_consensus import TeamConsensusResult

from .track_state import TrackState
from .track_segmentation import TrackSegmentation


@dataclass(frozen=True, slots=True)
class ReviewObservation:
    frame_number: int
    box: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "box": list(self.box),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewObservation":
        values = tuple(float(value) for value in data["box"])
        if len(values) != 4:
            raise ValueError("Een review-detectiekader moet vier waarden bevatten.")
        return cls(frame_number=int(data["frame_number"]), box=values)


@dataclass(frozen=True, slots=True)
class ReviewTrack:
    track_id: int
    first_frame: int
    last_frame: int
    frames_seen: int
    average_confidence: float
    observations: tuple[ReviewObservation, ...]
    segment_index: int | None = None
    final_team_id: int | None = None
    team_votes_a: int = 0
    team_votes_b: int = 0
    team_agreement_ratio: float = 0.0
    team_is_reliable: bool = False

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "segment_index": self.segment_index,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frames_seen": self.frames_seen,
            "average_confidence": self.average_confidence,
            "final_team_id": self.final_team_id,
            "team_votes": {
                "team_a": self.team_votes_a,
                "team_b": self.team_votes_b,
            },
            "team_agreement_ratio": self.team_agreement_ratio,
            "team_is_reliable": self.team_is_reliable,
            "observations": [item.to_dict() for item in self.observations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewTrack":
        votes = data.get("team_votes", {})
        team_id = data.get("final_team_id")
        return cls(
            track_id=int(data["track_id"]),
            segment_index=(
                int(data["segment_index"])
                if data.get("segment_index") is not None
                else None
            ),
            first_frame=int(data["first_frame"]),
            last_frame=int(data["last_frame"]),
            frames_seen=int(data["frames_seen"]),
            average_confidence=float(data.get("average_confidence", 0.0)),
            observations=tuple(
                ReviewObservation.from_dict(item)
                for item in data.get("observations", [])
            ),
            final_team_id=int(team_id) if team_id is not None else None,
            team_votes_a=int(votes.get("team_a", 0)),
            team_votes_b=int(votes.get("team_b", 0)),
            team_agreement_ratio=float(data.get("team_agreement_ratio", 0.0)),
            team_is_reliable=bool(data.get("team_is_reliable", False)),
        )


@dataclass(frozen=True, slots=True)
class EntityReviewManifest:
    source_video: str
    fps: float
    tracks: tuple[ReviewTrack, ...]
    schema_version: int = 2

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_video": self.source_video,
            "fps": self.fps,
            "tracks": [track.to_dict() for track in self.tracks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EntityReviewManifest":
        version = int(data.get("schema_version", 1))
        if version not in (1, 2):
            raise ValueError(f"Niet-ondersteunde reviewmanifestversie: {version}")
        return cls(
            source_video=str(data["source_video"]),
            fps=float(data["fps"]),
            tracks=tuple(ReviewTrack.from_dict(item) for item in data.get("tracks", [])),
            schema_version=2,
        )


def build_entity_review_manifest(
    source_video: str,
    fps: float,
    tracks: list[TrackState],
    maximum_observations_per_track: int = 7,
    team_consensus: dict[int, TeamConsensusResult] | None = None,
    track_segmentations: dict[int, TrackSegmentation] | None = None,
) -> EntityReviewManifest:
    if not source_video.strip():
        raise ValueError("source_video mag niet leeg zijn.")
    if fps <= 0:
        raise ValueError("fps moet groter dan nul zijn.")
    if maximum_observations_per_track < 1:
        raise ValueError("Minimaal één reviewwaarneming per track vereist.")

    review_tracks_list: list[ReviewTrack] = []
    for track in sorted(tracks, key=lambda item: item.track_id):
        segmentation = (track_segmentations or {}).get(track.track_id)
        if segmentation is None:
            review_tracks_list.append(
                _build_review_track(
                    track,
                    maximum_observations_per_track,
                    (team_consensus or {}).get(track.track_id),
                )
            )
            continue
        for segment in segmentation.segments:
            review_tracks_list.append(
                _build_review_track(
                    track,
                    maximum_observations_per_track,
                    (team_consensus or {}).get(track.track_id),
                    segment_index=segment.index,
                    first_frame=segment.first_frame,
                    last_frame=segment.last_frame,
                    team_override=segment.team_id,
                )
            )
    review_tracks = tuple(review_tracks_list)
    return EntityReviewManifest(source_video=source_video, fps=fps, tracks=review_tracks)


def save_entity_review_manifest(manifest: EntityReviewManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_entity_review_manifest(path: Path) -> EntityReviewManifest:
    return EntityReviewManifest.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _build_review_track(
    track: TrackState,
    maximum: int,
    consensus: TeamConsensusResult | None,
    segment_index: int | None = None,
    first_frame: int | None = None,
    last_frame: int | None = None,
    team_override: int | None = None,
) -> ReviewTrack:
    if len(track.observation_frames) != len(track.boxes):
        raise ValueError(
            f"Track {track.track_id} heeft niet evenveel frames als detectiekaders."
        )

    eligible_indices = [
        index
        for index, frame in enumerate(track.observation_frames)
        if (first_frame is None or frame >= first_frame)
        and (last_frame is None or frame <= last_frame)
    ]
    spread = _spread_indices(len(eligible_indices), maximum)
    indices = [eligible_indices[index] for index in spread]
    observations = tuple(
        ReviewObservation(
            frame_number=track.observation_frames[index],
            box=track.boxes[index],
        )
        for index in indices
    )
    return ReviewTrack(
        track_id=track.track_id,
        segment_index=segment_index,
        first_frame=first_frame if first_frame is not None else track.first_frame,
        last_frame=last_frame if last_frame is not None else track.last_frame,
        frames_seen=len(eligible_indices),
        average_confidence=(
            sum(track.confidences[index] for index in eligible_indices)
            / len(eligible_indices)
            if eligible_indices and len(track.confidences) == len(track.boxes)
            else track.average_confidence
        ),
        observations=observations,
        final_team_id=(
            team_override
            if team_override is not None
            else consensus.team_id if consensus is not None else None
        ),
        team_votes_a=consensus.votes_team_a if consensus is not None else 0,
        team_votes_b=consensus.votes_team_b if consensus is not None else 0,
        team_agreement_ratio=(
            consensus.agreement_ratio if consensus is not None else 0.0
        ),
        team_is_reliable=consensus.is_reliable if consensus is not None else False,
    )


def _spread_indices(length: int, maximum: int) -> tuple[int, ...]:
    if length <= maximum:
        return tuple(range(length))
    if maximum == 1:
        return (length // 2,)
    return tuple(
        round(index * (length - 1) / (maximum - 1))
        for index in range(maximum)
    )
