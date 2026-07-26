from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .entity_corrections import EntityCorrectionSet, EntityRole, TeamAssignment
from .entity_review_manifest import EntityReviewManifest, ReviewObservation, ReviewTrack


@dataclass(frozen=True, slots=True)
class IdentityLink:
    track_id_a: int
    track_id_b: int
    score: float
    gap_frames: int
    position_score: float
    appearance_score: float
    decision: str

    def to_dict(self) -> dict:
        return {
            "track_ids": [self.track_id_a, self.track_id_b],
            "score": self.score,
            "gap_frames": self.gap_frames,
            "position_score": self.position_score,
            "appearance_score": self.appearance_score,
            "decision": self.decision,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IdentityLink":
        track_ids = data["track_ids"]
        return cls(
            track_id_a=int(track_ids[0]),
            track_id_b=int(track_ids[1]),
            score=float(data["score"]),
            gap_frames=int(data["gap_frames"]),
            position_score=float(data["position_score"]),
            appearance_score=float(data["appearance_score"]),
            decision=str(data["decision"]),
        )


@dataclass(frozen=True, slots=True)
class PhysicalIdentity:
    identity_id: int
    label: str
    track_ids: tuple[int, ...]
    role: EntityRole
    team: TeamAssignment
    first_frame: int
    last_frame: int

    def to_dict(self) -> dict:
        return {
            "identity_id": self.identity_id,
            "label": self.label,
            "track_ids": list(self.track_ids),
            "role": self.role.value,
            "team": self.team.value,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhysicalIdentity":
        return cls(
            identity_id=int(data["identity_id"]),
            label=str(data["label"]),
            track_ids=tuple(int(item) for item in data["track_ids"]),
            role=EntityRole(data["role"]),
            team=TeamAssignment(data["team"]),
            first_frame=int(data["first_frame"]),
            last_frame=int(data["last_frame"]),
        )


@dataclass(frozen=True, slots=True)
class EntityIdentitySet:
    source_video: str
    identities: tuple[PhysicalIdentity, ...]
    links: tuple[IdentityLink, ...] = ()
    schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_video": self.source_video,
            "identities": [item.to_dict() for item in self.identities],
            "links": [item.to_dict() for item in self.links],
        }

    def identity_for_track(self, track_id: int) -> PhysicalIdentity | None:
        return next(
            (item for item in self.identities if track_id in item.track_ids),
            None,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "EntityIdentitySet":
        version = int(data.get("schema_version", 1))
        if version != 1:
            raise ValueError(f"Niet-ondersteunde identiteitsversie: {version}")
        return cls(
            source_video=str(data["source_video"]),
            identities=tuple(PhysicalIdentity.from_dict(item) for item in data.get("identities", [])),
            links=tuple(IdentityLink.from_dict(item) for item in data.get("links", [])),
            schema_version=version,
        )


def save_entity_identities(identities: EntityIdentitySet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(identities.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_entity_identities(path: Path) -> EntityIdentitySet:
    return EntityIdentitySet.from_dict(json.loads(path.read_text(encoding="utf-8")))


def build_entity_identities(
    manifest: EntityReviewManifest,
    video_path: Path,
    corrections: EntityCorrectionSet | None = None,
    team_a_name: str = "Team A",
    team_b_name: str = "Team B",
    maximum_gap_seconds: float = 2.0,
) -> EntityIdentitySet:
    """Join only very likely tracker fragments into physical people.

    Uniform colour identifies a team, not a person. Automatic joins therefore
    require temporal succession plus spatial continuity; appearance is only a
    supporting signal. Ambiguous fragments deliberately remain separate.
    """
    if corrections is not None and corrections.source_video != manifest.source_video:
        raise ValueError("Correcties en reviewmanifest horen bij verschillende video's.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video kon niet worden geopend: {video_path}")
    try:
        frame_width = max(1.0, capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = max(1.0, capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        descriptors = {
            track.track_id: _track_descriptor(capture, track)
            for track in manifest.tracks
        }
    finally:
        capture.release()

    tracks = {item.track_id: item for item in manifest.tracks}
    parents = {track_id: track_id for track_id in tracks}

    def find(track_id: int) -> int:
        while parents[track_id] != track_id:
            parents[track_id] = parents[parents[track_id]]
            track_id = parents[track_id]
        return track_id

    def members(root: int) -> set[int]:
        return {track_id for track_id in tracks if find(track_id) == root}

    maximum_gap = max(1, round(maximum_gap_seconds * manifest.fps))
    candidates: list[IdentityLink] = []
    ordered = sorted(manifest.tracks, key=lambda item: (item.first_frame, item.track_id))
    for first in ordered:
        for second in ordered:
            if second.first_frame <= first.last_frame:
                continue
            gap = second.first_frame - first.last_frame - 1
            if gap > maximum_gap:
                break
            link = _score_link(
                first,
                second,
                descriptors[first.track_id],
                descriptors[second.track_id],
                corrections,
                frame_width,
                frame_height,
                gap,
            )
            if link is not None:
                candidates.append(link)

    links = sorted(candidates, key=lambda item: item.score, reverse=True)
    accepted: list[IdentityLink] = []
    reported: list[IdentityLink] = []
    for link in links:
        root_a, root_b = find(link.track_id_a), find(link.track_id_b)
        decision = link.decision
        if root_a != root_b and decision == "auto_merge":
            combined = members(root_a) | members(root_b)
            if _group_has_overlap(combined, tracks):
                decision = "rejected_overlap"
            else:
                parents[root_b] = root_a
                accepted.append(link)
        reported.append(
            IdentityLink(
                track_id_a=link.track_id_a,
                track_id_b=link.track_id_b,
                score=link.score,
                gap_frames=link.gap_frames,
                position_score=link.position_score,
                appearance_score=link.appearance_score,
                decision=decision,
            )
        )

    groups: dict[int, list[ReviewTrack]] = {}
    for track in manifest.tracks:
        groups.setdefault(find(track.track_id), []).append(track)
    provisional = []
    for group_tracks in groups.values():
        team, role = _group_assignment(group_tracks, corrections)
        provisional.append((min(item.first_frame for item in group_tracks), group_tracks, team, role))
    provisional.sort(key=lambda item: (item[0], min(track.track_id for track in item[1])))

    counters: dict[tuple[TeamAssignment, EntityRole], int] = {}
    identities = []
    for identity_id, (_first, group_tracks, team, role) in enumerate(provisional, start=1):
        key = (team, role)
        counters[key] = counters.get(key, 0) + 1
        label = _identity_label(team, role, counters[key], team_a_name, team_b_name)
        identities.append(
            PhysicalIdentity(
                identity_id=identity_id,
                label=label,
                track_ids=tuple(sorted(item.track_id for item in group_tracks)),
                role=role,
                team=team,
                first_frame=min(item.first_frame for item in group_tracks),
                last_frame=max(item.last_frame for item in group_tracks),
            )
        )
    return EntityIdentitySet(
        source_video=manifest.source_video,
        identities=tuple(identities),
        links=tuple(reported),
    )


def grouped_review_tracks(
    manifest: EntityReviewManifest,
    identities: EntityIdentitySet,
    maximum_observations: int = 9,
) -> tuple[ReviewTrack, ...]:
    """Create one review item per physical identity instead of per tracker ID."""
    tracks = {item.track_id: item for item in manifest.tracks}
    result = []
    for identity in identities.identities:
        members = [tracks[item] for item in identity.track_ids if item in tracks]
        if not members:
            continue
        representative = max(members, key=lambda item: (item.frames_seen, -item.track_id))
        observations = sorted(
            (observation for item in members for observation in item.observations),
            key=lambda item: item.frame_number,
        )
        if len(observations) > maximum_observations:
            indices = np.linspace(0, len(observations) - 1, maximum_observations).round().astype(int)
            observations = [observations[index] for index in indices]
        result.append(
            ReviewTrack(
                track_id=representative.track_id,
                first_frame=identity.first_frame,
                last_frame=identity.last_frame,
                frames_seen=sum(item.frames_seen for item in members),
                average_confidence=float(np.mean([item.average_confidence for item in members])),
                observations=tuple(observations),
                final_team_id=representative.final_team_id,
                team_votes_a=sum(item.team_votes_a for item in members),
                team_votes_b=sum(item.team_votes_b for item in members),
                team_agreement_ratio=representative.team_agreement_ratio,
                team_is_reliable=all(item.team_is_reliable for item in members),
            )
        )
    return tuple(result)


def _score_link(
    first: ReviewTrack,
    second: ReviewTrack,
    descriptor_a: np.ndarray | None,
    descriptor_b: np.ndarray | None,
    corrections: EntityCorrectionSet | None,
    frame_width: float,
    frame_height: float,
    gap: int,
) -> IdentityLink | None:
    if not _assignments_compatible(first, second, corrections):
        return None
    last = max(first.observations, key=lambda item: item.frame_number)
    earliest = min(second.observations, key=lambda item: item.frame_number)
    center_a, height_a = _box_geometry(last)
    center_b, height_b = _box_geometry(earliest)
    distance = float(np.linalg.norm(center_a - center_b))
    body_scale = max(12.0, (height_a + height_b) / 2.0)
    normalized_distance = distance / body_scale
    position_score = max(0.0, 1.0 - normalized_distance / 4.0)
    size_score = min(height_a, height_b) / max(height_a, height_b, 1.0)
    appearance = _cosine(descriptor_a, descriptor_b)
    temporal_score = max(0.0, 1.0 - gap / max(1.0, 2.0 * 30.0))
    score = 0.50 * position_score + 0.25 * appearance + 0.15 * size_score + 0.10 * temporal_score
    strong_continuity = normalized_distance <= 1.8 and size_score >= 0.65
    if score < 0.68 or normalized_distance > 3.0 or size_score < 0.50:
        return None
    decision = "auto_merge" if score >= 0.82 and strong_continuity and appearance >= 0.55 else "candidate"
    return IdentityLink(
        first.track_id,
        second.track_id,
        round(score, 4),
        gap,
        round(position_score, 4),
        round(appearance, 4),
        decision,
    )


def _track_descriptor(capture: cv2.VideoCapture, track: ReviewTrack) -> np.ndarray | None:
    features = []
    for observation in track.observations:
        capture.set(cv2.CAP_PROP_POS_FRAMES, observation.frame_number)
        success, frame = capture.read()
        if not success:
            continue
        feature = _appearance_feature(frame, observation.box)
        if feature is not None:
            features.append(feature)
    if not features:
        return None
    value = np.mean(features, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 0 else None


def _appearance_feature(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, int(x1)), min(width, int(x2))
    y1, y2 = max(0, int(y1)), min(height, int(y2))
    if x2 - x1 < 6 or y2 - y1 < 16:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    regions = ((0.15, 0.58), (0.50, 0.78), (0.74, 1.0))
    features = []
    for start, end in regions:
        region = hsv[int(start * len(hsv)):max(int(end * len(hsv)), 1)]
        if region.size == 0:
            return None
        hist = cv2.calcHist([region], [0, 1], None, [12, 6], [0, 180, 0, 256]).flatten()
        total = float(hist.sum())
        features.append(hist / total if total > 0 else hist)
    return np.concatenate(features).astype(np.float32)


def _assignments_compatible(
    first: ReviewTrack,
    second: ReviewTrack,
    corrections: EntityCorrectionSet | None,
) -> bool:
    a = corrections.get(first.track_id) if corrections else None
    b = corrections.get(second.track_id) if corrections else None
    if a is not None and b is not None:
        if a.excluded != b.excluded or a.role != b.role or a.team != b.team:
            return False
    if first.team_is_reliable and second.team_is_reliable:
        return first.final_team_id == second.final_team_id
    return True


def _group_has_overlap(track_ids: set[int], tracks: dict[int, ReviewTrack]) -> bool:
    values = sorted((tracks[item] for item in track_ids), key=lambda item: item.first_frame)
    return any(left.last_frame >= right.first_frame for left, right in zip(values, values[1:]))


def _group_assignment(
    tracks: list[ReviewTrack], corrections: EntityCorrectionSet | None
) -> tuple[TeamAssignment, EntityRole]:
    manual = [corrections.get(item.track_id) for item in tracks] if corrections else []
    manual = [item for item in manual if item is not None]
    if manual:
        team = manual[0].team if all(item.team == manual[0].team for item in manual) else TeamAssignment.UNKNOWN
        role = manual[0].role if all(item.role == manual[0].role for item in manual) else EntityRole.UNKNOWN
        return team, role
    reliable = [item.final_team_id for item in tracks if item.team_is_reliable]
    if reliable and all(item == reliable[0] for item in reliable):
        team = TeamAssignment.TEAM_A if reliable[0] == 0 else TeamAssignment.TEAM_B
        return team, EntityRole.PLAYER
    return TeamAssignment.UNKNOWN, EntityRole.UNKNOWN


def _identity_label(
    team: TeamAssignment,
    role: EntityRole,
    number: int,
    team_a_name: str,
    team_b_name: str,
) -> str:
    team_name = team_a_name if team == TeamAssignment.TEAM_A else team_b_name if team == TeamAssignment.TEAM_B else "Onbekend"
    if role == EntityRole.GOALKEEPER:
        return f"{team_name} - Keeper {number}"
    if role == EntityRole.PLAYER:
        return f"{team_name} - Speler {number}"
    if role == EntityRole.REFEREE:
        return f"Scheidsrechter {number}"
    if role in (EntityRole.STAFF, EntityRole.SPECTATOR):
        return f"Uitgesloten persoon {number}"
    return f"Onbekende persoon {number}"


def _box_geometry(observation: ReviewObservation) -> tuple[np.ndarray, float]:
    x1, y1, x2, y2 = observation.box
    return np.array(((x1 + x2) / 2.0, (y1 + y2) / 2.0), dtype=np.float32), max(1.0, y2 - y1)


def _cosine(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None:
        return 0.5
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return max(0.0, min(1.0, float(np.dot(first, second)) / denominator)) if denominator else 0.5
