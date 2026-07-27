from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from football_ai.classification.color_features import extract_shirt_feature
from football_ai.classification.goalkeeper_classifier import (
    GoalkeeperAssessment,
    GoalkeeperClassifier,
    GoalkeeperEvidence,
    GoalLineReference,
    defensive_depth_score,
    goal_line_proximity_score,
)
from football_ai.tracking.entity_review_manifest import EntityReviewManifest, ReviewTrack


class GoalFrameReference(Protocol):
    frame_number: int
    goal_id: str
    first_ground: tuple[float, float]
    second_ground: tuple[float, float]


@dataclass(frozen=True, slots=True)
class GoalkeeperAnalysisReport:
    source_video: str
    assessments: tuple[GoalkeeperAssessment, ...]
    goal_evidence_available: bool
    schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_video": self.source_video,
            "goal_evidence_available": self.goal_evidence_available,
            "assessments": [
                {
                    "track_id": item.track_id,
                    "team_id": item.team_id,
                    "score": item.score,
                    "decision": item.decision.value,
                    "reasons": list(item.reasons),
                    "evidence": asdict(item.evidence),
                }
                for item in self.assessments
            ],
        }


def analyze_goalkeeper_candidates(
    video_path: Path,
    manifest: EntityReviewManifest,
    goal_seeds: tuple[GoalFrameReference, ...] = (),
) -> GoalkeeperAnalysisReport:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video kon niet worden geopend: {video_path}")
    frame_width = max(1.0, capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = max(1.0, capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    descriptors: dict[tuple[int, int | None], np.ndarray] = {}
    try:
        for track in manifest.tracks:
            descriptor = _track_shirt_descriptor(capture, track)
            if descriptor is not None:
                descriptors[(track.track_id, track.segment_index)] = descriptor
    finally:
        capture.release()

    team_prototypes = _team_prototypes(manifest, descriptors)
    team_scales = _team_distance_scales(manifest, descriptors, team_prototypes)
    assessments = []
    classifier = GoalkeeperClassifier()
    for track in manifest.tracks:
        key = (track.track_id, track.segment_index)
        descriptor = descriptors.get(key)
        team_id = track.final_team_id if track.final_team_id in (0, 1) else None
        uniform_score = _uniform_outlier_score(
            descriptor,
            team_prototypes.get(team_id),
            team_scales.get(team_id, 0.05),
        )
        goal_score, depth_score = _spatial_evidence(
            track,
            manifest,
            goal_seeds,
            frame_width,
            frame_height,
        )
        stability = min(1.0, track.frames_seen / max(manifest.fps * 5.0, 1.0))
        movement_confinement = _relative_movement_confinement_score(
            track,
            manifest,
            frame_width,
            frame_height,
        )
        assessments.append(
            classifier.assess(
                GoalkeeperEvidence(
                    track_id=track.track_id,
                    team_id=team_id,
                    uniform_outlier_score=uniform_score,
                    goal_proximity_score=goal_score,
                    defensive_depth_score=depth_score,
                    track_stability_score=stability,
                    movement_confinement_score=movement_confinement,
                )
            )
        )
    assessments.sort(key=lambda item: item.score, reverse=True)
    spatial_evidence_available = any(
        item.evidence.goal_proximity_score > 0.0
        or item.evidence.defensive_depth_score > 0.0
        for item in assessments
    )
    return GoalkeeperAnalysisReport(
        source_video=manifest.source_video,
        assessments=tuple(assessments),
        goal_evidence_available=spatial_evidence_available,
    )


def shortlist_goalkeeper_assessments(
    assessments: tuple[GoalkeeperAssessment, ...],
    maximum_per_team: int = 3,
) -> tuple[GoalkeeperAssessment, ...]:
    if maximum_per_team < 1:
        raise ValueError("maximum_per_team moet minimaal 1 zijn.")
    counts: dict[int | None, int] = {}
    shortlisted = []
    for assessment in assessments:
        if assessment.decision.value == "player":
            continue
        key = assessment.team_id
        if counts.get(key, 0) >= maximum_per_team:
            continue
        counts[key] = counts.get(key, 0) + 1
        shortlisted.append(assessment)
    return tuple(shortlisted)


def save_goalkeeper_analysis(report: GoalkeeperAnalysisReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _track_shirt_descriptor(
    capture: cv2.VideoCapture,
    track: ReviewTrack,
) -> np.ndarray | None:
    features = []
    for observation in track.observations:
        capture.set(cv2.CAP_PROP_POS_FRAMES, observation.frame_number)
        success, frame = capture.read()
        if not success:
            continue
        feature = extract_shirt_feature(frame, np.asarray(observation.box))
        if feature is not None:
            features.append(feature)
    if not features:
        return None
    descriptor = np.median(np.stack(features), axis=0).astype(np.float32)
    total = float(descriptor.sum())
    return descriptor / total if total > 0.0 else None


def _team_prototypes(
    manifest: EntityReviewManifest,
    descriptors: dict[tuple[int, int | None], np.ndarray],
) -> dict[int, np.ndarray]:
    result = {}
    for team_id in (0, 1):
        values = [
            descriptors[(track.track_id, track.segment_index)]
            for track in manifest.tracks
            if track.final_team_id == team_id
            and track.team_is_reliable
            and (track.track_id, track.segment_index) in descriptors
        ]
        if len(values) >= 2:
            prototype = np.median(np.stack(values), axis=0).astype(np.float32)
            result[team_id] = prototype / max(float(prototype.sum()), 1e-6)
    return result


def _team_distance_scales(
    manifest: EntityReviewManifest,
    descriptors: dict[tuple[int, int | None], np.ndarray],
    prototypes: dict[int, np.ndarray],
) -> dict[int, float]:
    scales = {}
    for team_id, prototype in prototypes.items():
        distances = [
            float(np.linalg.norm(descriptor - prototype))
            for key, descriptor in descriptors.items()
            if next(
                (
                    track.final_team_id
                    for track in manifest.tracks
                    if (track.track_id, track.segment_index) == key
                ),
                None,
            )
            == team_id
        ]
        if distances:
            scales[team_id] = max(float(np.median(distances)) * 2.5, 0.035)
    return scales


def _uniform_outlier_score(
    descriptor: np.ndarray | None,
    prototype: np.ndarray | None,
    scale: float,
) -> float:
    if descriptor is None or prototype is None:
        return 0.0
    distance = float(np.linalg.norm(descriptor - prototype))
    return min(1.0, distance / max(scale, 1e-6))


def _spatial_evidence(
    track: ReviewTrack,
    manifest: EntityReviewManifest,
    goal_seeds: tuple[GoalFrameReference, ...],
    frame_width: float,
    frame_height: float,
) -> tuple[float, float]:
    if not goal_seeds or track.final_team_id not in (0, 1):
        return 0.0, 0.0
    relevant_goals = tuple(
        reference
        for reference in goal_seeds
        if getattr(reference, "defending_team_id", track.final_team_id)
        == track.final_team_id
    )
    if not relevant_goals:
        return 0.0, 0.0
    maximum_frame_gap = max(1, round(manifest.fps * 2.0))
    maximum_distance = 0.12 * float(np.hypot(frame_width, frame_height))
    goal_scores = []
    depth_scores = []
    for observation in track.observations:
        seed = min(
            relevant_goals,
            key=lambda item: abs(item.frame_number - observation.frame_number),
        )
        if abs(seed.frame_number - observation.frame_number) > maximum_frame_gap:
            continue
        goal = GoalLineReference(seed.goal_id, seed.first_ground, seed.second_ground)
        footpoint = _footpoint(observation.box)
        goal_scores.append(goal_line_proximity_score(footpoint, goal, maximum_distance))
        teammates = []
        for other in manifest.tracks:
            if other is track or other.final_team_id != track.final_team_id:
                continue
            nearest = min(
                other.observations,
                key=lambda item: abs(item.frame_number - observation.frame_number),
                default=None,
            )
            if nearest is not None and abs(nearest.frame_number - observation.frame_number) <= 15:
                teammates.append(_footpoint(nearest.box))
        if teammates:
            depth_scores.append(defensive_depth_score(footpoint, teammates, goal))
    return (
        float(np.mean(goal_scores)) if goal_scores else 0.0,
        float(np.mean(depth_scores)) if depth_scores else 0.0,
    )


def _footpoint(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def _relative_movement_confinement_score(
    track: ReviewTrack,
    manifest: EntityReviewManifest,
    frame_width: float,
    frame_height: float,
) -> float:
    """Meet beweging ten opzichte van ploeggenoten, zodat camerapan minder meetelt."""

    if track.final_team_id not in (0, 1) or len(track.observations) < 2:
        return 0.0
    relative_positions = []
    for observation in track.observations:
        teammates = []
        for other in manifest.tracks:
            if other is track or other.final_team_id != track.final_team_id:
                continue
            nearest = min(
                other.observations,
                key=lambda item: abs(item.frame_number - observation.frame_number),
                default=None,
            )
            if nearest is not None and abs(nearest.frame_number - observation.frame_number) <= 15:
                teammates.append(_footpoint(nearest.box))
        if len(teammates) < 2:
            continue
        team_center = np.median(np.asarray(teammates, dtype=np.float64), axis=0)
        footpoint = np.asarray(_footpoint(observation.box), dtype=np.float64)
        relative_positions.append((observation.frame_number, footpoint - team_center))
    if len(relative_positions) < 2:
        return 0.0
    diagonal = max(float(np.hypot(frame_width, frame_height)), 1.0)
    speeds = []
    for (first_frame, first), (second_frame, second) in zip(
        relative_positions,
        relative_positions[1:],
    ):
        elapsed = max((second_frame - first_frame) / max(manifest.fps, 1e-6), 1e-3)
        speeds.append(float(np.linalg.norm(second - first)) / diagonal / elapsed)
    median_speed = float(np.median(speeds))
    return max(0.0, min(1.0, 1.0 - median_speed / 0.10))
