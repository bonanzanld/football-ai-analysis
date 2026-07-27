from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np

from football_ai.classification.participant_classifier import (
    ParticipantAssessment,
    ParticipantClassifier,
    ParticipantDecision,
    ParticipantEvidence,
)
from football_ai.tracking.entity_corrections import (
    EntityCorrectionSet,
    EntityRole,
)
from football_ai.tracking.entity_review_manifest import EntityReviewManifest, ReviewTrack


@dataclass(frozen=True, slots=True)
class ParticipantAnalysisReport:
    source_video: str
    assessments: tuple[ParticipantAssessment, ...]
    schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_video": self.source_video,
            "policy": "review_only_no_automatic_exclusion",
            "assessments": [
                {
                    "track_id": item.track_id,
                    "segment_index": item.segment_index,
                    "score": item.score,
                    "decision": item.decision.value,
                    "reasons": list(item.reasons),
                    "evidence": asdict(item.evidence),
                }
                for item in self.assessments
            ],
        }


def analyze_participants(
    manifest: EntityReviewManifest,
    corrections: EntityCorrectionSet | None = None,
) -> ParticipantAnalysisReport:
    classifier = ParticipantClassifier()
    assessments = []
    for track in manifest.tracks:
        evidence = ParticipantEvidence(
            track_id=track.track_id,
            segment_index=track.segment_index,
            team_reliability=(
                track.team_agreement_ratio if track.team_is_reliable else 0.0
            ),
            # Een lage of instabiele teamtoewijzing is hier een conservatieve
            # proxy. Een later uniformmodel kan deze waarde verder versterken.
            team_uniform_distance=_team_outlier_proxy(track),
            player_group_proximity=_player_group_proximity(track, manifest),
            relative_activity=_relative_activity(track, manifest),
            track_stability=min(
                1.0,
                track.frames_seen / max(manifest.fps * 5.0, 1.0),
            ),
        )
        assessment = classifier.assess(evidence)
        correction = (
            corrections.get(track.track_id, track.segment_index)
            if corrections is not None
            else None
        )
        if correction is not None and correction.role == EntityRole.REFEREE:
            assessment = replace(
                assessment,
                score=1.0,
                decision=ParticipantDecision.CONFIRMED_REFEREE,
                reasons=("handmatig bevestigd als scheidsrechter",),
            )
        elif correction is not None and correction.excluded:
            assessment = replace(
                assessment,
                score=1.0,
                decision=ParticipantDecision.CONFIRMED_EXCLUDED,
                reasons=("handmatig uitgesloten van de wedstrijdanalyse",),
            )
        assessments.append(assessment)
    assessments.sort(key=lambda item: item.score, reverse=True)
    return ParticipantAnalysisReport(manifest.source_video, tuple(assessments))


def review_candidates(
    assessments: tuple[ParticipantAssessment, ...],
) -> tuple[ParticipantAssessment, ...]:
    return tuple(
        item
        for item in assessments
        if item.decision
        in (ParticipantDecision.REFEREE_REVIEW, ParticipantDecision.OUTSIDER_REVIEW)
    )


def save_participant_analysis(report: ParticipantAnalysisReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _team_outlier_proxy(track: ReviewTrack) -> float:
    if track.team_is_reliable:
        return max(0.0, 1.0 - track.team_agreement_ratio)
    total_votes = track.team_votes_a + track.team_votes_b
    if total_votes == 0:
        return 0.65
    return max(0.45, 1.0 - track.team_agreement_ratio)


def _player_group_proximity(
    track: ReviewTrack,
    manifest: EntityReviewManifest,
) -> float:
    values = []
    for observation in track.observations:
        player_points = _nearby_reliable_player_points(
            manifest,
            track,
            observation.frame_number,
        )
        if len(player_points) < 3:
            continue
        point = np.asarray(_footpoint(observation.box), dtype=np.float64)
        points = np.asarray(player_points, dtype=np.float64)
        low = np.percentile(points, 10.0, axis=0)
        high = np.percentile(points, 90.0, axis=0)
        span = np.maximum(high - low, np.asarray([40.0, 30.0]))
        margin = 0.20 * span
        outside = np.maximum(low - margin - point, 0.0) + np.maximum(
            point - high - margin,
            0.0,
        )
        distance = float(np.linalg.norm(outside / span))
        values.append(max(0.0, 1.0 - distance / 0.75))
    return float(np.median(values)) if values else 0.0


def _relative_activity(track: ReviewTrack, manifest: EntityReviewManifest) -> float:
    relative_positions = []
    for observation in track.observations:
        players = _nearby_reliable_player_points(
            manifest,
            track,
            observation.frame_number,
        )
        if len(players) < 3:
            continue
        center = np.median(np.asarray(players, dtype=np.float64), axis=0)
        point = np.asarray(_footpoint(observation.box), dtype=np.float64)
        relative_positions.append((observation.frame_number, point - center))
    if len(relative_positions) < 2:
        return 0.0
    speeds = []
    for (first_frame, first), (last_frame, last) in zip(
        relative_positions,
        relative_positions[1:],
    ):
        elapsed = max((last_frame - first_frame) / max(manifest.fps, 1e-6), 1e-3)
        speeds.append(float(np.linalg.norm(last - first)) / elapsed)
    median_speed = float(np.median(speeds))
    return max(0.0, min(1.0, median_speed / 45.0))


def _nearby_reliable_player_points(
    manifest: EntityReviewManifest,
    candidate: ReviewTrack,
    frame_number: int,
) -> list[tuple[float, float]]:
    points = []
    maximum_gap = max(2, round(manifest.fps * 0.6))
    for other in manifest.tracks:
        if other is candidate or not other.team_is_reliable:
            continue
        nearest = min(
            other.observations,
            key=lambda item: abs(item.frame_number - frame_number),
            default=None,
        )
        if nearest is not None and abs(nearest.frame_number - frame_number) <= maximum_gap:
            points.append(_footpoint(nearest.box))
    return points


def _footpoint(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)
