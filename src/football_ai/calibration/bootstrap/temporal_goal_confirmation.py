from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from football_ai.calibration.bootstrap.goal_detection import GoalCandidate, GoalDetection


@dataclass(frozen=True, slots=True)
class ConfirmedGoal:
    representative: GoalCandidate
    supporting_frame_count: int
    analyzed_frame_count: int
    support_ratio: float
    position_spread: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "representative": self.representative.to_dict(),
            "supporting_frame_count": self.supporting_frame_count,
            "analyzed_frame_count": self.analyzed_frame_count,
            "support_ratio": self.support_ratio,
            "position_spread": self.position_spread,
            "confidence": self.confidence,
        }


def confirm_goals_temporally(
    detections: list[GoalDetection],
    frame_sizes: list[tuple[int, int]],
    minimum_support_ratio: float = 0.35,
) -> tuple[ConfirmedGoal, ...]:
    """Bevestig doelen die op vergelijkbare beeldpositie in meerdere frames staan."""
    if len(detections) != len(frame_sizes):
        raise ValueError("Detections en frame_sizes moeten even lang zijn.")
    if not detections:
        return ()
    tracks: list[list[tuple[int, GoalCandidate, np.ndarray]]] = []
    for frame_index, (detection, size) in enumerate(zip(detections, frame_sizes)):
        width, height = size
        used_tracks: set[int] = set()
        for candidate in detection.candidates:
            signature = _signature(candidate, width, height)
            best_track = None
            best_distance = float("inf")
            for track_index, track in enumerate(tracks):
                if track_index in used_tracks:
                    continue
                distance = float(np.linalg.norm(signature - np.mean([item[2] for item in track], axis=0)))
                if distance < 0.13 and distance < best_distance:
                    best_track, best_distance = track_index, distance
            if best_track is None:
                tracks.append([(frame_index, candidate, signature)])
                used_tracks.add(len(tracks) - 1)
            else:
                tracks[best_track].append((frame_index, candidate, signature))
                used_tracks.add(best_track)

    confirmed: list[ConfirmedGoal] = []
    analyzed_count = len(detections)
    for track in tracks:
        frame_count = len({item[0] for item in track})
        support_ratio = frame_count / analyzed_count
        if frame_count < 2 or support_ratio < minimum_support_ratio:
            continue
        signatures = np.vstack([item[2] for item in track])
        center = np.mean(signatures, axis=0)
        distances = np.linalg.norm(signatures - center, axis=1)
        representative_index = int(np.argmin(distances))
        representative = track[representative_index][1]
        spread = float(np.sqrt(np.mean(np.square(distances))))
        mean_visual = float(np.mean([item[1].confidence for item in track]))
        confidence = float(
            np.clip(0.52 * support_ratio + 0.34 * mean_visual + 0.14 * max(0.0, 1.0 - spread / 0.13), 0.0, 1.0)
        )
        confirmed.append(
            ConfirmedGoal(
                representative=representative,
                supporting_frame_count=frame_count,
                analyzed_frame_count=analyzed_count,
                support_ratio=support_ratio,
                position_spread=spread,
                confidence=confidence,
            )
        )
    return tuple(sorted(confirmed, key=lambda item: item.confidence, reverse=True))


def _signature(candidate: GoalCandidate, width: int, height: int) -> np.ndarray:
    center = candidate.center_ground
    goal_width = abs(candidate.right_ground[0] - candidate.left_ground[0])
    goal_height = (
        abs(candidate.left_ground[1] - candidate.left_top[1])
        + abs(candidate.right_ground[1] - candidate.right_top[1])
    ) / 2.0
    return np.asarray(
        [
            center[0] / width,
            center[1] / height,
            np.log(max(goal_width / width, 1e-5)),
            np.log(max(goal_height / height, 1e-5)),
        ],
        dtype=np.float64,
    )
