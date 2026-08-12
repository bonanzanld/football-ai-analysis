from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np

from football_ai.classification.goalkeeper_analysis import (
    _relative_movement_confinement_score,
    _spatial_evidence,
    _uniform_outlier_score,
    shortlist_goalkeeper_assessments,
    load_tracked_goal_references,
)
from football_ai.classification.goalkeeper_goal_reference import GoalkeeperGoalReference
from football_ai.tracking.entity_review_manifest import (
    EntityReviewManifest,
    ReviewObservation,
    ReviewTrack,
)
from football_ai.classification.goalkeeper_classifier import (
    GoalkeeperAssessment,
    GoalkeeperDecision,
    GoalkeeperEvidence,
)


class GoalkeeperAnalysisTests(unittest.TestCase):
    def test_loads_dynamic_goal_tracking_as_frame_references(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "goals.json"
            path.write_text(json.dumps({"records": [{
                "frame_number": 120,
                "goal": "B",
                "ground_points": [[10, 20], [40, 22]],
            }]}), encoding="utf-8")
            result = load_tracked_goal_references(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].goal_id, "B")
        self.assertEqual(result[0].first_ground, (10.0, 20.0))

    def test_matching_uniform_has_low_outlier_score(self) -> None:
        feature = np.asarray([0.5, 0.5], dtype=np.float32)
        self.assertEqual(_uniform_outlier_score(feature, feature, 0.1), 0.0)

    def test_different_uniform_has_high_outlier_score(self) -> None:
        feature = np.asarray([1.0, 0.0], dtype=np.float32)
        prototype = np.asarray([0.0, 1.0], dtype=np.float32)
        self.assertEqual(_uniform_outlier_score(feature, prototype, 0.1), 1.0)

    def test_missing_team_prototype_does_not_guess(self) -> None:
        feature = np.asarray([1.0, 0.0], dtype=np.float32)
        self.assertEqual(_uniform_outlier_score(feature, None, 0.1), 0.0)

    def test_shortlist_limits_review_work_per_team(self) -> None:
        values = tuple(
            GoalkeeperAssessment(
                track_id=index,
                team_id=0,
                score=1.0 - index / 10.0,
                decision=GoalkeeperDecision.REVIEW,
                evidence=GoalkeeperEvidence(index, 0, 0.8, 0.0, 0.0, 0.8),
                reasons=(),
            )
            for index in range(5)
        )
        result = shortlist_goalkeeper_assessments(values, maximum_per_team=2)
        self.assertEqual([item.track_id for item in result], [0, 1])

    def test_spatial_evidence_uses_only_goal_defended_by_tracks_team(self) -> None:
        track = ReviewTrack(
            track_id=7,
            first_frame=30,
            last_frame=30,
            frames_seen=1,
            average_confidence=0.9,
            observations=(ReviewObservation(30, (90.0, 50.0, 110.0, 100.0)),),
            final_team_id=0,
        )
        manifest = EntityReviewManifest("test.mp4", 30.0, (track,))
        wrong_team_goal = GoalkeeperGoalReference(
            frame_number=30,
            time_seconds=1.0,
            defending_team_id=1,
            first_post=(90.0, 100.0),
            second_post=(110.0, 100.0),
        )
        goal_score, depth_score = _spatial_evidence(
            track,
            manifest,
            (wrong_team_goal,),
            1280.0,
            720.0,
        )
        self.assertEqual(goal_score, 0.0)
        self.assertEqual(depth_score, 0.0)

    def test_unassigned_track_can_use_goal_proximity_but_not_team_depth(self) -> None:
        track = ReviewTrack(
            track_id=8,
            first_frame=30,
            last_frame=30,
            frames_seen=1,
            average_confidence=0.9,
            observations=(ReviewObservation(30, (90.0, 50.0, 110.0, 100.0)),),
            final_team_id=None,
        )
        manifest = EntityReviewManifest("test.mp4", 30.0, (track,))
        goal = GoalkeeperGoalReference(30, 1.0, 0, (90.0, 100.0), (110.0, 100.0))

        goal_score, depth_score = _spatial_evidence(track, manifest, (goal,), 1280.0, 720.0)

        self.assertEqual(goal_score, 1.0)
        self.assertEqual(depth_score, 0.0)

    def test_relative_movement_compensates_shared_camera_motion(self) -> None:
        def make_track(track_id: int, x_offset: float) -> ReviewTrack:
            return ReviewTrack(
                track_id=track_id,
                first_frame=0,
                last_frame=30,
                frames_seen=31,
                average_confidence=0.9,
                observations=(
                    ReviewObservation(0, (x_offset, 50.0, x_offset + 20.0, 100.0)),
                    ReviewObservation(30, (x_offset + 100.0, 50.0, x_offset + 120.0, 100.0)),
                ),
                final_team_id=0,
            )

        candidate = make_track(1, 100.0)
        teammate_a = make_track(2, 200.0)
        teammate_b = make_track(3, 300.0)
        manifest = EntityReviewManifest(
            "test.mp4",
            30.0,
            (candidate, teammate_a, teammate_b),
        )
        score = _relative_movement_confinement_score(
            candidate,
            manifest,
            1280.0,
            720.0,
        )
        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
