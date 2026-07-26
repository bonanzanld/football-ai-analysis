from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football_ai.tracking.entity_review_manifest import (
    build_entity_review_manifest,
    load_entity_review_manifest,
    save_entity_review_manifest,
)
from football_ai.tracking.track_state import TrackState
from football_ai.classification.team_consensus import TeamConsensusResult
from football_ai.tracking.track_segmentation import TrackSegment, TrackSegmentation


class EntityReviewManifestTests(unittest.TestCase):
    def test_observations_are_spread_over_complete_track(self) -> None:
        track = TrackState(track_id=4, first_frame=10, last_frame=19, frames_seen=10)
        track.observation_frames = list(range(10, 20))
        track.boxes = [(float(i), 1.0, float(i + 2), 5.0) for i in range(10)]
        track.confidences = [0.8, 1.0]

        manifest = build_entity_review_manifest(
            source_video="wedstrijd.mp4",
            fps=30.0,
            tracks=[track],
            maximum_observations_per_track=3,
        )

        review_track = manifest.tracks[0]
        self.assertEqual(
            [item.frame_number for item in review_track.observations],
            [10, 14, 19],
        )
        self.assertAlmostEqual(review_track.average_confidence, 0.9)

    def test_team_consensus_is_included_for_manual_review(self) -> None:
        track = TrackState(track_id=4, first_frame=10, last_frame=10, frames_seen=1)
        track.observation_frames = [10]
        track.boxes = [(1.0, 2.0, 3.0, 4.0)]
        consensus = TeamConsensusResult(
            track_id=4,
            team_id=0,
            votes_team_a=80,
            votes_team_b=20,
            total_votes=100,
            agreement_ratio=0.8,
            is_reliable=True,
        )

        manifest = build_entity_review_manifest(
            "wedstrijd.mp4",
            30.0,
            [track],
            team_consensus={4: consensus},
        )

        review_track = manifest.tracks[0]
        self.assertEqual(review_track.final_team_id, 0)
        self.assertEqual(review_track.team_votes_a, 80)
        self.assertTrue(review_track.team_is_reliable)

    def test_mismatched_observations_are_rejected(self) -> None:
        track = TrackState(track_id=9, first_frame=1, last_frame=1, frames_seen=1)
        track.observation_frames = [1]

        with self.assertRaises(ValueError):
            build_entity_review_manifest("wedstrijd.mp4", 30.0, [track])

    def test_json_roundtrip_preserves_consensus(self) -> None:
        track = TrackState(track_id=2, first_frame=3, last_frame=3, frames_seen=1)
        track.observation_frames = [3]
        track.boxes = [(1.0, 2.0, 3.0, 4.0)]
        manifest = build_entity_review_manifest("wedstrijd.mp4", 25.0, [track])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            save_entity_review_manifest(manifest, path)
            loaded = load_entity_review_manifest(path)

        self.assertEqual(loaded, manifest)

    def test_split_track_becomes_two_independent_review_items(self) -> None:
        track = TrackState(track_id=23, first_frame=0, last_frame=9, frames_seen=10)
        track.observation_frames = list(range(10))
        track.boxes = [(float(i), 1.0, float(i + 2), 5.0) for i in range(10)]
        track.confidences = [0.8] * 10
        segmentation = TrackSegmentation(
            23,
            (
                TrackSegment(1, 0, 3, 0),
                TrackSegment(2, 4, 9, 1),
            ),
        )

        manifest = build_entity_review_manifest(
            "wedstrijd.mp4",
            30.0,
            [track],
            track_segmentations={23: segmentation},
        )

        self.assertEqual(
            [(item.segment_index, item.frames_seen, item.final_team_id) for item in manifest.tracks],
            [(1, 4, 0), (2, 6, 1)],
        )
        self.assertTrue(all(item.observations for item in manifest.tracks))


if __name__ == "__main__":
    unittest.main()
