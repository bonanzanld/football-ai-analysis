import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from football_ai.tracking.track_segmentation import (
    TeamEvidence,
    TrackSegmentationSet,
    load_track_segmentations,
    save_track_segmentations,
    segment_track_by_team_switches,
)


class TrackSegmentationTests(unittest.TestCase):
    def test_splits_after_sustained_team_change_and_backdates_boundary(self) -> None:
        evidence = [
            TeamEvidence(frame, 0 if frame < 10 else 1, 0.5)
            for frame in range(30)
        ]

        result = segment_track_by_team_switches(23, evidence, initial_team_id=0)

        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].last_frame, 9)
        self.assertEqual(result.segments[1].first_frame, 10)
        self.assertEqual(result.segments[1].team_id, 1)

    def test_ignores_short_colour_incident(self) -> None:
        evidence = [TeamEvidence(frame, 0, 0.5) for frame in range(20)]
        evidence[8] = TeamEvidence(8, 1, 0.7)
        evidence[9] = TeamEvidence(9, 1, 0.7)

        result = segment_track_by_team_switches(7, evidence, initial_team_id=0)

        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].team_id, 0)

    def test_ignores_weak_evidence(self) -> None:
        evidence = [TeamEvidence(frame, 1, 0.05) for frame in range(20)]

        result = segment_track_by_team_switches(7, evidence, initial_team_id=0)

        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].team_id, 0)

    def test_persists_versioned_segmentation_set(self) -> None:
        segmentation = segment_track_by_team_switches(
            23,
            [TeamEvidence(frame, 0 if frame < 5 else 1, 0.5) for frame in range(24)],
            initial_team_id=0,
        )
        data = TrackSegmentationSet("match.mov", 30.0, (segmentation,))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "segments.json"
            save_track_segmentations(data, path)
            restored = load_track_segmentations(path)

        self.assertEqual(restored, data)
        self.assertEqual(data.to_dict()["tracks"][0]["segments"][1]["segment_id"], "23.2")


if __name__ == "__main__":
    unittest.main()
