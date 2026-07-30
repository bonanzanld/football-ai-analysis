from __future__ import annotations

import unittest

from football_ai.analysis.entity_timeline import apply_team_roster, build_entity_timeline
from football_ai.tracking.entity_corrections import EntityRole, TeamAssignment
from football_ai.tracking.entity_identity import EntityIdentitySet, PhysicalIdentity
from football_ai.tracking.entity_resolver import EntityResolver
from football_ai.tracking.entity_corrections import EntityCorrectionSet, TrackCorrection
from football_ai.tracking.track_segmentation import TrackSegment, TrackSegmentation
from football_ai.tracking.track_state import TrackState
from football_ai.tracking.entity_roster import PlayerProfile, TeamRoster


class EntityTimelineTests(unittest.TestCase):
    def test_roster_adds_name_only_to_own_team(self) -> None:
        own = PhysicalIdentity(
            2, "Brabantia - Speler 2", (7,), EntityRole.PLAYER,
            TeamAssignment.TEAM_B, 0, 0,
        )
        opponent = PhysicalIdentity(
            3, "Brandevoort - Speler 1", (8,), EntityRole.PLAYER,
            TeamAssignment.TEAM_A, 0, 0,
        )
        tracks = []
        for track_id in (7, 8):
            track = TrackState(track_id, 0, 0)
            track.observation_frames.append(0)
            track.boxes.append((10.0, 20.0, 30.0, 80.0))
            tracks.append(track)
        timeline = build_entity_timeline(
            "test.mp4", 30.0, tracks, {}, EntityResolver(),
            EntityIdentitySet("test.mp4", (own, opponent)), {7: 1, 8: 0},
        )
        named = apply_team_roster(
            timeline,
            TeamRoster(
                "test.mp4", "Brabantia", TeamAssignment.TEAM_B,
                (PlayerProfile(2, "Daan", "7"),),
            ),
        )
        labels = {item.track_id: item.label for item in named.observations}
        self.assertEqual(labels[7], "Brabantia - Daan (#7)")
        self.assertEqual(labels[8], "Brandevoort - Speler 1")

    def test_confirmed_identity_team_overrides_temporary_track_team(self) -> None:
        track = TrackState(7, 0, 0)
        track.observation_frames.append(0)
        track.boxes.append((10.0, 20.0, 30.0, 80.0))
        identity = PhysicalIdentity(
            2, "Brabantia - Speler 2", (7,), EntityRole.PLAYER,
            TeamAssignment.TEAM_B, 0, 0,
        )
        timeline = build_entity_timeline(
            "test.mp4", 30.0, [track], {}, EntityResolver(),
            EntityIdentitySet("test.mp4", (identity,)), {7: 0},
        )
        self.assertEqual(timeline.observations[0].team, TeamAssignment.TEAM_B)

    def test_segment_correction_overrides_incompatible_physical_identity(self) -> None:
        track = TrackState(23, 0, 0)
        track.observation_frames.append(90)
        track.boxes.append((10.0, 20.0, 30.0, 80.0))
        identity = PhysicalIdentity(
            14, "Brandevoort - Speler 3", (23,), EntityRole.PLAYER,
            TeamAssignment.TEAM_A, 27, 319,
        )
        segmentation = TrackSegmentation(
            track_id=23,
            segments=(TrackSegment(2, 82, 319, 1),),
        )
        corrections = EntityCorrectionSet(
            source_video="test.mp4",
            corrections=(
                TrackCorrection(
                    23,
                    segment_index=2,
                    role=EntityRole.PLAYER,
                    team=TeamAssignment.TEAM_B,
                ),
            ),
        )

        timeline = build_entity_timeline(
            "test.mp4",
            30.0,
            [track],
            {23: segmentation},
            EntityResolver(corrections),
            EntityIdentitySet("test.mp4", (identity,)),
            {23: 0},
        )

        observation = timeline.observations[0]
        self.assertEqual(observation.team, TeamAssignment.TEAM_B)
        self.assertIsNone(observation.identity_id)
        self.assertEqual(observation.label, "ID 23.2")

    def test_stable_identity_overrides_temporary_automatic_team_segment(self) -> None:
        track = TrackState(23, 0, 0)
        track.observation_frames.append(90)
        track.boxes.append((10.0, 20.0, 30.0, 80.0))
        identity = PhysicalIdentity(
            14, "Brabantia - Speler 10", (23,), EntityRole.PLAYER,
            TeamAssignment.TEAM_B, 0, 180,
        )
        segmentation = TrackSegmentation(
            track_id=23,
            segments=(TrackSegment(2, 82, 110, 0),),
        )

        timeline = build_entity_timeline(
            "test.mp4",
            30.0,
            [track],
            {23: segmentation},
            EntityResolver(),
            EntityIdentitySet("test.mp4", (identity,)),
            {23: 1},
        )

        observation = timeline.observations[0]
        self.assertEqual(observation.team, TeamAssignment.TEAM_B)
        self.assertEqual(observation.identity_id, identity.identity_id)
        self.assertEqual(observation.label, identity.label)


if __name__ == "__main__":
    unittest.main()
