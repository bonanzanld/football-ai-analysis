from __future__ import annotations

import unittest

from football_ai.analysis.entity_timeline import build_entity_timeline
from football_ai.tracking.entity_corrections import EntityRole, TeamAssignment
from football_ai.tracking.entity_identity import EntityIdentitySet, PhysicalIdentity
from football_ai.tracking.entity_resolver import EntityResolver
from football_ai.tracking.track_state import TrackState


class EntityTimelineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
