from __future__ import annotations

import unittest

from football_ai.tracking.entity_corrections import (
    EntityCorrectionSet,
    EntityRole,
    TeamAssignment,
    TrackCorrection,
)
from football_ai.tracking.entity_resolver import (
    EntityDecisionSource,
    EntityResolver,
)


class EntityResolverTests(unittest.TestCase):
    def test_automatic_team_becomes_player_team(self) -> None:
        entity = EntityResolver().resolve(track_id=8, automatic_team_id=1)

        self.assertEqual(entity.role, EntityRole.PLAYER)
        self.assertEqual(entity.team, TeamAssignment.TEAM_B)
        self.assertEqual(entity.source, EntityDecisionSource.AUTOMATIC)

    def test_manual_goalkeeper_overrides_automatic_player(self) -> None:
        corrections = EntityCorrectionSet(
            source_video="wedstrijd.mp4",
            corrections=(
                TrackCorrection(
                    track_id=8,
                    role=EntityRole.GOALKEEPER,
                    team=TeamAssignment.TEAM_A,
                    note="Keeper met afwijkend shirt",
                ),
            ),
        )

        entity = EntityResolver(corrections).resolve(8, automatic_team_id=1)

        self.assertEqual(entity.role, EntityRole.GOALKEEPER)
        self.assertEqual(entity.team, TeamAssignment.TEAM_A)
        self.assertEqual(entity.source, EntityDecisionSource.MANUAL)

    def test_manual_staff_is_excluded(self) -> None:
        corrections = EntityCorrectionSet(
            source_video="wedstrijd.mp4",
            corrections=(
                TrackCorrection(
                    track_id=12,
                    role=EntityRole.STAFF,
                    team=TeamAssignment.NONE,
                    excluded=True,
                ),
            ),
        )

        entity = EntityResolver(corrections).resolve(12, automatic_team_id=0)

        self.assertTrue(entity.excluded)
        self.assertFalse(entity.included_in_football_analysis)

    def test_missing_team_stays_unknown(self) -> None:
        entity = EntityResolver().resolve(track_id=3, automatic_team_id=None)

        self.assertEqual(entity.role, EntityRole.UNKNOWN)
        self.assertEqual(entity.source, EntityDecisionSource.UNCLASSIFIED)


if __name__ == "__main__":
    unittest.main()
