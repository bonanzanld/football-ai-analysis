from __future__ import annotations

import unittest

from football_ai.tracking.entity_corrections import TeamAssignment
from football_ai.tracking.entity_roster import PlayerProfile, PositionPeriod, TeamRoster


class EntityRosterTests(unittest.TestCase):
    def test_name_and_optional_squad_number_override_fallback(self) -> None:
        roster = TeamRoster(
            source_video="match.mov",
            own_team_name="Brabantia",
            own_team=TeamAssignment.TEAM_B,
            players=(PlayerProfile(3, "Daan", "7"),),
        )
        self.assertEqual(roster.display_label(3, "Brabantia - Speler 2"), "Daan (#7)")
        self.assertEqual(roster.display_label(4, "Brabantia - Speler 3"), "Brabantia - Speler 3")

    def test_player_can_have_multiple_positions_during_match(self) -> None:
        profile = PlayerProfile(
            3,
            "Daan",
            position_periods=(
                PositionPeriod("linksachter", 0.0),
                PositionPeriod("linksvoor", 18.0),
            ),
        )
        restored = PlayerProfile.from_dict(profile.to_dict())
        self.assertEqual(restored.position_periods[1].position, "linksvoor")
        self.assertEqual(restored.position_periods[1].start_minute, 18.0)

    def test_old_roster_defaults_to_team_b(self) -> None:
        restored = TeamRoster.from_dict({
            "schema_version": 1,
            "source_video": "match.mov",
            "own_team_name": "Brabantia",
            "players": [],
        })
        self.assertEqual(restored.own_team, TeamAssignment.TEAM_B)


if __name__ == "__main__":
    unittest.main()
