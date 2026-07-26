from __future__ import annotations

import unittest

from football_ai.tracking.entity_corrections import EntityRole, TeamAssignment
from football_ai.tracking.entity_resolver import EntityDecisionSource, ResolvedEntity
from football_ai.visualizer import _entity_style


class EntityVisualizerTests(unittest.TestCase):
    def test_goalkeeper_keeps_team_color_and_role_label(self) -> None:
        entity = ResolvedEntity(
            track_id=5,
            role=EntityRole.GOALKEEPER,
            team=TeamAssignment.TEAM_B,
            excluded=False,
            source=EntityDecisionSource.MANUAL,
        )

        color, label = _entity_style(entity)

        self.assertEqual(color, (0, 0, 255))
        self.assertEqual(label, "Keeper B")

    def test_unknown_person_has_neutral_label(self) -> None:
        entity = ResolvedEntity(
            track_id=6,
            role=EntityRole.UNKNOWN,
            team=TeamAssignment.UNKNOWN,
            excluded=False,
            source=EntityDecisionSource.UNCLASSIFIED,
        )

        _color, label = _entity_style(entity)

        self.assertEqual(label, "Persoon ?")


if __name__ == "__main__":
    unittest.main()
