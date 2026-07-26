from __future__ import annotations

import unittest

from football_ai.tracking.entity_corrections import EntityRole, TeamAssignment
from football_ai.tracking.entity_review_app import correction_for_action


class EntityReviewAppTests(unittest.TestCase):
    def test_keeper_action_assigns_role_and_team(self) -> None:
        correction = correction_for_action(7, "goalkeeper_b")

        self.assertEqual(correction.role, EntityRole.GOALKEEPER)
        self.assertEqual(correction.team, TeamAssignment.TEAM_B)
        self.assertFalse(correction.excluded)

    def test_exclude_action_is_removed_from_analysis(self) -> None:
        correction = correction_for_action(9, "exclude")

        self.assertTrue(correction.excluded)
        self.assertFalse(correction.included_in_football_analysis)

    def test_unknown_action_is_reversible_manual_choice(self) -> None:
        correction = correction_for_action(3, "unknown")

        self.assertEqual(correction.role, EntityRole.UNKNOWN)
        self.assertFalse(correction.excluded)


if __name__ == "__main__":
    unittest.main()
