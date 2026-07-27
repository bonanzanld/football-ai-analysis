from __future__ import annotations

import unittest

from football_ai.classification.participant_classifier import (
    ParticipantClassifier,
    ParticipantDecision,
    ParticipantEvidence,
)


class ParticipantClassifierTests(unittest.TestCase):
    def test_reliable_team_member_remains_player(self) -> None:
        result = ParticipantClassifier().assess(
            ParticipantEvidence(1, None, 0.95, 0.8, 0.9, 0.8, 1.0)
        )
        self.assertEqual(result.decision, ParticipantDecision.PLAYER)

    def test_active_outlier_inside_player_group_requires_referee_review(self) -> None:
        result = ParticipantClassifier().assess(
            ParticipantEvidence(2, None, 0.0, 0.9, 0.9, 0.8, 0.9)
        )
        self.assertEqual(result.decision, ParticipantDecision.REFEREE_REVIEW)

    def test_stationary_track_outside_group_requires_outsider_review(self) -> None:
        result = ParticipantClassifier().assess(
            ParticipantEvidence(3, None, 0.0, 0.7, 0.1, 0.1, 0.9)
        )
        self.assertEqual(result.decision, ParticipantDecision.OUTSIDER_REVIEW)

    def test_values_must_be_normalized(self) -> None:
        with self.assertRaises(ValueError):
            ParticipantEvidence(4, None, 0.0, 0.0, 1.1, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
