from __future__ import annotations

import unittest

from football_ai.classification.goalkeeper_classifier import (
    GoalkeeperClassifier,
    GoalkeeperDecision,
    GoalkeeperEvidence,
    GoalLineReference,
    defensive_depth_score,
    goal_line_proximity_score,
)


class GoalkeeperClassifierTests(unittest.TestCase):
    def test_combined_evidence_creates_keeper_candidate(self) -> None:
        result = GoalkeeperClassifier().assess(
            GoalkeeperEvidence(7, 0, 0.85, 0.80, 1.0, 0.90)
        )
        self.assertEqual(result.decision, GoalkeeperDecision.GOALKEEPER_CANDIDATE)
        self.assertGreater(result.score, 0.70)

    def test_different_uniform_and_stable_track_create_candidate_without_goal(self) -> None:
        result = GoalkeeperClassifier().assess(
            GoalkeeperEvidence(7, 0, 0.90, 0.0, 0.0, 0.80, 0.75)
        )
        self.assertEqual(result.decision, GoalkeeperDecision.GOALKEEPER_CANDIDATE)

    def test_last_player_without_uniform_difference_is_not_auto_keeper(self) -> None:
        result = GoalkeeperClassifier().assess(
            GoalkeeperEvidence(7, 0, 0.20, 0.90, 1.0, 0.90)
        )
        self.assertEqual(result.decision, GoalkeeperDecision.REVIEW)

    def test_goal_proximity_uses_finite_segment(self) -> None:
        goal = GoalLineReference("A", (100.0, 100.0), (200.0, 100.0))
        self.assertAlmostEqual(goal_line_proximity_score((150.0, 110.0), goal, 50.0), 0.8)
        self.assertEqual(goal_line_proximity_score((300.0, 100.0), goal, 50.0), 0.0)

    def test_defensive_depth_ranks_nearest_teammate_highest(self) -> None:
        goal = GoalLineReference("A", (0.0, 0.0), (0.0, 20.0))
        score = defensive_depth_score(
            (10.0, 10.0),
            ((30.0, 10.0), (50.0, 10.0)),
            goal,
        )
        self.assertEqual(score, 1.0)

    def test_invalid_evidence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GoalkeeperEvidence(7, 0, 1.2, 0.0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
