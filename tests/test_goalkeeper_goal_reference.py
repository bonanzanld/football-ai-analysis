from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football_ai.classification.goalkeeper_goal_reference import (
    GoalkeeperGoalReference,
    load_goalkeeper_goal_references,
    save_goalkeeper_goal_references,
)


class GoalkeeperGoalReferenceTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        reference = GoalkeeperGoalReference(30, 1.0, 0, (10.0, 20.0), (40.0, 22.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "goals.json"
            save_goalkeeper_goal_references("match.mp4", (reference,), path)
            restored = load_goalkeeper_goal_references(path)
        self.assertEqual(restored, (reference,))

    def test_same_post_points_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GoalkeeperGoalReference(30, 1.0, 0, (10.0, 20.0), (10.0, 20.0))

    def test_unknown_team_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GoalkeeperGoalReference(30, 1.0, 3, (10.0, 20.0), (40.0, 20.0))


if __name__ == "__main__":
    unittest.main()
