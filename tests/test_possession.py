from __future__ import annotations

import unittest

from football_ai.analysis.entity_timeline import TimelineEntity
from football_ai.analysis.possession import PossessionState, PossessionTracker
from football_ai.detection.ball_tracking import BallObservation
from football_ai.tracking.entity_corrections import EntityRole, TeamAssignment


def entity(track_id: int, team: TeamAssignment, x: float) -> TimelineEntity:
    return TimelineEntity(0, track_id, track_id, f"Speler {track_id}", EntityRole.PLAYER, team, (x - 10, 50, x + 10, 100), (x, 100))


def ball(frame: int, x: float, confidence: float = 0.9) -> BallObservation:
    return BallObservation(frame, (x, 100), (x - 3, 97, x + 3, 103), confidence, "detected")


class PossessionTests(unittest.TestCase):
    def test_requires_multiple_frames_before_confirming_owner(self) -> None:
        tracker = PossessionTracker(confirmation_frames=3)
        player = entity(1, TeamAssignment.TEAM_A, 100)
        self.assertEqual(tracker.update(0, ball(0, 100), [player]).state, PossessionState.CONTESTED)
        self.assertEqual(tracker.update(1, ball(1, 100), [player]).state, PossessionState.CONTESTED)
        self.assertEqual(tracker.update(2, ball(2, 100), [player]).state, PossessionState.CONTROLLED)

    def test_close_opponents_are_contested(self) -> None:
        tracker = PossessionTracker()
        first = entity(1, TeamAssignment.TEAM_A, 100)
        second = entity(2, TeamAssignment.TEAM_B, 105)
        result = tracker.update(0, ball(0, 102), [first, second])
        self.assertEqual(result.state, PossessionState.CONTESTED)

    def test_same_team_owner_change_creates_pass_after_confirmation(self) -> None:
        tracker = PossessionTracker(confirmation_frames=2, minimum_pass_confidence=0.0)
        first = entity(1, TeamAssignment.TEAM_A, 100)
        second = entity(2, TeamAssignment.TEAM_A, 200)
        tracker.update(0, ball(0, 100), [first])
        tracker.update(1, ball(1, 100), [first])
        tracker.update(2, ball(2, 200), [second])
        tracker.update(3, ball(3, 200), [second])
        self.assertEqual(len(tracker.passes), 1)
        self.assertEqual(tracker.passes[0].from_label, "Speler 1")
        self.assertEqual(tracker.passes[0].to_label, "Speler 2")


if __name__ == "__main__":
    unittest.main()
