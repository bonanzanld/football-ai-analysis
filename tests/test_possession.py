from __future__ import annotations

import unittest

from football_ai.analysis.entity_timeline import TimelineEntity
from football_ai.analysis.possession import (
    PossessionObservation,
    PossessionState,
    PossessionTracker,
    build_possession_statistics,
    should_render_inferred_ball,
)
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

    def test_confirmed_low_confidence_arrival_still_creates_pass(self) -> None:
        tracker = PossessionTracker(confirmation_frames=2)
        first = entity(1, TeamAssignment.TEAM_A, 100)
        second = entity(2, TeamAssignment.TEAM_A, 200)
        tracker.update(0, ball(0, 100), [first])
        tracker.update(1, ball(1, 100), [first])
        tracker.update(2, ball(2, 198, confidence=0.15), [second])
        tracker.update(3, ball(3, 198, confidence=0.15), [second])

        self.assertEqual(len(tracker.passes), 1)

    def test_very_weak_transfer_does_not_create_pass_event(self) -> None:
        tracker = PossessionTracker(confirmation_frames=2)
        first = entity(1, TeamAssignment.TEAM_A, 100)
        second = entity(2, TeamAssignment.TEAM_A, 200)
        tracker.update(0, ball(0, 100), [first])
        tracker.update(1, ball(1, 100), [first])
        tracker.update(2, ball(2, 215, confidence=0.08), [second])
        tracker.update(3, ball(3, 215, confidence=0.08), [second])

        self.assertEqual(len(tracker.passes), 0)

    def test_confirmed_opponent_arrival_creates_turnover(self) -> None:
        tracker = PossessionTracker(
            confirmation_frames=2,
            opponent_confirmation_frames=2,
        )
        first = entity(1, TeamAssignment.TEAM_A, 100)
        opponent = entity(2, TeamAssignment.TEAM_B, 200)
        tracker.update(0, ball(0, 100), [first])
        tracker.update(1, ball(1, 100), [first])
        tracker.update(2, ball(2, 200), [opponent])
        tracker.update(3, ball(3, 200), [opponent])

        self.assertEqual(len(tracker.turnovers), 1)
        self.assertEqual(tracker.turnovers[0].from_team, TeamAssignment.TEAM_A.value)
        self.assertEqual(tracker.turnovers[0].to_team, TeamAssignment.TEAM_B.value)

    def test_interpolated_ball_can_confirm_possession_near_footpoint(self) -> None:
        tracker = PossessionTracker(confirmation_frames=2)
        player = entity(1, TeamAssignment.TEAM_A, 100)
        interpolated = BallObservation(
            0, (100, 100), (97, 97, 103, 103), 0.4, "interpolated"
        )

        tracker.update(0, interpolated, [player])
        result = tracker.update(1, interpolated, [player])

        self.assertEqual(result.state, PossessionState.CONTROLLED)

    def test_keeps_same_owner_when_ball_temporarily_disappears(self) -> None:
        tracker = PossessionTracker(confirmation_frames=2)
        player = entity(1, TeamAssignment.TEAM_A, 100)
        tracker.update(0, ball(0, 100), [player])
        tracker.update(1, ball(1, 100), [player])

        result = tracker.update(2, None, [player])

        self.assertEqual(result.state, PossessionState.INFERRED)
        self.assertEqual(result.identity_id, player.identity_id)
        self.assertEqual(result.team, TeamAssignment.TEAM_A.value)

    def test_opponent_must_be_confirmed_before_possession_changes(self) -> None:
        tracker = PossessionTracker(
            confirmation_frames=2,
            opponent_confirmation_frames=2,
        )
        first = entity(1, TeamAssignment.TEAM_A, 100)
        opponent = entity(2, TeamAssignment.TEAM_B, 200)
        tracker.update(0, ball(0, 100), [first])
        tracker.update(1, ball(1, 100), [first])

        pending = tracker.update(2, ball(2, 200), [opponent])
        changed = tracker.update(3, ball(3, 200), [opponent])

        self.assertEqual(pending.state, PossessionState.INFERRED)
        self.assertEqual(pending.team, TeamAssignment.TEAM_A.value)
        self.assertEqual(changed.state, PossessionState.CONTROLLED)
        self.assertEqual(changed.team, TeamAssignment.TEAM_B.value)

    def test_opponent_flyby_does_not_break_teammate_pass(self) -> None:
        tracker = PossessionTracker(
            confirmation_frames=2,
            opponent_confirmation_frames=6,
        )
        passer = entity(1, TeamAssignment.TEAM_A, 100)
        opponent = entity(2, TeamAssignment.TEAM_B, 200)
        receiver = entity(3, TeamAssignment.TEAM_A, 300)
        tracker.update(0, ball(0, 100), [passer])
        tracker.update(1, ball(1, 100), [passer])
        for frame in range(2, 6):
            tracker.update(frame, ball(frame, 200), [opponent])
        tracker.update(6, ball(6, 300), [receiver])
        tracker.update(7, ball(7, 300), [receiver])

        self.assertEqual(len(tracker.turnovers), 0)
        self.assertEqual(len(tracker.passes), 1)
        self.assertEqual(tracker.passes[0].from_label, "Speler 1")
        self.assertEqual(tracker.passes[0].to_label, "Speler 3")

    def test_opponent_must_be_closer_than_teammate_receiver(self) -> None:
        tracker = PossessionTracker(
            confirmation_frames=2,
            opponent_confirmation_frames=2,
            opponent_control_radius=0.45,
        )
        owner = entity(1, TeamAssignment.TEAM_A, 100)
        opponent = entity(2, TeamAssignment.TEAM_B, 200)
        tracker.update(0, ball(0, 100), [owner])
        tracker.update(1, ball(1, 100), [owner])
        # 25 px is binnen de oude algemene zone, maar niet overtuigend genoeg
        # voor een tegenstander die werkelijk de bal overneemt.
        pending = tracker.update(2, ball(2, 225), [opponent])
        pending = tracker.update(3, ball(3, 225), [opponent])

        self.assertEqual(pending.state, PossessionState.INFERRED)
        self.assertEqual(pending.team, TeamAssignment.TEAM_A.value)
        self.assertEqual(len(tracker.turnovers), 0)

    def test_fast_flyby_does_not_create_possession_without_control_evidence(self) -> None:
        tracker = PossessionTracker(
            confirmation_frames=2,
            dwell_control_frames=6,
            low_speed_threshold=2.0,
        )
        player = entity(1, TeamAssignment.TEAM_A, 100)

        for frame, x in enumerate((40, 60, 80, 100, 120)):
            result = tracker.update(frame, ball(frame, x), [player])

        self.assertNotEqual(result.state, PossessionState.CONTROLLED)
        self.assertIsNone(result.identity_id)

    def test_direction_change_confirms_one_touch_control(self) -> None:
        tracker = PossessionTracker(
            confirmation_frames=2,
            dwell_control_frames=8,
            low_speed_threshold=1.0,
            direction_change_degrees=30.0,
        )
        player = entity(1, TeamAssignment.TEAM_A, 100)

        tracker.update(0, ball(0, 60), [])
        tracker.update(1, ball(1, 85), [player])
        result = tracker.update(2, ball(2, 100), [player])
        self.assertNotEqual(result.state, PossessionState.CONTROLLED)
        tracker.update(3, ball(3, 88), [player])
        result = tracker.update(4, ball(4, 76), [player])

        self.assertEqual(result.state, PossessionState.CONTROLLED)
        self.assertEqual(result.identity_id, player.identity_id)

    def test_inferred_possession_counts_for_team_and_player_statistics(self) -> None:
        observations = [
            PossessionObservation(
                0, PossessionState.CONTROLLED, 7, 12, "Speler 7", "team_a", 0.8
            ),
            PossessionObservation(
                1, PossessionState.INFERRED, 7, 12, "Speler 7", "team_a", 0.6
            ),
            PossessionObservation(
                2, PossessionState.INFERRED, 7, 12, "Speler 7", "team_a", 0.5
            ),
        ]

        statistics = build_possession_statistics(observations, fps=2.0)

        self.assertEqual(statistics["teams"]["team_a"]["total_possession_frames"], 3)
        self.assertEqual(statistics["players"]["7"]["inferred_frames"], 2)
        self.assertAlmostEqual(
            statistics["players"]["7"]["total_possession_seconds"],
            1.5,
        )

    def test_reliable_detected_ball_suppresses_inferred_ball_marker(self) -> None:
        possession = PossessionObservation(
            5, PossessionState.INFERRED, 7, 12, "Speler 7", "team_a", 0.6
        )

        self.assertFalse(
            should_render_inferred_ball(possession, ball(5, 300, confidence=0.84))
        )

    def test_missing_or_weak_ball_allows_inferred_ball_marker(self) -> None:
        possession = PossessionObservation(
            5, PossessionState.INFERRED, 7, 12, "Speler 7", "team_a", 0.6
        )

        self.assertTrue(should_render_inferred_ball(possession, None))
        self.assertTrue(
            should_render_inferred_ball(possession, ball(5, 300, confidence=0.06))
        )


if __name__ == "__main__":
    unittest.main()
