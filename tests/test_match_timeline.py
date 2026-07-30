from __future__ import annotations

import unittest

from football_ai.analysis.match_timeline import MatchTimelineEngine, MatchTimelineEvent
from football_ai.analysis.possession import PossessionObservation, PossessionState


def observation(
    frame: int,
    state: PossessionState,
    player: int | None,
    team: str | None,
    confidence: float = 0.8,
) -> PossessionObservation:
    return PossessionObservation(
        frame_number=frame,
        state=state,
        identity_id=player,
        track_id=player,
        label=f"Speler {player}" if player is not None else None,
        team=team,
        confidence=confidence,
    )


def temporary_track_observation(
    frame: int,
    track_id: int,
    team: str,
    confidence: float = 0.8,
) -> PossessionObservation:
    return PossessionObservation(
        frame_number=frame,
        state=PossessionState.CONTROLLED,
        identity_id=None,
        track_id=track_id,
        label=f"ID {track_id}",
        team=team,
        confidence=confidence,
    )


class MatchTimelineEngineTests(unittest.TestCase):
    def test_unknown_gap_freezes_possession_but_preserves_pass_context(self) -> None:
        raw = [
            observation(0, PossessionState.CONTROLLED, 1, "team_a"),
            observation(1, PossessionState.CONTROLLED, 1, "team_a"),
            observation(2, PossessionState.UNKNOWN, None, None, 0.0),
            observation(3, PossessionState.UNKNOWN, None, None, 0.0),
            observation(4, PossessionState.CONTROLLED, 2, "team_a"),
            observation(5, PossessionState.CONTROLLED, 2, "team_a"),
        ]

        result = MatchTimelineEngine(fps=30.0).resolve(raw)

        self.assertEqual(result.observations[2].state, PossessionState.UNKNOWN)
        self.assertEqual(result.observations[3].state, PossessionState.UNKNOWN)
        self.assertEqual(len(result.passes), 1)
        self.assertEqual(result.passes[0].from_identity_id, 1)
        self.assertEqual(result.passes[0].to_identity_id, 2)

    def test_weak_brief_opponent_control_does_not_change_team(self) -> None:
        raw = [
            observation(0, PossessionState.CONTROLLED, 1, "team_a"),
            observation(1, PossessionState.CONTROLLED, 1, "team_a"),
            observation(2, PossessionState.UNKNOWN, None, None, 0.0),
            observation(3, PossessionState.CONTROLLED, 7, "team_b", 0.11),
            observation(4, PossessionState.CONTROLLED, 7, "team_b", 0.12),
            observation(5, PossessionState.UNKNOWN, None, None, 0.0),
            observation(6, PossessionState.CONTROLLED, 2, "team_a"),
            observation(7, PossessionState.CONTROLLED, 2, "team_a"),
        ]

        result = MatchTimelineEngine(fps=30.0).resolve(raw)

        self.assertEqual(result.observations[3].state, PossessionState.CONTESTED)
        self.assertIsNone(result.observations[3].team)
        self.assertEqual(result.turnovers, ())
        self.assertEqual(len(result.passes), 1)

    def test_sustained_confident_opponent_control_creates_turnover(self) -> None:
        raw = [
            observation(0, PossessionState.CONTROLLED, 1, "team_a"),
            observation(1, PossessionState.CONTROLLED, 1, "team_a"),
        ]
        raw.extend(
            observation(frame, PossessionState.CONTROLLED, 7, "team_b", 0.7)
            for frame in range(2, 9)
        )

        result = MatchTimelineEngine(fps=30.0).resolve(raw)

        self.assertEqual(len(result.turnovers), 1)
        self.assertEqual(result.turnovers[0].from_team, "team_a")
        self.assertEqual(result.turnovers[0].to_team, "team_b")
        self.assertEqual(result.turnovers[0].event_type, "possession_change")

    def test_opponent_receiving_travelling_ball_creates_interception(self) -> None:
        raw = [
            observation(0, PossessionState.CONTROLLED, 1, "team_a"),
            observation(1, PossessionState.CONTROLLED, 1, "team_a"),
        ]
        raw.extend(
            observation(frame, PossessionState.INFERRED, 1, "team_a", 0.8)
            for frame in range(2, 8)
        )
        raw.extend(
            observation(frame, PossessionState.CONTESTED, None, None, 0.3)
            for frame in range(8, 10)
        )
        raw.extend(
            observation(frame, PossessionState.CONTROLLED, 7, "team_b", 0.7)
            for frame in range(10, 17)
        )

        result = MatchTimelineEngine(fps=30.0).resolve(raw)

        self.assertEqual(len(result.turnovers), 1)
        self.assertEqual(result.turnovers[0].event_type, "intercepted_pass")
        self.assertEqual(result.turnovers[0].from_team, "team_a")
        self.assertEqual(result.turnovers[0].to_team, "team_b")

    def test_same_player_cannot_pass_to_itself(self) -> None:
        raw = [
            observation(0, PossessionState.CONTROLLED, 1, "team_a"),
            observation(1, PossessionState.UNKNOWN, None, None, 0.0),
            observation(2, PossessionState.CONTROLLED, 1, "team_a"),
            observation(3, PossessionState.CONTROLLED, 1, "team_a"),
        ]

        result = MatchTimelineEngine(fps=30.0).resolve(raw)

        self.assertEqual(result.passes, ())

    def test_weak_teammate_touch_keeps_team_possession_and_counts_one_pass(self) -> None:
        raw = [
            observation(0, PossessionState.CONTROLLED, 1, "team_b", 0.8),
            observation(1, PossessionState.CONTROLLED, 1, "team_b", 0.8),
            observation(2, PossessionState.CONTROLLED, 2, "team_b", 0.12),
            observation(3, PossessionState.CONTROLLED, 2, "team_b", 0.08),
            observation(4, PossessionState.CONTROLLED, 2, "team_b", 0.10),
            observation(5, PossessionState.INFERRED, 2, "team_b", 0.98),
        ]

        result = MatchTimelineEngine(fps=30.0).resolve(raw)

        self.assertEqual(len(result.passes), 1)
        self.assertEqual(result.passes[0].from_identity_id, 1)
        self.assertEqual(result.passes[0].to_identity_id, 2)
        self.assertTrue(
            all(item.team == "team_b" for item in result.observations),
        )

    def test_quick_same_team_relay_is_collapsed_to_one_pass(self) -> None:
        raw = [
            observation(0, PossessionState.CONTROLLED, 1, "team_b", 0.8),
            observation(1, PossessionState.CONTROLLED, 1, "team_b", 0.8),
            observation(2, PossessionState.CONTROLLED, 2, "team_b", 0.4),
            observation(3, PossessionState.CONTROLLED, 2, "team_b", 0.4),
            observation(4, PossessionState.CONTROLLED, 3, "team_b", 0.8),
            observation(5, PossessionState.CONTROLLED, 3, "team_b", 0.8),
        ]

        result = MatchTimelineEngine(fps=30.0).resolve(raw)

        self.assertEqual(len(result.passes), 1)
        self.assertEqual(result.passes[0].from_identity_id, 1)
        self.assertEqual(result.passes[0].to_identity_id, 3)

    def test_established_same_team_possessions_remain_two_passes(self) -> None:
        raw = [
            observation(0, PossessionState.CONTROLLED, 1, "team_b"),
            observation(1, PossessionState.CONTROLLED, 1, "team_b"),
            observation(2, PossessionState.CONTROLLED, 2, "team_b"),
        ]
        raw.extend(
            observation(frame, PossessionState.CONTROLLED, 2, "team_b")
            for frame in range(3, 36)
        )
        raw.extend(
            observation(frame, PossessionState.CONTROLLED, 3, "team_b")
            for frame in range(36, 38)
        )

        result = MatchTimelineEngine(fps=30.0).resolve(raw)

        self.assertEqual(len(result.passes), 2)

    def test_immediate_temporary_track_handover_is_not_a_pass(self) -> None:
        raw = [
            observation(0, PossessionState.CONTROLLED, 48, "team_b"),
            observation(1, PossessionState.CONTROLLED, 48, "team_b"),
            PossessionObservation(
                frame_number=2,
                state=PossessionState.CONTROLLED,
                identity_id=None,
                track_id=55,
                label="ID 55.2",
                team="team_b",
                confidence=0.7,
            ),
            PossessionObservation(
                frame_number=3,
                state=PossessionState.CONTROLLED,
                identity_id=None,
                track_id=55,
                label="ID 55.2",
                team="team_b",
                confidence=0.7,
            ),
        ]

        result = MatchTimelineEngine(fps=30.0).resolve(raw)

        self.assertEqual(result.passes, ())

    def test_distinct_temporary_tracks_can_form_a_pass(self) -> None:
        raw = [
            temporary_track_observation(0, 55, "team_b"),
            temporary_track_observation(1, 55, "team_b"),
            observation(2, PossessionState.UNKNOWN, None, None, 0.0),
            observation(3, PossessionState.UNKNOWN, None, None, 0.0),
            temporary_track_observation(4, 61, "team_b"),
            temporary_track_observation(5, 61, "team_b"),
        ]

        result = MatchTimelineEngine(fps=30.0).resolve(raw)

        self.assertEqual(len(result.passes), 1)
        self.assertIsNone(result.passes[0].from_identity_id)
        self.assertIsNone(result.passes[0].to_identity_id)
        self.assertEqual(result.passes[0].from_track_id, 55)
        self.assertEqual(result.passes[0].to_track_id, 61)

    def test_temporary_track_cannot_pass_to_itself(self) -> None:
        raw = [
            temporary_track_observation(0, 55, "team_b"),
            temporary_track_observation(1, 55, "team_b"),
            observation(2, PossessionState.UNKNOWN, None, None, 0.0),
            temporary_track_observation(3, 55, "team_b"),
            temporary_track_observation(4, 55, "team_b"),
        ]

        result = MatchTimelineEngine(fps=30.0).resolve(raw)

        self.assertEqual(result.passes, ())

    def test_weak_opponent_episode_stays_unknown_including_inference(self) -> None:
        raw = [
            observation(0, PossessionState.CONTROLLED, 1, "team_b", 0.8),
            observation(1, PossessionState.CONTROLLED, 1, "team_b", 0.8),
        ]
        raw.extend(
            observation(frame, PossessionState.CONTROLLED, 7, "team_a", 0.20)
            for frame in range(2, 8)
        )
        raw.extend(
            observation(frame, PossessionState.INFERRED, 7, "team_a", 0.9)
            for frame in range(8, 12)
        )

        result = MatchTimelineEngine(fps=30.0).resolve(raw)

        self.assertEqual(result.turnovers, ())
        self.assertTrue(
            all(item.team is None for item in result.observations[2:]),
        )

    def test_public_events_are_chronological_and_use_club_names(self) -> None:
        raw = [
            PossessionObservation(
                0, PossessionState.CONTROLLED, 1, 1,
                "Brabantia - Speler 1", "team_b", 0.9,
            ),
            PossessionObservation(
                1, PossessionState.CONTROLLED, 1, 1,
                "Brabantia - Speler 1", "team_b", 0.9,
            ),
            PossessionObservation(
                2, PossessionState.CONTROLLED, 2, 2,
                "Brabantia - Speler 2", "team_b", 0.9,
            ),
            PossessionObservation(
                3, PossessionState.CONTROLLED, 2, 2,
                "Brabantia - Speler 2", "team_b", 0.9,
            ),
        ]

        result = MatchTimelineEngine(fps=25.0).resolve(raw)
        event = result.events[0].to_dict(25.0)

        self.assertEqual(event["event_type"], "successful_pass")
        self.assertEqual(event["from_club"], "Brabantia")
        self.assertEqual(event["to_club"], "Brabantia")
        self.assertEqual(event["confirmed_at_frame"], 2)
        self.assertAlmostEqual(event["confirmed_at_seconds"], 0.08)
        self.assertEqual(event["from_track_id"], 1)
        self.assertEqual(event["to_track_id"], 2)

    def test_public_event_uses_team_name_for_temporary_track_label(self) -> None:
        event = MatchTimelineEvent(
            event_type="successful_pass",
            start_frame=10,
            end_frame=20,
            from_identity_id=None,
            to_identity_id=8,
            from_label="ID 55.2",
            to_label="Brabantia - Speler 3",
            from_team="team_b",
            to_team="team_b",
            confidence=0.8,
        )

        serialized = event.to_dict(10.0, {"team_b": "Brabantia"})

        self.assertEqual(serialized["from_club"], "Brabantia")
        self.assertEqual(serialized["to_club"], "Brabantia")

    def test_public_event_uses_team_name_for_unknown_player_label(self) -> None:
        event = MatchTimelineEvent(
            event_type="intercepted_pass",
            start_frame=10,
            end_frame=20,
            from_identity_id=3,
            to_identity_id=None,
            from_label="Brabantia - Speler 3",
            to_label="Onbekend - Speler 1",
            from_team="team_b",
            to_team="team_a",
            confidence=0.7,
        )

        serialized = event.to_dict(
            10.0,
            {"team_a": "Brandevoort", "team_b": "Brabantia"},
        )

        self.assertEqual(serialized["from_club"], "Brabantia")
        self.assertEqual(serialized["to_club"], "Brandevoort")


if __name__ == "__main__":
    unittest.main()
