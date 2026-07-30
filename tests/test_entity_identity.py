from __future__ import annotations

import unittest

from football_ai.tracking.entity_corrections import EntityRole, TeamAssignment
from football_ai.tracking.entity_identity import (
    EntityIdentitySet,
    PhysicalIdentity,
    _group_assignment,
    _predict_track_center,
    _score_link,
    grouped_review_tracks,
)
from football_ai.tracking.entity_corrections import (
    EntityCorrectionSet,
    TrackCorrection,
)
from football_ai.tracking.entity_review_manifest import (
    EntityReviewManifest,
    ReviewObservation,
    ReviewTrack,
)


def track(track_id: int, first: int, last: int) -> ReviewTrack:
    return ReviewTrack(
        track_id=track_id,
        first_frame=first,
        last_frame=last,
        frames_seen=last - first + 1,
        average_confidence=0.9,
        observations=(
            ReviewObservation(first, (10.0, 20.0, 30.0, 80.0)),
            ReviewObservation(last, (20.0, 20.0, 40.0, 80.0)),
        ),
        final_team_id=1,
        team_votes_b=20,
        team_agreement_ratio=1.0,
        team_is_reliable=True,
    )


class EntityIdentityTests(unittest.TestCase):
    def test_serialization_preserves_numbered_identity(self) -> None:
        source = EntityIdentitySet(
            source_video="match.mov",
            identities=(
                PhysicalIdentity(
                    identity_id=1,
                    label="Brabantia - Speler 1",
                    track_ids=(4, 9),
                    role=EntityRole.PLAYER,
                    team=TeamAssignment.TEAM_B,
                    first_frame=10,
                    last_frame=80,
                ),
            ),
        )
        restored = EntityIdentitySet.from_dict(source.to_dict())
        self.assertEqual(restored, source)

    def test_grouped_review_combines_fragment_observations(self) -> None:
        manifest = EntityReviewManifest(
            source_video="match.mov",
            fps=30.0,
            tracks=(track(4, 10, 20), track(9, 24, 30)),
        )
        identities = EntityIdentitySet(
            source_video="match.mov",
            identities=(
                PhysicalIdentity(
                    1,
                    "Brabantia - Speler 1",
                    (4, 9),
                    EntityRole.PLAYER,
                    TeamAssignment.TEAM_B,
                    10,
                    30,
                ),
            ),
        )
        grouped = grouped_review_tracks(manifest, identities)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0].frames_seen, 18)
        self.assertEqual(len(grouped[0].observations), 4)

    def test_predicts_reappearance_in_same_movement_direction(self) -> None:
        source = track(4, 10, 20)

        predicted = _predict_track_center(
            source,
            target_frame=30,
            body_scale=60.0,
        )

        self.assertIsNotNone(predicted)
        self.assertAlmostEqual(float(predicted[0]), 40.0)
        self.assertAlmostEqual(float(predicted[1]), 50.0)

    def test_repairs_conflicting_team_for_nearly_certain_track_continuation(self) -> None:
        first = ReviewTrack(
            track_id=15,
            first_frame=23,
            last_frame=116,
            frames_seen=66,
            average_confidence=0.70,
            observations=(
                ReviewObservation(105, (1000.0, 280.0, 1025.0, 320.0)),
                ReviewObservation(116, (1018.0, 285.0, 1043.0, 316.0)),
            ),
            final_team_id=0,
            team_votes_a=64,
            team_agreement_ratio=1.0,
            team_is_reliable=True,
        )
        second = ReviewTrack(
            track_id=60,
            first_frame=120,
            last_frame=898,
            frames_seen=779,
            average_confidence=0.82,
            observations=(
                ReviewObservation(120, (1002.0, 286.0, 1030.0, 325.0)),
                ReviewObservation(898, (860.0, 309.0, 900.0, 376.0)),
            ),
            final_team_id=1,
            team_votes_b=750,
            team_agreement_ratio=0.965,
            team_is_reliable=True,
        )
        corrections = EntityCorrectionSet(
            source_video="match.mov",
            corrections=(
                TrackCorrection(15, role=EntityRole.PLAYER, team=TeamAssignment.TEAM_A),
                TrackCorrection(60, role=EntityRole.PLAYER, team=TeamAssignment.TEAM_B),
            ),
        )
        descriptor = __import__("numpy").ones(4, dtype="float32") / 2.0

        link = _score_link(
            first,
            second,
            descriptor,
            descriptor,
            corrections,
            1280.0,
            720.0,
            gap=3,
        )
        team, role = _group_assignment([first, second], corrections)

        self.assertIsNotNone(link)
        self.assertEqual(link.decision, "auto_merge_team_repair")
        self.assertEqual(team, TeamAssignment.TEAM_B)
        self.assertEqual(role, EntityRole.PLAYER)

    def test_does_not_repair_team_conflict_without_extreme_continuity(self) -> None:
        first = track(4, 10, 20)
        second = track(9, 24, 40)
        second = ReviewTrack(
            track_id=second.track_id,
            first_frame=second.first_frame,
            last_frame=second.last_frame,
            frames_seen=second.frames_seen,
            average_confidence=second.average_confidence,
            observations=(
                ReviewObservation(24, (300.0, 20.0, 320.0, 80.0)),
                ReviewObservation(40, (320.0, 20.0, 340.0, 80.0)),
            ),
            final_team_id=0,
            team_votes_a=17,
            team_agreement_ratio=1.0,
            team_is_reliable=True,
        )
        corrections = EntityCorrectionSet(
            source_video="match.mov",
            corrections=(
                TrackCorrection(4, role=EntityRole.PLAYER, team=TeamAssignment.TEAM_B),
                TrackCorrection(9, role=EntityRole.PLAYER, team=TeamAssignment.TEAM_A),
            ),
        )
        descriptor = __import__("numpy").ones(4, dtype="float32") / 2.0

        link = _score_link(
            first,
            second,
            descriptor,
            descriptor,
            corrections,
            1280.0,
            720.0,
            gap=3,
        )

        self.assertIsNone(link)


if __name__ == "__main__":
    unittest.main()
