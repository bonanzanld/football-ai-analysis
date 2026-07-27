from __future__ import annotations

import unittest

from football_ai.tracking.entity_corrections import EntityRole, TeamAssignment
from football_ai.tracking.entity_identity import (
    EntityIdentitySet,
    PhysicalIdentity,
    _predict_track_center,
    grouped_review_tracks,
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


if __name__ == "__main__":
    unittest.main()
