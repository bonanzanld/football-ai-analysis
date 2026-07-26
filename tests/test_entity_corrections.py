import tempfile
import unittest
from pathlib import Path

from football_ai.tracking.entity_corrections import (
    EntityCorrectionSet,
    EntityRole,
    TeamAssignment,
    TrackCorrection,
    load_entity_corrections,
    save_entity_corrections,
)


class EntityCorrectionTests(unittest.TestCase):
    def test_manual_roles_determine_analysis_inclusion(self):
        player = TrackCorrection(1, role=EntityRole.PLAYER, team=TeamAssignment.TEAM_A)
        goalkeeper = TrackCorrection(2, role=EntityRole.GOALKEEPER, team=TeamAssignment.TEAM_B)
        parent = TrackCorrection(
            3, role=EntityRole.SPECTATOR, team=TeamAssignment.NONE, excluded=True
        )
        result = EntityCorrectionSet("match.mp4", (player, goalkeeper, parent))

        self.assertEqual(result.included_track_ids(), frozenset((1, 2)))
        self.assertEqual(result.excluded_track_ids(), frozenset((3,)))

    def test_upsert_replaces_existing_track_decision(self):
        original = EntityCorrectionSet(
            "match.mp4", (TrackCorrection(8, role=EntityRole.UNKNOWN),)
        )
        corrected = original.with_correction(
            TrackCorrection(8, role=EntityRole.REFEREE, team=TeamAssignment.OFFICIAL)
        )

        self.assertEqual(len(corrected.corrections), 1)
        self.assertEqual(corrected.get(8).role, EntityRole.REFEREE)

    def test_duplicate_track_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "maar één"):
            EntityCorrectionSet(
                "match.mp4",
                (TrackCorrection(4), TrackCorrection(4)),
            )

    def test_staff_must_be_excluded(self):
        with self.assertRaisesRegex(ValueError, "uitgesloten"):
            TrackCorrection(4, role=EntityRole.STAFF, team=TeamAssignment.NONE)

    def test_json_round_trip(self):
        source = EntityCorrectionSet(
            "videos/match.mp4",
            (
                TrackCorrection(
                    12, role=EntityRole.GOALKEEPER, team=TeamAssignment.TEAM_A,
                    note="Oranje shirt, staat meestal in doel A.",
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrections.json"
            save_entity_corrections(source, path)
            restored = load_entity_corrections(path)

        self.assertEqual(restored, source)

    def test_segment_correction_overrides_track_fallback(self):
        source = EntityCorrectionSet(
            "match.mp4",
            (
                TrackCorrection(23, role=EntityRole.PLAYER, team=TeamAssignment.TEAM_A),
                TrackCorrection(
                    23,
                    segment_index=2,
                    role=EntityRole.PLAYER,
                    team=TeamAssignment.TEAM_B,
                ),
            ),
        )

        self.assertEqual(source.get(23, 1).team, TeamAssignment.TEAM_A)
        self.assertEqual(source.get(23, 2).team, TeamAssignment.TEAM_B)


if __name__ == "__main__":
    unittest.main()
