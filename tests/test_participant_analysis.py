from __future__ import annotations

import unittest

from football_ai.classification.participant_analysis import analyze_participants
from football_ai.classification.participant_classifier import ParticipantDecision
from football_ai.tracking.entity_corrections import (
    EntityCorrectionSet,
    EntityRole,
    TeamAssignment,
    TrackCorrection,
)
from football_ai.tracking.entity_review_manifest import (
    EntityReviewManifest,
    ReviewObservation,
    ReviewTrack,
)


class ParticipantAnalysisTests(unittest.TestCase):
    def test_manual_referee_decision_is_authoritative(self) -> None:
        track = ReviewTrack(
            track_id=7,
            first_frame=0,
            last_frame=30,
            frames_seen=31,
            average_confidence=0.9,
            observations=(ReviewObservation(0, (0.0, 0.0, 20.0, 50.0)),),
        )
        manifest = EntityReviewManifest("test.mp4", 30.0, (track,))
        corrections = EntityCorrectionSet(
            "test.mp4",
            (TrackCorrection(7, role=EntityRole.REFEREE, team=TeamAssignment.OFFICIAL),),
        )
        report = analyze_participants(manifest, corrections)
        self.assertEqual(
            report.assessments[0].decision,
            ParticipantDecision.CONFIRMED_REFEREE,
        )

    def test_manual_exclusion_is_authoritative(self) -> None:
        track = ReviewTrack(
            track_id=8,
            first_frame=0,
            last_frame=30,
            frames_seen=31,
            average_confidence=0.9,
            observations=(ReviewObservation(0, (0.0, 0.0, 20.0, 50.0)),),
        )
        manifest = EntityReviewManifest("test.mp4", 30.0, (track,))
        corrections = EntityCorrectionSet(
            "test.mp4",
            (TrackCorrection(8, role=EntityRole.STAFF, team=TeamAssignment.NONE, excluded=True),),
        )
        report = analyze_participants(manifest, corrections)
        self.assertEqual(
            report.assessments[0].decision,
            ParticipantDecision.CONFIRMED_EXCLUDED,
        )


if __name__ == "__main__":
    unittest.main()
