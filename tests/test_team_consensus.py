from __future__ import annotations

import unittest

from football_ai.classification.team_consensus import TeamConsensus


class TeamConsensusTests(unittest.TestCase):
    def test_dominant_team_is_applied_to_complete_track(self) -> None:
        consensus = TeamConsensus(minimum_votes=5, minimum_agreement_ratio=0.8)
        for team_id in (0, 0, 0, 1, 0):
            consensus.record([12], {12: team_id})

        result = consensus.finalize([12])[12]

        self.assertTrue(result.is_reliable)
        self.assertEqual(result.team_id, 0)
        self.assertEqual(result.agreement_ratio, 0.8)

    def test_uncertain_track_remains_unknown(self) -> None:
        consensus = TeamConsensus(minimum_votes=4, minimum_agreement_ratio=0.8)
        for team_id in (0, 1, 0, 1):
            consensus.record([7], {7: team_id})

        result = consensus.finalize([7])[7]

        self.assertFalse(result.is_reliable)
        self.assertIsNone(result.team_id)

    def test_frames_without_team_assignment_do_not_vote(self) -> None:
        consensus = TeamConsensus(minimum_votes=1)
        consensus.record([3], {})

        result = consensus.finalize([3])[3]

        self.assertEqual(result.total_votes, 0)
        self.assertIsNone(result.team_id)


if __name__ == "__main__":
    unittest.main()
