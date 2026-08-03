from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from football_ai.detection.ball_tracking import BallCandidate
from tools.analyze_ball import _build_tracker, _load_candidate_cache, _save_candidate_cache


class AnalyzeBallCandidateCacheTests(unittest.TestCase):
    def test_preserves_default_reacquisition_policy(self) -> None:
        tracker = _build_tracker(30.0, 0.05)

        self.assertEqual(tracker.strong_reacquisition_confidence, 0.30)
        self.assertEqual(tracker.weak_reacquisition_confidence, 0.05)
        self.assertEqual(tracker.remote_weak_reacquisition_confidence, 0.30)

    def test_round_trips_candidate_and_player_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.json"
            _save_candidate_cache(
                path,
                source_video=Path("/tmp/match.mp4"),
                fps=29.97,
                frame_candidates=[(BallCandidate((1.0, 2.0, 11.0, 12.0), 0.42),)],
                frame_player_footpoints=[((20.0, 30.0),)],
                frame_player_boxes=[((10.0, 5.0, 30.0, 30.0),)],
                frame_transforms=[np.eye(3, dtype=np.float64)],
                accepted_camera_updates=7,
                rejected_camera_updates=2,
            )

            loaded = _load_candidate_cache(path)

        self.assertEqual(loaded[0], "/tmp/match.mp4")
        self.assertAlmostEqual(loaded[1], 29.97)
        self.assertEqual(loaded[2][0][0].box, (1.0, 2.0, 11.0, 12.0))
        self.assertEqual(loaded[2][0][0].confidence, 0.42)
        self.assertEqual(loaded[3], [((20.0, 30.0),)])
        self.assertEqual(loaded[4], [((10.0, 5.0, 30.0, 30.0),)])
        np.testing.assert_array_equal(loaded[5][0], np.eye(3))
        self.assertEqual(loaded[6:], (7, 2))


if __name__ == "__main__":
    unittest.main()
