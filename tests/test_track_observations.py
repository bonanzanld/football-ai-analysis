from __future__ import annotations

import unittest

import numpy as np

from football_ai.tracking.track_manager import TrackManager


class TrackObservationTests(unittest.TestCase):
    def test_manager_keeps_frame_and_box_for_every_observation(self) -> None:
        manager = TrackManager()

        manager._update_track(
            xyxy=np.array([10.0, 20.0, 30.0, 60.0]),
            tracker_id=7,
            confidence=0.8,
            frame_number=12,
        )
        manager._update_track(
            xyxy=np.array([12.0, 22.0, 32.0, 62.0]),
            tracker_id=7,
            confidence=0.9,
            frame_number=15,
        )

        track = manager.get_track(7)

        self.assertIsNotNone(track)
        assert track is not None
        self.assertEqual(track.observation_frames, [12, 15])
        self.assertEqual(
            track.boxes,
            [
                (10.0, 20.0, 30.0, 60.0),
                (12.0, 22.0, 32.0, 62.0),
            ],
        )
        self.assertEqual(track.latest_box, (12.0, 22.0, 32.0, 62.0))
        self.assertEqual(track.frames_seen, 2)


if __name__ == "__main__":
    unittest.main()
