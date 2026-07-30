from __future__ import annotations

import unittest

from football_ai.tracking.box_interpolation import observations_with_short_gaps
from football_ai.tracking.track_state import TrackState


class BoxInterpolationTests(unittest.TestCase):
    def test_bridges_short_plausible_gap(self) -> None:
        track = TrackState(7, 10, 13)
        track.observation_frames.extend((10, 13))
        track.boxes.extend(((10.0, 20.0, 30.0, 80.0), (16.0, 20.0, 36.0, 80.0)))

        observations = observations_with_short_gaps(track)

        self.assertEqual([item[0] for item in observations], [10, 11, 12, 13])
        self.assertEqual(observations[1][1], (12.0, 20.0, 32.0, 80.0))

    def test_does_not_bridge_long_gap(self) -> None:
        track = TrackState(7, 10, 30)
        track.observation_frames.extend((10, 30))
        track.boxes.extend(((10.0, 20.0, 30.0, 80.0), (20.0, 20.0, 40.0, 80.0)))

        observations = observations_with_short_gaps(track)

        self.assertEqual([item[0] for item in observations], [10, 30])

    def test_rejects_implausible_jump(self) -> None:
        track = TrackState(7, 10, 13)
        track.observation_frames.extend((10, 13))
        track.boxes.extend(((10.0, 20.0, 30.0, 80.0), (800.0, 20.0, 820.0, 80.0)))

        observations = observations_with_short_gaps(track)

        self.assertEqual([item[0] for item in observations], [10, 13])


if __name__ == "__main__":
    unittest.main()
