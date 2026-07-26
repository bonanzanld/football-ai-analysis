import unittest

import numpy as np

from football_ai.detection.ball_tracking import BallObservation
from football_ai.visualizer import FOOTPOINT_COLOR, draw_ball_observation, draw_footpoint


class FootpointVisualizerTests(unittest.TestCase):
    def test_draws_footpoint_at_bottom_center_of_box(self) -> None:
        frame = np.zeros((100, 120, 3), dtype=np.uint8)

        draw_footpoint(frame, (20, 10, 60, 80))

        self.assertTupleEqual(tuple(frame[80, 40]), FOOTPOINT_COLOR)

    def test_clips_footpoint_to_frame(self) -> None:
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        draw_footpoint(frame, (40, 20, 70, 80))

        self.assertTupleEqual(tuple(frame[49, 49]), FOOTPOINT_COLOR)

    def test_clips_extreme_ball_position_to_frame(self) -> None:
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        observation = BallObservation(
            frame_number=1,
            center=(10**30, -10**30),
            box=(0.0, 0.0, 4.0, 4.0),
            confidence=0.5,
            source="predicted",
        )
        rendered = draw_ball_observation(frame, observation)
        self.assertEqual(rendered.shape, frame.shape)

    def test_ignores_non_finite_ball_position(self) -> None:
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        observation = BallObservation(
            frame_number=1,
            center=(float("nan"), 20.0),
            box=(0.0, 0.0, 4.0, 4.0),
            confidence=0.5,
            source="predicted",
        )
        rendered = draw_ball_observation(frame, observation)
        self.assertTrue(np.array_equal(rendered, frame))


if __name__ == "__main__":
    unittest.main()
