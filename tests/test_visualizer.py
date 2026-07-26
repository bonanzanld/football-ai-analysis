import unittest

import numpy as np

from football_ai.visualizer import FOOTPOINT_COLOR, draw_footpoint


class FootpointVisualizerTests(unittest.TestCase):
    def test_draws_footpoint_at_bottom_center_of_box(self) -> None:
        frame = np.zeros((100, 120, 3), dtype=np.uint8)

        draw_footpoint(frame, (20, 10, 60, 80))

        self.assertTupleEqual(tuple(frame[80, 40]), FOOTPOINT_COLOR)

    def test_clips_footpoint_to_frame(self) -> None:
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        draw_footpoint(frame, (40, 20, 70, 80))

        self.assertTupleEqual(tuple(frame[49, 49]), FOOTPOINT_COLOR)


if __name__ == "__main__":
    unittest.main()
