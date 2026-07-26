from pathlib import Path
import tempfile
import unittest

import numpy as np

from football_ai.calibration.global_ground_registration import (
    GlobalGroundRegistration,
    RegisteredGroundFrame,
    load_global_ground_registration,
    save_global_ground_registration,
)


class GlobalGroundRegistrationTests(unittest.TestCase):
    def test_round_trip_and_nearest_frame(self) -> None:
        frames = (
            RegisteredGroundFrame(10, 1.0, np.eye(3)),
            RegisteredGroundFrame(20, 2.0, np.asarray(((1.0, 0.0, 5.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))),
        )
        registration = GlobalGroundRegistration(
            "match.mp4", "8v8", "goal-a", frames, 1.0, 20, True, "ok"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registration.json"
            save_global_ground_registration(registration, path)
            restored = load_global_ground_registration(path)
        self.assertTrue(restored.solved_for_playable_field)
        self.assertEqual(restored.nearest(1.8).frame_number, 20)
        interpolated = restored.ground_to_image_at(1.5)
        self.assertAlmostEqual(interpolated[0, 2], 2.5)


if __name__ == "__main__":
    unittest.main()
