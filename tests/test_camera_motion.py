from __future__ import annotations

import unittest

import numpy as np

from football_ai.calibration.camera_motion import (
    CameraMotionKeyframe,
    CameraMotionTrajectory,
)


class CameraMotionTrajectoryTests(unittest.TestCase):
    def test_interpolates_affine_camera_motion_between_keyframes(self) -> None:
        trajectory = self._trajectory()

        matrix = trajectory.frame_to_panorama_for_frame(50)

        np.testing.assert_allclose(
            matrix,
            np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]]),
        )

    def test_clamps_frames_outside_trajectory(self) -> None:
        trajectory = self._trajectory()

        np.testing.assert_allclose(
            trajectory.frame_to_panorama_for_frame(-10),
            trajectory.keyframes[0].frame_to_panorama_matrix,
        )
        np.testing.assert_allclose(
            trajectory.frame_to_panorama_for_frame(500),
            trajectory.keyframes[-1].frame_to_panorama_matrix,
        )

    def test_json_round_trip_preserves_interpolation(self) -> None:
        trajectory = self._trajectory()

        restored = CameraMotionTrajectory.from_dict(trajectory.to_dict())

        np.testing.assert_allclose(
            restored.image_to_pitch_for_frame(25),
            trajectory.image_to_pitch_for_frame(25),
        )

    def test_rejects_non_affine_camera_keyframe(self) -> None:
        with self.assertRaisesRegex(ValueError, "affine"):
            CameraMotionKeyframe(
                0,
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.01, 0.0, 1.0]]
                ),
            )

    @staticmethod
    def _trajectory() -> CameraMotionTrajectory:
        return CameraMotionTrajectory(
            panorama_to_pitch_matrix=np.array(
                [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]]
            ),
            keyframes=(
                CameraMotionKeyframe(0, np.eye(3)),
                CameraMotionKeyframe(
                    100,
                    np.array(
                        [
                            [1.0, 0.0, 200.0],
                            [0.0, 1.0, 40.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ),
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
