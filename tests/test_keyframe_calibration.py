from __future__ import annotations

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from football_ai.pitch.field_model import create_half_pitch_profile
from football_ai.pitch.manual_calibrator import (
    FrameLineObservation,
    LandmarkObservation,
    MultiFramePitchCalibrator,
    OpenCvCalibrationApp,
    SelectedFrame,
)


class KeyframeCalibrationTests(unittest.TestCase):
    def test_calculates_each_keyframe_from_its_own_observations(self) -> None:
        calibrator = MultiFramePitchCalibrator(create_half_pitch_profile())
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        calibrator.selected_frames = [
            SelectedFrame(frame, 100, 3.33),
            SelectedFrame(frame.copy(), 300, 10.0),
        ]
        pitch_to_images = (
            np.array(
                [[12.0, 1.5, 150.0], [0.8, 7.5, 80.0], [0.002, 0.001, 1.0]]
            ),
            np.array(
                [[10.5, -1.0, 260.0], [0.4, 7.0, 95.0], [0.001, -0.001, 1.0]]
            ),
        )

        for frame_index, pitch_to_image in enumerate(pitch_to_images):
            for landmark_key in (1, 2, 3, 4):
                pitch_point = calibrator.landmarks[landmark_key].pitch_point
                image_point = self._project([pitch_point], pitch_to_image)[0]
                calibrator.observations.append(
                    LandmarkObservation(
                        frame_index,
                        landmark_key,
                        tuple(image_point),
                    )
                )
            for image_point in self._project(
                [(0.0, 8.0), (0.0, 28.0), (0.0, 52.0)],
                pitch_to_image,
            ):
                calibrator.line_observations.append(
                    FrameLineObservation(frame_index, 3, tuple(image_point))
                )
        keyframes, failures = calibrator._calculate_keyframes()

        self.assertEqual(failures, [])
        self.assertEqual(len(keyframes), 2)
        self.assertTrue(all(item[1].is_valid for item in keyframes))
        self.assertTrue(
            all(item[1].line_rms_error_pixels < 0.1 for item in keyframes)
        )
        preview = calibrator._create_independent_keyframe_preview(
            SimpleNamespace(
                keyframes=tuple(item[1] for item in keyframes),
                quality=None,
            ),
            5.0,
        )
        self.assertEqual(preview.shape, (432, 1280, 3))

    def test_landmark_definitions_follow_top_down_diagram(self) -> None:
        calibrator = MultiFramePitchCalibrator(create_half_pitch_profile())

        self.assertEqual(calibrator.landmarks[1].pitch_point, (0.0, 0.0))
        self.assertEqual(calibrator.landmarks[2].pitch_point, (0.0, 64.0))
        self.assertEqual(calibrator.landmarks[3].pitch_point, (42.5, 0.0))
        self.assertEqual(calibrator.landmarks[4].pitch_point, (42.5, 64.0))
        self.assertIn("Doel A", calibrator.landmarks[5].name)
        self.assertIn("Doel B", calibrator.landmarks[8].name)

    def test_zoom_keeps_click_mapping_in_original_frame_coordinates(self) -> None:
        app = OpenCvCalibrationApp.__new__(OpenCvCalibrationApp)
        app.current_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        app.zoom_factor = 1.0
        app.zoom_center = None
        app.status_message = ""
        app.display_image_rect = (440, 0, 1200, 900)
        app.display_scale = 2.0
        app.display_view_origin = (100.0, 50.0)

        app._change_zoom(2.0, (300.0, 250.0))
        mapped = app._canvas_to_original(640, 200)

        self.assertEqual(app.zoom_factor, 2.0)
        self.assertEqual(app.zoom_center, (300.0, 250.0))
        self.assertEqual(mapped, (200.0, 150.0))

    @staticmethod
    def _project(points, matrix: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(
            np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
            matrix,
        ).reshape(-1, 2)


if __name__ == "__main__":
    unittest.main()
