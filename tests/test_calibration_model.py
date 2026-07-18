from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from football_ai.calibration.quality_report import (
    assess_calibration_quality,
    calculate_quality_report,
)
from football_ai.pitch.calibration_model import (
    CalibrationKeyframe,
    PitchCalibration,
)
from football_ai.pitch.field_model import create_half_pitch_profile


class PitchCalibrationJsonTests(unittest.TestCase):
    def test_save_and_load_preserves_quality_report(self) -> None:
        calibration = self._create_calibration(with_quality=True)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibration.save(path)
            saved_data = json.loads(path.read_text(encoding="utf-8"))
            restored = PitchCalibration.load(path)

        self.assertIn("quality", saved_data)
        self.assertEqual(saved_data["quality"]["inliers"], 2)
        self.assertEqual(saved_data["quality"]["outliers"], 0)
        self.assertEqual(restored.quality, calibration.quality)

    def test_load_accepts_legacy_json_without_quality(self) -> None:
        calibration = self._create_calibration(with_quality=False)
        legacy_data = calibration.to_dict()
        self.assertNotIn("quality", legacy_data)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_calibration.json"
            path.write_text(json.dumps(legacy_data), encoding="utf-8")
            restored = PitchCalibration.load(path)

        self.assertIsNone(restored.quality)
        with self.assertRaisesRegex(RuntimeError, "geen bruikbaarheidsbeoordeling"):
            restored.require_usable()

    def test_rejects_calibration_with_failed_assessment(self) -> None:
        calibration = self._create_calibration(with_quality=True)
        calibration.quality = assess_calibration_quality(
            calibration.quality,
            pitch_width=42.5,
            pitch_length=64.0,
        )

        self.assertFalse(calibration.is_usable)
        with self.assertRaisesRegex(RuntimeError, "status FAIL"):
            calibration.require_usable()

    def test_keyframes_round_trip_and_nearest_frame_selection(self) -> None:
        calibration = self._create_calibration(with_quality=False)
        first_matrix = np.eye(3)
        second_matrix = np.array(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
        )
        calibration.keyframes = (
            self._keyframe(100, first_matrix),
            self._keyframe(300, second_matrix),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence.json"
            calibration.save(path)
            restored = PitchCalibration.load(path)

        self.assertEqual(len(restored.keyframes), 2)
        np.testing.assert_allclose(
            restored.image_to_pitch_for_frame(260),
            second_matrix,
        )

    @staticmethod
    def _keyframe(
        frame_number: int,
        matrix: np.ndarray,
    ) -> CalibrationKeyframe:
        return CalibrationKeyframe(
            frame_number=frame_number,
            time_seconds=frame_number / 30.0,
            image_to_pitch_matrix=matrix,
            pitch_to_image_matrix=np.linalg.inv(matrix),
            image_corners=np.array(
                [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
            ),
            point_count=4,
            line_point_count=12,
            line_rms_error_pixels=1.0,
        )

    @staticmethod
    def _create_calibration(with_quality: bool) -> PitchCalibration:
        quality = None
        if with_quality:
            points = np.array([[10.0, 20.0], [30.0, 40.0]])
            quality = calculate_quality_report(
                image_points=points,
                pitch_points=points,
                image_to_pitch_matrix=np.eye(3),
            )

        return PitchCalibration(
            profile=create_half_pitch_profile(),
            image_corners=np.array(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            ),
            image_to_pitch_matrix=np.eye(3),
            pitch_to_image_matrix=np.eye(3),
            source_video="test.mov",
            source_frame_number=42,
            source_time_seconds=1.4,
            frame_width=1920,
            frame_height=1080,
            quality=quality,
        )


if __name__ == "__main__":
    unittest.main()
