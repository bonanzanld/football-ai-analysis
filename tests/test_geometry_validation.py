from __future__ import annotations

import unittest

import numpy as np

from football_ai.calibration.geometry_validation import (
    validate_projected_pitch_geometry,
)


class ProjectedPitchGeometryTests(unittest.TestCase):
    def test_accepts_convex_perspective_quadrilateral(self) -> None:
        result = validate_projected_pitch_geometry(
            np.array([[100, 100], [900, 160], [760, 620], [180, 650]]),
            frame_width=1000,
            frame_height=700,
        )

        self.assertTrue(result.valid)
        self.assertFalse(result.errors)

    def test_rejects_crossing_field_edges(self) -> None:
        result = validate_projected_pitch_geometry(
            np.array([[100, 100], [900, 600], [900, 100], [100, 600]]),
            frame_width=1000,
            frame_height=700,
        )

        self.assertFalse(result.valid)
        self.assertTrue(any("kruisen" in error for error in result.errors))

    def test_rejects_collapsed_field(self) -> None:
        result = validate_projected_pitch_geometry(
            np.array([[100, 100], [105, 101], [110, 102], [115, 103]]),
            frame_width=1000,
            frame_height=700,
        )

        self.assertFalse(result.valid)
        self.assertGreaterEqual(len(result.errors), 2)


if __name__ == "__main__":
    unittest.main()
