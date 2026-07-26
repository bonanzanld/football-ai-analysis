import unittest

import numpy as np

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.playable_boundary_semantics import (
    BoundaryEvidenceSource,
    PlayableBoundaryBinding,
    PlayableBoundaryRole,
)
from football_ai.calibration.playable_field_contour import (
    create_playable_field_contour,
    validate_playable_contour_geometry,
)


class PlayableFieldContourTests(unittest.TestCase):
    @staticmethod
    def _bindings(confirmed: bool = True) -> tuple[PlayableBoundaryBinding, ...]:
        return (
            PlayableBoundaryBinding(PlayableBoundaryRole.END_LINE_A, BoundaryEvidenceSource.FULL_PITCH_SIDELINE, "left", confirmed),
            PlayableBoundaryBinding(PlayableBoundaryRole.END_LINE_B, BoundaryEvidenceSource.FULL_PITCH_SIDELINE, "right", confirmed),
        )

    def test_creates_nominal_8v8_ground_contour(self) -> None:
        contour = create_playable_field_contour(create_detection_profile("8v8"), self._bindings())
        np.testing.assert_allclose(contour.polygon_ground_m, ((0, 0), (64, 0), (64, 42.5), (0, 42.5)))
        quality = validate_playable_contour_geometry(contour.polygon_ground_m, 64.0, 42.5)
        self.assertTrue(quality.valid)

    def test_rejects_unconfirmed_end_line(self) -> None:
        with self.assertRaises(ValueError):
            create_playable_field_contour(create_detection_profile("8v8"), self._bindings(False))

    def test_rejects_crossed_corner_order(self) -> None:
        quality = validate_playable_contour_geometry(
            np.asarray(((0, 0), (64, 42.5), (64, 0), (0, 42.5))),
            64.0,
            42.5,
        )
        self.assertFalse(quality.valid)


if __name__ == "__main__":
    unittest.main()
