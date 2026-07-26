from __future__ import annotations

import unittest

from football_ai.calibration.camera_profile import CameraKind, ZoomMode, create_camera_profile


class CameraProfileTests(unittest.TestCase):
    def test_unknown_camera_has_no_hard_optical_assumptions(self) -> None:
        profile = create_camera_profile()
        self.assertEqual(profile.kind, CameraKind.UNKNOWN)
        self.assertIsNone(profile.horizontal_fov_degrees)
        self.assertIsNone(profile.expected_height_m)

    def test_falcon_is_only_an_optional_prior(self) -> None:
        profile = create_camera_profile(CameraKind.XBOTGO_FALCON, zoom_mode=ZoomMode.FIXED)
        self.assertEqual(profile.display_name, "XbotGo Falcon")
        self.assertEqual(profile.zoom_mode, ZoomMode.FIXED)
        self.assertTrue(profile.supports_multiple_lenses)


if __name__ == "__main__":
    unittest.main()
