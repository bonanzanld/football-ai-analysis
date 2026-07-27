from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from football_ai.tracking.entity_review_manifest import (
    EntityReviewManifest,
    ReviewObservation,
    ReviewTrack,
)


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "review_goalkeepers.py"
SPEC = importlib.util.spec_from_file_location("review_goalkeepers", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
from football_ai.calibration.manual_ui_controls import mouse_wheel_direction


class ReviewGoalkeepersTests(unittest.TestCase):
    def test_mouse_wheel_direction_supports_opencv_flags_without_helper(self) -> None:
        self.assertEqual(mouse_wheel_direction(120 << 16), 1)
        self.assertEqual(
            mouse_wheel_direction((-120 & 0xFFFF) << 16),
            -1,
        )

    def test_manifest_contains_only_candidate_tracks(self) -> None:
        tracks = tuple(
            ReviewTrack(
                track_id=track_id,
                first_frame=0,
                last_frame=10,
                frames_seen=10,
                average_confidence=0.9,
                observations=(ReviewObservation(0, (0.0, 0.0, 10.0, 20.0)),),
            )
            for track_id in (1, 2, 3)
        )
        manifest = EntityReviewManifest("test.mp4", 30.0, tracks)
        filtered = MODULE.goalkeeper_review_manifest(manifest, {2, 3})
        self.assertEqual([track.track_id for track in filtered.tracks], [2, 3])


if __name__ == "__main__":
    unittest.main()
