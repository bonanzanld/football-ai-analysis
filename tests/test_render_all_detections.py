from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.render_all_detections import _load_ball_observations


class RenderAllDetectionsTests(unittest.TestCase):
    def test_loads_observations_by_frame_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ball.json"
            path.write_text(
                json.dumps(
                    {
                        "observations": [
                            {
                                "frame_number": 12,
                                "center": [10.0, 20.0],
                                "box": [8.0, 18.0, 12.0, 22.0],
                                "confidence": 0.8,
                                "source": "detected",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = _load_ball_observations(path)
        self.assertEqual(result[12].center, (10.0, 20.0))
        self.assertEqual(result[12].source, "detected")


if __name__ == "__main__":
    unittest.main()
