from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.train_ball_detector import _validate_dataset


class TrainBallDetectorTests(unittest.TestCase):
    def _dataset(self, root: Path, train_sources: list[str], valid_sources: list[str]) -> Path:
        for split in ("train", "valid"):
            directory = root / split
            directory.mkdir(parents=True)
            (directory / "_annotations.coco.json").write_text("{}", encoding="utf-8")
        (root / "dataset_summary.json").write_text(
            json.dumps(
                {
                    "validation_sources": valid_sources,
                    "splits": {
                        "train": {"source_videos": train_sources},
                        "valid": {"source_videos": valid_sources},
                    },
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_accepts_clip_separated_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory), ["videos/a.mp4"], ["videos/b.mp4"])

            summary = _validate_dataset(dataset)

            self.assertEqual(summary["validation_sources"], ["videos/b.mp4"])

    def test_rejects_source_video_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory), ["videos/a.mp4"], ["videos/a.mp4"])

            with self.assertRaisesRegex(ValueError, "leakage"):
                _validate_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
