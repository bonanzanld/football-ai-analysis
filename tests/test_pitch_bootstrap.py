import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from football_ai.calibration.bootstrap import (
    BootstrapFrameSample,
    PitchBootstrapAnalyzer,
    PitchBootstrapReport,
    cluster_camera_states,
)


class PitchBootstrapTests(unittest.TestCase):
    def test_groups_visually_distinct_camera_views(self) -> None:
        samples = []
        colors = ((35, 95, 35), (90, 80, 35), (35, 75, 100))
        for index in range(12):
            group = index // 4
            frame = np.full((180, 320, 3), colors[group], dtype=np.uint8)
            x = 35 + group * 100
            cv2.rectangle(frame, (x, 30), (x + 55, 150), (235, 235, 235), 4)
            samples.append(BootstrapFrameSample(index, index * 15, index * 0.5, frame))

        result = cluster_camera_states(samples, requested_cluster_count=3)

        self.assertEqual(len(result.clusters), 3)
        memberships = [set(cluster.sample_indices) for cluster in result.clusters]
        self.assertIn(set(range(0, 4)), memberships)
        self.assertIn(set(range(4, 8)), memberships)
        self.assertIn(set(range(8, 12)), memberships)

    def test_report_serializes_without_embedding_video_frames(self) -> None:
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        samples = [
            BootstrapFrameSample(index, index * 15, index * 0.5, frame.copy())
            for index in range(4)
        ]
        clustering = cluster_camera_states(samples, requested_cluster_count=2)
        report = PitchBootstrapReport(
            source_video="match.mov",
            fps=30.0,
            total_frames=1800,
            requested_duration_seconds=60.0,
            analyzed_duration_seconds=1.5,
            sample_interval_seconds=0.5,
            dense_sample_count=4,
            coverage_scan_interval_seconds=5.0,
            samples=tuple(samples),
            clustering=clustering,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bootstrap.json"
            report.save_json(path)
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["sample_count"], 4)
        self.assertNotIn("frame", data["samples"][0])
        preview = PitchBootstrapAnalyzer.create_contact_sheet(report)
        self.assertEqual(preview.shape[1], 480)


if __name__ == "__main__":
    unittest.main()
