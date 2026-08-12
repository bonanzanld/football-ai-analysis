import json
from pathlib import Path
from tempfile import TemporaryDirectory

from football_ai.detection.goalkeeper_dataset import build_goalkeeper_window_dataset


def test_exports_only_three_displayed_positive_boxes_and_no_ambiguous_negatives():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "resolved.json"
        positive = {"goal": "A", "start_seconds": 1, "end_seconds": 2, "path": [
            {"frame_number": frame, "box": [1, 2, 3, 4]} for frame in range(7)
        ]}
        negative = {"goal": "B", "start_seconds": 3, "end_seconds": 4, "path": [
            {"frame_number": 99, "box": [5, 6, 7, 8]}
        ]}
        path.write_text(json.dumps({
            "video_name": "match.mp4", "human_reviewed": True,
            "accepted_keeper_windows": [positive], "rejected_keeper_windows": [negative],
        }), encoding="utf-8")
        result = build_goalkeeper_window_dataset(path)
    assert [item["frame_number"] for item in result["examples"]] == [0, 3, 6]
    assert {item["label"] for item in result["examples"]} == {"goalkeeper"}
    assert result["negative_examples_exported"] is False
