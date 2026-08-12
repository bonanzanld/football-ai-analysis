import json
from pathlib import Path
from tempfile import TemporaryDirectory

from football_ai.detection.goalkeeper_dataset import build_goalkeeper_window_dataset


def test_preserves_selected_person_scope_for_negative_examples():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "resolved.json"
        window = {"goal": "A", "start_seconds": 1, "end_seconds": 2, "path": [{"frame_number": 30, "box": [1, 2, 3, 4]}]}
        path.write_text(json.dumps({
            "video_name": "match.mp4", "human_reviewed": True,
            "accepted_keeper_windows": [], "rejected_keeper_windows": [window],
        }), encoding="utf-8")
        result = build_goalkeeper_window_dataset(path)
    assert result["examples"][0]["label"] == "not_goalkeeper"
    assert result["examples"][0]["review_scope"] == "selected_person_only"
    assert "may contain another goalkeeper" in result["negative_scope"]
