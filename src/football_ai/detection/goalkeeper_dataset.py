from __future__ import annotations

import json
from pathlib import Path


def build_goalkeeper_window_dataset(resolved_path: Path) -> dict:
    source = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not source.get("human_reviewed", False):
        raise ValueError("Keeperdataset vereist expliciet menselijk beoordeelde vensters.")
    examples = []
    for label, key in (("goalkeeper", "accepted_keeper_windows"), ("not_goalkeeper", "rejected_keeper_windows")):
        for window in source.get(key, ()):
            for item in window.get("path", ()):
                examples.append({
                    "frame_number": int(item["frame_number"]),
                    "box": list(map(float, item["box"])),
                    "label": label,
                    "goal": str(window["goal"]),
                    "window_start_seconds": float(window["start_seconds"]),
                    "window_end_seconds": float(window["end_seconds"]),
                    "review_scope": "selected_person_only",
                    "provenance": "human_reviewed_goalkeeper_window",
                })
    return {
        "schema_version": 1,
        "video_name": source["video_name"],
        "human_reviewed": True,
        "negative_scope": "selected person is not goalkeeper; frame may contain another goalkeeper",
        "examples": examples,
    }
