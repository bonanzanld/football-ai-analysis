from __future__ import annotations

import json
from pathlib import Path


def build_goalkeeper_window_dataset(resolved_path: Path) -> dict:
    source = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not source.get("human_reviewed", False):
        raise ValueError("Keeperdataset vereist expliciet menselijk beoordeelde vensters.")
    examples = []
    for window in source.get("accepted_keeper_windows", ()):
        path = tuple(window.get("path", ()))
        if not path:
            continue
        # The review UI shows exactly first, middle and last. A positive answer
        # means all three displayed selections were the correct goalkeeper; it
        # does not review the hidden intermediate path frames.
        reviewed_indices = sorted({0, len(path) // 2, len(path) - 1})
        for index in reviewed_indices:
            item = path[index]
            examples.append({
                "frame_number": int(item["frame_number"]),
                "box": list(map(float, item["box"])),
                "label": "goalkeeper",
                "goal": str(window["goal"]),
                "window_start_seconds": float(window["start_seconds"]),
                "window_end_seconds": float(window["end_seconds"]),
                "review_scope": "displayed_first_middle_last_only",
                "provenance": "human_reviewed_three_of_three_goalkeeper_window",
            })
    return {
        "schema_version": 1,
        "video_name": source["video_name"],
        "human_reviewed": True,
        "negative_examples_exported": False,
        "negative_review_semantics": "not_keeper means the three-of-three condition failed; it does not identify which displayed box was wrong",
        "examples": examples,
    }
