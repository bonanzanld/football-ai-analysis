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


def summarize_goalkeeper_manifests(manifest_paths: tuple[Path, ...]) -> dict:
    """Combine metadata while preserving video-level evaluation boundaries."""
    sources = []
    total = 0
    for path in manifest_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("human_reviewed", False):
            raise ValueError(f"Keepermanifest is niet menselijk beoordeeld: {path}")
        examples = tuple(payload.get("examples", ()))
        if any(item.get("label") != "goalkeeper" for item in examples):
            raise ValueError(f"Onverwacht niet-positief keeperlabel: {path}")
        total += len(examples)
        sources.append({
            "video_name": str(payload["video_name"]),
            "example_count": len(examples),
            "manifest": str(path),
        })
    video_names = [item["video_name"] for item in sources]
    if len(video_names) != len(set(video_names)):
        raise ValueError("Elke bronvideo mag maar eenmaal voorkomen.")
    return {
        "schema_version": 1,
        "human_reviewed": True,
        "positive_examples": total,
        "negative_examples": 0,
        "source_video_count": len(sources),
        "sources": sources,
        "split_policy": "keep entire source videos together; never split frames from one video across train and validation",
        "limitations": [
            "positive-only data cannot measure precision or train a binary classifier",
            "selection-conditioned examples do not measure missed goalkeeper windows",
        ],
    }
