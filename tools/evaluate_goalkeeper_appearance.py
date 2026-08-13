from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.classification.color_features import extract_shirt_feature
from football_ai.classification.goalkeeper_appearance_evaluation import leave_one_video_out_appearance


def main() -> None:
    root = PROJECT_ROOT / "data" / "goalkeeper_ground_truth"
    manifests = tuple(sorted((*root.glob("*_window_examples.json"), *root.glob("*_box_examples.json"))))
    examples = []
    captures = {}
    try:
        for path in manifests:
            payload = json.loads(path.read_text())
            video_name = str(payload["video_name"])
            capture = captures.setdefault(video_name, cv2.VideoCapture(str(PROJECT_ROOT / "videos" / video_name)))
            for item in payload.get("examples", ()):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(item["frame_number"]))
                ok, frame = capture.read()
                if not ok:
                    continue
                feature = extract_shirt_feature(frame, np.asarray(item["box"], dtype=np.float64))
                if feature is None:
                    continue
                examples.append({
                    "video_name": video_name,
                    "label": int(item["label"] in {"keeper", "goalkeeper"}),
                    "feature": feature.tolist(),
                })
    finally:
        for capture in captures.values():
            capture.release()
    result = leave_one_video_out_appearance(tuple(examples))
    result["usable_examples"] = len(examples)
    target = root / "appearance_evaluation.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Uiterlijk-evaluatie: {target}")


if __name__ == "__main__":
    main()
