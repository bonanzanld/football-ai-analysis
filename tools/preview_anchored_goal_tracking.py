from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Maak een gespreide QA van gevolgde doellijnen.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", default="8v8")
    parser.add_argument("--samples-per-goal", type=int, default=12)
    args = parser.parse_args()
    if args.samples_per_goal < 1:
        parser.error("Aantal voorbeelden moet positief zijn.")
    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    payload = json.loads(
        (output / f"{prefix}_anchored_goal_tracking_qa.json").read_text(encoding="utf-8")
    )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    panels = []
    try:
        for goal in ("A", "B"):
            records = [item for item in payload["records"] if item["goal"] == goal]
            for item in _spread(records, args.samples_per_goal):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(item["frame_number"]))
                ok, frame = capture.read()
                if not ok:
                    continue
                first, second = (tuple(np.round(point).astype(int)) for point in item["ground_points"])
                cv2.line(frame, first, second, (0, 255, 255), 5, cv2.LINE_AA)
                cv2.circle(frame, first, 8, (255, 0, 255), -1)
                cv2.circle(frame, second, 8, (255, 0, 255), -1)
                cv2.putText(
                    frame, f"Doel {goal} | {item['time_seconds']:.1f}s | verschil {item['model_disagreement_px']:.1f}px",
                    (20, 36), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 255), 2, cv2.LINE_AA,
                )
                panels.append(cv2.resize(frame, (480, 270)))
    finally:
        capture.release()
    columns = 4
    rows = int(np.ceil(len(panels) / columns))
    canvas = np.zeros((rows * 270, columns * 480, 3), np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        canvas[row * 270:(row + 1) * 270, column * 480:(column + 1) * 480] = panel
    target = output / f"{prefix}_anchored_goal_tracking_spread_qa.jpg"
    cv2.imwrite(str(target), canvas)
    print(f"Gespreide doeltracking-QA: {target} | {len(panels)} voorbeelden")


def _spread(records, count):
    if len(records) <= count:
        return tuple(records)
    indices = np.linspace(0, len(records) - 1, count).round().astype(int)
    return tuple(records[index] for index in indices)


if __name__ == "__main__":
    main()
