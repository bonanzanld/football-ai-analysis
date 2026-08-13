from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", default="8v8")
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    frames = json.loads((output / f"{prefix}_goalkeeper_frame_review_candidates.json").read_text())["frames"]
    reviews = {item["frame_id"]: item for item in json.loads((output / f"{prefix}_goalkeeper_frame_reviews.json").read_text())["reviews"]}
    windows = json.loads((output / f"{prefix}_goalkeeper_review_candidates.json").read_text())["windows"]
    selected = {(window["goal"], int(item["frame_number"])): item for window in windows for item in window.get("path", ())}
    mismatches = []
    for frame in frames:
        review = reviews.get(frame["frame_id"])
        automatic = selected.get((frame["goal"], frame["frame_number"]))
        if review is None or review.get("status") != "selected" or automatic is None:
            continue
        human = frame["candidates"][int(review["candidate_index"])]
        if _iou(human["box"], automatic["box"]) < .5:
            mismatches.append((frame, human, automatic))
    capture = cv2.VideoCapture(str(video))
    window_name = "Football AI - keeper mismatch bevestigen"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1400, 850)
    answers = {}
    index = 0
    try:
        while index < len(mismatches):
            frame_info, human, automatic = mismatches[index]
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_info["frame_number"])
            ok, frame = capture.read()
            if not ok:
                break
            for label, candidate, colour in (("H = nieuwe klik", human, (255, 0, 255)), ("A = automatisch/eerder", automatic, (0, 255, 255))):
                box = np.round(candidate["box"]).astype(int)
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), colour, 5)
                cv2.putText(frame, label, (box[0], max(30, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, .75, colour, 2)
            canvas = cv2.resize(frame, (1360, 700))
            canvas = np.vstack((canvas, np.full((130, 1360, 3), 20, np.uint8)))
            lines = (
                f"Mismatch {index + 1}/{len(mismatches)} | {frame_info['frame_id']}",
                "Welke gemarkeerde persoon is de keeper? H = PAARS (nieuwe klik) | A = GEEL (automatisch/eerder) | U = onzeker",
                "Kies alleen tussen de twee gemarkeerde personen.",
            )
            for row, text in enumerate(lines):
                cv2.putText(canvas, text, (18, 738 + row * 34), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKeyEx(30)
            if key in (27, ord("q"), ord("Q")):
                break
            choice = "human" if key in (ord("h"), ord("H")) else "automatic" if key in (ord("a"), ord("A")) else "uncertain" if key in (ord("u"), ord("U")) else None
            if choice:
                answers[frame_info["frame_id"]] = choice
                index += 1
    finally:
        capture.release()
        cv2.destroyWindow(window_name)
    target = output / f"{prefix}_goalkeeper_mismatch_reviews.json"
    target.write_text(json.dumps({"human_reviewed": True, "reviews": answers}, indent=2), encoding="utf-8")
    print(f"Mismatchreviews: {len(answers)}/{len(mismatches)} | {target}")


def _iou(first, second):
    ax1, ay1, ax2, ay2 = map(float, first); bx1, by1, bx2, by2 = map(float, second)
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union else 0.0


if __name__ == "__main__":
    main()
