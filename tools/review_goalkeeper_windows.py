from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Controleer de korte lijst met keepertrajecten.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", default="8v8")
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    source_path = output / f"{prefix}_goalkeeper_review_candidates.json"
    answer_path = output / f"{prefix}_goalkeeper_window_reviews.json"
    candidates = [
        item for item in json.loads(source_path.read_text(encoding="utf-8"))["windows"]
        if item["quality"]["classification"] == "consistent_review_candidate"
    ]
    previous = json.loads(answer_path.read_text(encoding="utf-8")) if answer_path.exists() else {"reviews": []}
    answers = {
        (item["goal"], float(item["start_seconds"]), float(item["end_seconds"])): item
        for item in previous.get("reviews", ())
    }
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    window = "Football AI - keepertrajecten controleren"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1600, 810)
    view = {"zoom": 1.0, "center": None}
    def mouse(event, _x, _y, flags, _data):
        if event == cv2.EVENT_MOUSEWHEEL:
            view["zoom"] = float(np.clip(view["zoom"] * (1.25 if flags > 0 else 0.8), 1.0, 8.0))
            if view["zoom"] == 1.0:
                view["center"] = None
    cv2.setMouseCallback(window, mouse)
    index = 0
    try:
        while index < len(candidates):
            candidate = candidates[index]
            canvas = _render(capture, candidate, index, len(candidates), answers, view)
            cv2.imshow(window, canvas)
            key = cv2.waitKeyEx(30)
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("k"), ord("K"), ord("1")):
                _answer(answers, candidate, "keeper")
                _save(answer_path, video.name, answers)
                index += 1
            elif key in (ord("n"), ord("N"), ord("2")):
                _answer(answers, candidate, "not_keeper")
                _save(answer_path, video.name, answers)
                index += 1
            elif key in (ord("u"), ord("U"), ord("3")):
                _answer(answers, candidate, "uncertain")
                _save(answer_path, video.name, answers)
                index += 1
            elif key in (ord(","), ord("[")) and index > 0:
                index -= 1
                view["zoom"], view["center"] = 1.0, None
            elif key in (ord("."), ord("]")) and index + 1 < len(candidates):
                index += 1
                view["zoom"], view["center"] = 1.0, None
            elif key in (ord("+"), ord("=")):
                view["zoom"] = min(8.0, view["zoom"] * 1.25)
            elif key in (ord("-"), ord("_")):
                view["zoom"] = max(1.0, view["zoom"] / 1.25)
            elif key == ord("0"):
                view["zoom"], view["center"] = 1.0, None
            elif key in (2424832, 65361, 63234, ord("a"), ord("A")):
                _pan(view, -0.16, 0.0)
            elif key in (2555904, 65363, 63235, ord("d"), ord("D")):
                _pan(view, 0.16, 0.0)
            elif key in (2490368, 65362, 63232, ord("w"), ord("W")):
                _pan(view, 0.0, -0.16)
            elif key in (2621440, 65364, 63233, ord("s"), ord("S")):
                _pan(view, 0.0, 0.16)
    finally:
        capture.release()
        cv2.destroyWindow(window)
        _save(answer_path, video.name, answers)
    print(f"Keepertrajectreviews opgeslagen: {answer_path} | {len(answers)}/{len(candidates)}")


def _render(capture, candidate, index, total, answers, view):
    path = candidate["path"]
    samples = (path[0], path[len(path) // 2], path[-1])
    panels = []
    for item in samples:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(item["frame_number"]))
        ok, frame = capture.read()
        if not ok:
            frame = np.zeros((720, 1280, 3), np.uint8)
        box = np.round(item["box"]).astype(int)
        cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 255), 5)
        cv2.circle(frame, tuple(np.round(item["footpoint"]).astype(int)), 8, (255, 0, 255), -1)
        panels.append(cv2.resize(_crop(frame, view), (520, 585)))
    canvas = np.full((810, 1560, 3), 20, np.uint8)
    canvas[:585] = np.hstack(panels)
    key = (candidate["goal"], float(candidate["start_seconds"]), float(candidate["end_seconds"]))
    current = answers.get(key, {}).get("answer", "nog niet beoordeeld")
    lines = (
        f"Traject {index + 1}/{total} | Doel {candidate['goal']} | {candidate['start_seconds']:.1f}-{candidate['end_seconds']:.1f}s",
        "KIES KEEPER ALLEEN als de GELE persoon in ALLE 3 beelden de juiste keeper is.",
        "K of 1 = KEEPER    N of 2 = GEEN KEEPER    U of 3 = ONZEKER",
        f"Huidig: {current} | zoom {view['zoom']:.1f}x | ,/. navigeren | Esc = bewaren en stoppen",
        "Muiswiel of +/- = zoom | pijltjes of W/A/S/D = bewegen | 0 = volledig beeld",
    )
    for row, text in enumerate(lines):
        cv2.putText(canvas, text, (22, 625 + row * 34), cv2.FONT_HERSHEY_SIMPLEX, .68 if row < 2 else .55, (0, 230, 255) if row < 2 else (235, 235, 235), 2 if row < 2 else 1, cv2.LINE_AA)
    return canvas


def _crop(frame, view):
    height, width = frame.shape[:2]
    zoom = float(view["zoom"])
    crop_width, crop_height = width / zoom, height / zoom
    center_x, center_y = view["center"] or (width / 2.0, height / 2.0)
    x0 = int(np.clip(center_x - crop_width / 2.0, 0, width - crop_width))
    y0 = int(np.clip(center_y - crop_height / 2.0, 0, height - crop_height))
    return frame[y0:int(y0 + crop_height), x0:int(x0 + crop_width)]


def _pan(view, dx, dy):
    if view["zoom"] <= 1.0:
        return
    width, height = 1280.0, 720.0
    crop_width, crop_height = width / view["zoom"], height / view["zoom"]
    center_x, center_y = view["center"] or (width / 2.0, height / 2.0)
    view["center"] = (
        float(np.clip(center_x + dx * crop_width, crop_width / 2.0, width - crop_width / 2.0)),
        float(np.clip(center_y + dy * crop_height, crop_height / 2.0, height - crop_height / 2.0)),
    )


def _answer(answers, candidate, answer):
    key = (candidate["goal"], float(candidate["start_seconds"]), float(candidate["end_seconds"]))
    answers[key] = {
        "goal": key[0], "start_seconds": key[1], "end_seconds": key[2], "answer": answer,
        "review_semantics": "keeper means selected person is correct goalkeeper in all three displayed frames",
    }


def _save(path, video_name, answers):
    reviews = sorted(answers.values(), key=lambda item: (item["goal"], item["start_seconds"]))
    path.write_text(json.dumps({
        "schema_version": 2, "video_name": video_name, "human_reviewed": True,
        "keeper_answer_requires": "correct selected goalkeeper in all three displayed frames",
        "not_keeper_semantics": "three-of-three condition failed; failing frame is unknown",
        "reviews": reviews,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
