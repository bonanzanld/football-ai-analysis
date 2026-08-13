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
    answer_path = output / f"{prefix}_goalkeeper_frame_reviews.json"
    previous = json.loads(answer_path.read_text()) if answer_path.exists() else {"reviews": []}
    valid_ids = {item["frame_id"] for item in frames}
    answers = {item["frame_id"]: item for item in previous.get("reviews", ()) if item["frame_id"] in valid_ids}
    capture = cv2.VideoCapture(str(video))
    window = "Football AI - keeper recall per frame"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1400, 850)
    state = {"zoom": 1.0, "center": None, "hits": [], "clicked": None}

    def mouse(event, x, y, flags, _data):
        if event == cv2.EVENT_MOUSEWHEEL:
            state["zoom"] = float(np.clip(state["zoom"] * (1.25 if flags > 0 else .8), 1, 8))
        elif event == cv2.EVENT_LBUTTONDOWN and y < 700:
            for index, x1, y1, x2, y2 in state["hits"]:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    state["clicked"] = index
                    break
    cv2.setMouseCallback(window, mouse)
    index = next((i for i, item in enumerate(frames) if item["frame_id"] not in answers), 0)
    completed = False
    try:
        while frames:
            item = frames[index]
            capture.set(cv2.CAP_PROP_POS_FRAMES, item["frame_number"])
            ok, frame = capture.read()
            if not ok:
                frame = np.zeros((720, 1280, 3), np.uint8)
            rendered, hits = _render(frame, item, state)
            state["hits"] = hits
            canvas = np.vstack((rendered, np.full((130, 1360, 3), 20, np.uint8)))
            current = answers.get(item["frame_id"], {}).get("status", "nog niet beoordeeld")
            lines = (
                f"Frame {index + 1}/{len(frames)} | doel {item['goal']} | huidig: {current}",
                "KLIK op de echte keeper (of toets 1-9) | X = zichtbaar maar NIET gedetecteerd | U = niet zichtbaar/onzeker",
                ("ALLES BEOORDEELD | P vorige | V volgende | wijzig keuze | Esc bewaren" if completed else
                 "P vorige | V volgende | muiswiel zoom | WASD/pijltjes bewegen | Esc bewaren"),
            )
            for row, text in enumerate(lines):
                cv2.putText(canvas, text, (18, 738 + row * 34), cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 230, 255) if row < 2 else (235, 235, 235), 2 if row < 2 else 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKeyEx(30)
            selected = state.pop("clicked", None)
            state["clicked"] = None
            if key in (27, ord("q"), ord("Q")):
                break
            if ord("1") <= key <= ord("9"):
                selected = key - ord("1")
            answer = None
            if selected is not None and selected < len(item["candidates"]):
                answer = {"frame_id": item["frame_id"], "status": "selected", "candidate_index": selected}
            elif key in (ord("x"), ord("X")):
                answer = {"frame_id": item["frame_id"], "status": "not_detected"}
            elif key in (ord("u"), ord("U")):
                answer = {"frame_id": item["frame_id"], "status": "not_visible"}
            if answer:
                answers[item["frame_id"]] = answer
                _save(answer_path, video.name, answers)
                index, completed = _advance(index, frames, answers)
            elif key in (ord("p"), ord("P"), ord(",")):
                index, completed = (index - 1) % len(frames), False
            elif key in (ord("v"), ord("V"), ord(".")):
                index, completed = (index + 1) % len(frames), False
            elif key in (2424832, 65361, 63234, ord("a"), ord("A")):
                _pan(state, frame.shape, -.16, 0)
            elif key in (2555904, 65363, 63235, ord("d"), ord("D")):
                _pan(state, frame.shape, .16, 0)
            elif key in (2490368, 65362, 63232, ord("w"), ord("W")):
                _pan(state, frame.shape, 0, -.16)
            elif key in (2621440, 65364, 63233, ord("s"), ord("S")):
                _pan(state, frame.shape, 0, .16)
    finally:
        capture.release()
        cv2.destroyWindow(window)
        _save(answer_path, video.name, answers)
    print(f"Keeperframereviews: {len(answers)}/{len(frames)} | {answer_path}")


def _render(frame, item, state):
    height, width = frame.shape[:2]
    zoom = state["zoom"]
    crop_width, crop_height = width / zoom, height / zoom
    center_x, center_y = state["center"] or (width / 2, height / 2)
    x0 = float(np.clip(center_x - crop_width / 2, 0, width - crop_width))
    y0 = float(np.clip(center_y - crop_height / 2, 0, height - crop_height))
    crop = frame[int(y0):int(y0 + crop_height), int(x0):int(x0 + crop_width)].copy()
    scale_x, scale_y = 1360 / crop_width, 700 / crop_height
    hits = []
    for index, candidate in enumerate(item["candidates"]):
        box = candidate["box"]
        transformed = (
            int((box[0] - x0) * scale_x), int((box[1] - y0) * scale_y),
            int((box[2] - x0) * scale_x), int((box[3] - y0) * scale_y),
        )
        hits.append((index, *transformed))
    rendered = cv2.resize(crop, (1360, 700))
    for index, x1, y1, x2, y2 in hits:
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 255, 255), 4)
        cv2.putText(rendered, str(index + 1), (x1, max(24, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
    return rendered, hits


def _pan(state, shape, dx, dy):
    if state["zoom"] <= 1:
        return
    height, width = shape[:2]
    crop_width, crop_height = width / state["zoom"], height / state["zoom"]
    center_x, center_y = state["center"] or (width / 2, height / 2)
    state["center"] = (
        float(np.clip(center_x + dx * crop_width, crop_width / 2, width - crop_width / 2)),
        float(np.clip(center_y + dy * crop_height, crop_height / 2, height - crop_height / 2)),
    )


def _advance(index, frames, answers):
    for offset in range(1, len(frames) + 1):
        candidate_index = (index + offset) % len(frames)
        if frames[candidate_index]["frame_id"] not in answers:
            return candidate_index, False
    return index, True


def _save(path, video_name, answers):
    path.write_text(json.dumps({
        "schema_version": 1, "video_name": video_name, "human_reviewed": True,
        "reviews": sorted(answers.values(), key=lambda item: item["frame_id"]),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
