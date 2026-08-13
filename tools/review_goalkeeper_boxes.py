from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Beoordeel afzonderlijke keepervakken.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", default="8v8")
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    candidates = json.loads(
        (output / f"{prefix}_goalkeeper_box_review_candidates.json").read_text(encoding="utf-8")
    )["examples"]
    answer_path = output / f"{prefix}_goalkeeper_box_reviews.json"
    previous = json.loads(answer_path.read_text()) if answer_path.exists() else {"reviews": []}
    answers = _current_answers(candidates, previous)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    window = "Football AI - afzonderlijke keepervakken"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1400, 850)
    view = {"zoom": 1.0, "center": None, "frame_size": (1280, 720)}

    def mouse(event, _x, _y, flags, _data):
        if event == cv2.EVENT_MOUSEWHEEL:
            view["zoom"] = float(np.clip(view["zoom"] * (1.25 if flags > 0 else .8), 1, 8))
    cv2.setMouseCallback(window, mouse)
    index = next((i for i, item in enumerate(candidates) if item["candidate_id"] not in answers), 0)
    completed_message = False
    try:
        while index < len(candidates):
            item = candidates[index]
            capture.set(cv2.CAP_PROP_POS_FRAMES, item["frame_number"])
            ok, frame = capture.read()
            if not ok:
                frame = np.zeros((720, 1280, 3), np.uint8)
            view["frame_size"] = (frame.shape[1], frame.shape[0])
            box = np.round(item["box"]).astype(int)
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 255), 5)
            canvas = cv2.resize(_crop(frame, view), (1360, 700))
            canvas = np.vstack((canvas, np.full((130, 1360, 3), 20, np.uint8)))
            current = answers.get(item["candidate_id"], {}).get("answer", "nog niet beoordeeld")
            lines = (
                f"Vak {index + 1}/{len(candidates)} | Doel {item['goal']} | frame {item['frame_number']} | huidig: {current}",
                "Is alleen de GELE persoon in DIT beeld de keeper? K/1 = JA | N/2 = NEE | U/3 = ONZEKER",
                ("ALLES BEOORDEELD | P = VORIGE | V = VOLGENDE | wijzig met K/N/U | Esc bewaren" if completed_message else
                 "P = VORIGE | V = VOLGENDE | muiswiel of +/- zoom | pijltjes/WASD bewegen | Esc bewaren"),
            )
            for row, text in enumerate(lines):
                cv2.putText(canvas, text, (18, 738 + row * 34), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 230, 255) if row < 2 else (235, 235, 235), 2 if row < 2 else 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKeyEx(30)
            if key in (27, ord("q"), ord("Q")):
                break
            answer = None
            if key in (ord("k"), ord("K"), ord("1")):
                answer = "keeper"
            elif key in (ord("n"), ord("N"), ord("2")):
                answer = "not_keeper"
            elif key in (ord("u"), ord("U"), ord("3")):
                answer = "uncertain"
            if answer:
                answers[item["candidate_id"]] = {"candidate_id": item["candidate_id"], "answer": answer}
                _save(answer_path, video.name, answers)
                index, completed_message = _advance_box_review(index, candidates, answers)
            elif key in (ord("p"), ord("P"), ord(","), ord("[")):
                index = (index - 1) % len(candidates)
                completed_message = False
                view["zoom"], view["center"] = 1.0, None
            elif key in (ord("v"), ord("V"), ord("."), ord("]")):
                index = (index + 1) % len(candidates)
                completed_message = False
                view["zoom"], view["center"] = 1.0, None
            elif key in (ord("+"), ord("=")):
                view["zoom"] = min(8, view["zoom"] * 1.25)
            elif key in (ord("-"), ord("_")):
                view["zoom"] = max(1, view["zoom"] / 1.25)
            elif key == ord("0"):
                view["zoom"], view["center"] = 1.0, None
            elif key in (2424832, 65361, 63234, ord("a"), ord("A")):
                _pan(view, -.16, 0)
            elif key in (2555904, 65363, 63235, ord("d"), ord("D")):
                _pan(view, .16, 0)
            elif key in (2490368, 65362, 63232, ord("w"), ord("W")):
                _pan(view, 0, -.16)
            elif key in (2621440, 65364, 63233, ord("s"), ord("S")):
                _pan(view, 0, .16)
    finally:
        capture.release()
        cv2.destroyWindow(window)
        _save(answer_path, video.name, answers)
    print(f"Keeperboxreviews opgeslagen: {answer_path} | {len(answers)}/{len(candidates)}")


def _crop(frame, view):
    height, width = frame.shape[:2]
    crop_width, crop_height = width / view["zoom"], height / view["zoom"]
    center_x, center_y = view["center"] or (width / 2, height / 2)
    x0 = int(np.clip(center_x - crop_width / 2, 0, width - crop_width))
    y0 = int(np.clip(center_y - crop_height / 2, 0, height - crop_height))
    return frame[y0:int(y0 + crop_height), x0:int(x0 + crop_width)]


def _pan(view, dx, dy):
    if view["zoom"] <= 1:
        return
    width, height = map(float, view["frame_size"])
    crop_width, crop_height = width / view["zoom"], height / view["zoom"]
    center_x, center_y = view["center"] or (width / 2, height / 2)
    view["center"] = (
        float(np.clip(center_x + dx * crop_width, crop_width / 2, width - crop_width / 2)),
        float(np.clip(center_y + dy * crop_height, crop_height / 2, height - crop_height / 2)),
    )


def _save(path, video_name, answers):
    path.write_text(json.dumps({
        "schema_version": 1, "video_name": video_name, "human_reviewed": True,
        "review_semantics": "each answer labels only the single displayed yellow box",
        "reviews": sorted(answers.values(), key=lambda item: item["candidate_id"]),
    }, indent=2), encoding="utf-8")


def _advance_box_review(index, candidates, answers):
    for offset in range(1, len(candidates) + 1):
        candidate_index = (index + offset) % len(candidates)
        if candidates[candidate_index]["candidate_id"] not in answers:
            return candidate_index, False
    return index, True


def _current_answers(candidates, previous):
    candidate_ids = {item["candidate_id"] for item in candidates}
    return {
        item["candidate_id"]: item for item in previous.get("reviews", ())
        if item["candidate_id"] in candidate_ids
    }


if __name__ == "__main__":
    main()
