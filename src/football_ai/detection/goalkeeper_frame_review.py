from __future__ import annotations


def select_frame_review_candidates(people: dict, maximum_per_goal: int = 6) -> dict:
    if maximum_per_goal < 1:
        raise ValueError("Maximum aantal frames moet positief zijn.")
    selected = []
    for goal in ("A", "B"):
        records = sorted(
            (item for item in people.get("records", ()) if item["goal"] == goal and item.get("candidates")),
            key=lambda item: float(item["time_seconds"]),
        )
        indices = _spread_indices(len(records), min(maximum_per_goal, len(records)))
        for index in indices:
            record = records[index]
            selected.append({
                "frame_id": f"{goal}:{int(record['frame_number'])}",
                "goal": goal,
                "frame_number": int(record["frame_number"]),
                "time_seconds": float(record["time_seconds"]),
                "candidates": [
                    {
                        "candidate_index": candidate_index,
                        "box": list(map(float, candidate["box"])),
                        "footpoint": list(map(float, candidate["footpoint"])),
                        "goal_relative_position": candidate.get("goal_relative_position"),
                    }
                    for candidate_index, candidate in enumerate(record["candidates"])
                ],
            })
    return {"schema_version": 1, "video_name": people["video_name"], "frames": selected}


def evaluate_frame_reviews(candidates: dict, reviews: dict, selected_paths: dict) -> dict:
    if not reviews.get("human_reviewed", False):
        raise ValueError("Framereviews moeten menselijk beoordeeld zijn.")
    answers = {item["frame_id"]: item for item in reviews.get("reviews", ())}
    path_boxes = {
        (str(window["goal"]), int(item["frame_number"])): item["box"]
        for window in selected_paths.get("windows", ())
        for item in window.get("path", ())
    }
    detected_visible = selected_correct = 0
    missed_detection = not_visible = 0
    reviewed = 0
    for frame in candidates.get("frames", ()):
        answer = answers.get(frame["frame_id"])
        if answer is None:
            continue
        reviewed += 1
        status = answer["status"]
        if status == "selected":
            detected_visible += 1
            chosen = frame["candidates"][int(answer["candidate_index"])]["box"]
            selected = path_boxes.get((str(frame["goal"]), int(frame["frame_number"])))
            if selected is not None and _box_iou(chosen, selected) >= 0.5:
                selected_correct += 1
        elif status == "not_detected":
            missed_detection += 1
        elif status == "not_visible":
            not_visible += 1
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "reviewed_frames": reviewed,
        "keeper_visible_and_detected": detected_visible,
        "keeper_visible_not_detected": missed_detection,
        "keeper_not_visible_or_uncertain": not_visible,
        "person_detector_recall_when_keeper_visible": (
            detected_visible / (detected_visible + missed_detection)
            if detected_visible + missed_detection else None
        ),
        "path_selection_accuracy_when_keeper_detected": (
            selected_correct / detected_visible if detected_visible else None
        ),
        "correct_selected_paths": selected_correct,
    }


def _spread_indices(length: int, count: int) -> tuple[int, ...]:
    if not length or not count:
        return ()
    if length <= count:
        return tuple(range(length))
    return tuple(round(index * (length - 1) / (count - 1)) for index in range(count)) if count > 1 else (length // 2,)


def _box_iou(first, second) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0
