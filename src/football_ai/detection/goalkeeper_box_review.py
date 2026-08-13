from __future__ import annotations


def build_box_review_examples(candidates: dict, reviews: dict) -> dict:
    if not reviews.get("human_reviewed", False):
        raise ValueError("Boxlabels moeten expliciet menselijk beoordeeld zijn.")
    answers = {str(item["candidate_id"]): str(item["answer"]) for item in reviews.get("reviews", ())}
    examples = []
    for item in candidates.get("examples", ()):
        answer = answers.get(str(item["candidate_id"]))
        if answer not in {"keeper", "not_keeper"}:
            continue
        examples.append({
            "frame_number": int(item["frame_number"]),
            "box": list(map(float, item["box"])),
            "label": answer,
            "goal": str(item["goal"]),
            "candidate_id": str(item["candidate_id"]),
            "goal_relative_position": (
                list(map(float, item["goal_relative_position"]))
                if item.get("goal_relative_position") is not None else None
            ),
            "review_scope": "single_displayed_box",
            "provenance": "human_reviewed_single_goalkeeper_box",
        })
    return {
        "schema_version": 1,
        "video_name": str(candidates["video_name"]),
        "human_reviewed": True,
        "examples": examples,
    }


def select_box_review_candidates(payload: dict, maximum_windows: int = 4) -> dict:
    if maximum_windows < 1:
        raise ValueError("Maximum aantal vensters moet positief zijn.")
    windows = [
        item for item in payload.get("windows", ())
        if item.get("quality", {}).get("classification") == "ambiguous" and item.get("path")
    ]
    if len(windows) > maximum_windows:
        indices = _spread_indices(len(windows), maximum_windows)
        windows = [windows[index] for index in indices]
    examples = []
    for window in windows:
        path = window["path"]
        for position, index in zip(("first", "middle", "last"), sorted({0, len(path) // 2, len(path) - 1})):
            item = path[index]
            candidate_id = f"{window['goal']}:{float(window['start_seconds']):.3f}:{int(item['frame_number'])}:{position}"
            examples.append({
                "candidate_id": candidate_id,
                "goal": str(window["goal"]),
                "window_start_seconds": float(window["start_seconds"]),
                "frame_number": int(item["frame_number"]),
                "box": list(map(float, item["box"])),
                "footpoint": list(map(float, item["footpoint"])),
                "goal_relative_position": (
                    list(map(float, item["goal_relative_position"]))
                    if item.get("goal_relative_position") is not None else None
                ),
            })
    return {
        "schema_version": 1,
        "video_name": str(payload["video_name"]),
        "examples": examples,
    }


def _spread_indices(length: int, count: int) -> tuple[int, ...]:
    if length <= count:
        return tuple(range(length))
    return tuple(round(index * (length - 1) / (count - 1)) for index in range(count)) if count > 1 else (length // 2,)
