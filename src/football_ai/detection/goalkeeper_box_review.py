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


def select_competitor_box_candidates(
    review_candidates: dict,
    people: dict,
    *,
    maximum_examples: int = 12,
) -> dict:
    """Select hard alternative people beside confirmed keeper paths for review."""
    if maximum_examples < 1:
        raise ValueError("Maximum aantal voorbeelden moet positief zijn.")
    by_frame = {
        (str(record["goal"]), int(record["frame_number"])): record
        for record in people.get("records", ())
    }
    pool = []
    for window in review_candidates.get("windows", ()):
        if window.get("quality", {}).get("classification") != "consistent_review_candidate":
            continue
        path = window.get("path", ())
        for position, index in zip(("first", "middle", "last"), sorted({0, len(path) // 2, len(path) - 1})):
            keeper = path[index]
            record = by_frame.get((str(window["goal"]), int(keeper["frame_number"])))
            if record is None:
                continue
            alternatives = [
                item for item in record.get("candidates", ())
                if _box_iou(item["box"], keeper["box"]) < 0.20
            ]
            if not alternatives:
                continue
            alternative = max(alternatives, key=lambda item: float(item["goal_proximity_score"]))
            pool.append({
                "candidate_id": (
                    f"competitor:{window['goal']}:{float(window['start_seconds']):.3f}:"
                    f"{int(alternative.get('frame_number', keeper['frame_number']))}:{position}"
                ),
                "goal": str(window["goal"]),
                "window_start_seconds": float(window["start_seconds"]),
                "frame_number": int(keeper["frame_number"]),
                "box": list(map(float, alternative["box"])),
                "footpoint": list(map(float, alternative["footpoint"])),
                "goal_relative_position": (
                    list(map(float, alternative["goal_relative_position"]))
                    if alternative.get("goal_relative_position") is not None else None
                ),
                "selection_reason": "highest_goal_proximity_non_keeper_path_competitor",
            })
    if len(pool) > maximum_examples:
        indices = _spread_indices(len(pool), maximum_examples)
        pool = [pool[index] for index in indices]
    return {
        "schema_version": 1,
        "video_name": str(review_candidates["video_name"]),
        "examples": pool,
    }


def _box_iou(first, second) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def _spread_indices(length: int, count: int) -> tuple[int, ...]:
    if length <= count:
        return tuple(range(length))
    return tuple(round(index * (length - 1) / (count - 1)) for index in range(count)) if count > 1 else (length // 2,)
