from football_ai.detection.goalkeeper_box_review import (
    build_box_review_examples,
    select_box_review_candidates,
)


def _window(start):
    return {
        "goal": "A", "start_seconds": start,
        "quality": {"classification": "ambiguous"},
        "path": [
            {"frame_number": start * 10 + index, "box": [1, 2, 3, 4], "footpoint": [2, 4]}
            for index in range(5)
        ],
    }


def test_selects_three_displayed_boxes_from_spread_ambiguous_windows():
    result = select_box_review_candidates({
        "video_name": "match.mp4", "windows": [_window(index) for index in range(6)]
    }, maximum_windows=3)

    assert len(result["examples"]) == 9
    assert [item["window_start_seconds"] for item in result["examples"][::3]] == [0, 2, 5]


def test_exports_individual_positive_and_negative_box_answers():
    candidates = {"video_name": "match.mp4", "examples": [
        {"candidate_id": "a", "frame_number": 1, "box": [1, 2, 3, 4], "goal": "A"},
        {"candidate_id": "b", "frame_number": 2, "box": [5, 6, 7, 8], "goal": "B"},
    ]}
    reviews = {"human_reviewed": True, "reviews": [
        {"candidate_id": "a", "answer": "keeper"},
        {"candidate_id": "b", "answer": "not_keeper"},
    ]}

    result = build_box_review_examples(candidates, reviews)

    assert [item["label"] for item in result["examples"]] == ["keeper", "not_keeper"]
    assert all(item["review_scope"] == "single_displayed_box" for item in result["examples"])
