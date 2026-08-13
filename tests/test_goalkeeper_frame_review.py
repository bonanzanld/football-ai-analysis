from football_ai.detection.goalkeeper_frame_review import (
    contradicted_positive_windows,
    evaluate_frame_reviews,
    select_frame_review_candidates,
)


def test_frame_selection_is_spread_per_goal():
    people = {"video_name": "m.mp4", "records": [
        {"goal": "A", "frame_number": i, "time_seconds": i, "candidates": [{"box": [0, 0, 1, 1], "footpoint": [1, 1]}]}
        for i in range(10)
    ]}
    result = select_frame_review_candidates(people, maximum_per_goal=3)
    assert [item["frame_number"] for item in result["frames"]] == [0, 4, 9]


def test_frame_evaluation_separates_detector_and_path_recall():
    candidates = {"frames": [
        {"frame_id": "A:1", "goal": "A", "frame_number": 1, "candidates": [{"box": [0, 0, 10, 10]}]},
        {"frame_id": "A:2", "goal": "A", "frame_number": 2, "candidates": []},
    ]}
    reviews = {"human_reviewed": True, "reviews": [
        {"frame_id": "A:1", "status": "selected", "candidate_index": 0},
        {"frame_id": "A:2", "status": "not_detected"},
    ]}
    paths = {"windows": [{"goal": "A", "path": [{"frame_number": 1, "box": [0, 0, 10, 10]}]}]}
    result = evaluate_frame_reviews(candidates, reviews, paths)
    assert result["person_detector_recall_when_keeper_visible"] == .5
    assert result["path_selection_accuracy_when_keeper_detected"] == 1.0


def test_reports_positive_window_contradicted_by_frame_review():
    candidates = {"frames": [{
        "frame_id": "A:1", "goal": "A", "frame_number": 1,
        "candidates": [{"box": [20, 0, 30, 10]}],
    }]}
    reviews = {"reviews": [{"frame_id": "A:1", "status": "selected", "candidate_index": 0}]}
    paths = {"windows": [{
        "goal": "A", "start_seconds": 1, "end_seconds": 2,
        "quality": {"classification": "consistent_review_candidate"},
        "path": [{"frame_number": 1, "box": [0, 0, 10, 10]}],
    }]}

    assert contradicted_positive_windows(candidates, reviews, paths) == [{
        "goal": "A", "start_seconds": 1.0, "end_seconds": 2.0, "frame_number": 1,
    }]
