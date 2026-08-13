from tools.review_goalkeeper_boxes import _advance_box_review, _current_answers


def test_box_review_advances_to_unanswered_and_stays_open_when_complete():
    candidates = [{"candidate_id": "a"}, {"candidate_id": "b"}]

    assert _advance_box_review(0, candidates, {"a": {}}) == (1, False)
    assert _advance_box_review(1, candidates, {"a": {}, "b": {}}) == (1, True)


def test_box_review_discards_answers_for_stale_candidate_set():
    candidates = [{"candidate_id": "current"}]
    previous = {"reviews": [
        {"candidate_id": "old", "answer": "keeper"},
        {"candidate_id": "current", "answer": "not_keeper"},
    ]}

    assert list(_current_answers(candidates, previous)) == ["current"]
