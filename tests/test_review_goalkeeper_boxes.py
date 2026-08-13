from tools.review_goalkeeper_boxes import _advance_box_review


def test_box_review_advances_to_unanswered_and_stays_open_when_complete():
    candidates = [{"candidate_id": "a"}, {"candidate_id": "b"}]

    assert _advance_box_review(0, candidates, {"a": {}}) == (1, False)
    assert _advance_box_review(1, candidates, {"a": {}, "b": {}}) == (1, True)
