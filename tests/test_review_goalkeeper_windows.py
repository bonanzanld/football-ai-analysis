from tools.review_goalkeeper_windows import _candidate_signature, _first_unanswered_index, _pan


def test_pan_uses_source_video_dimensions():
    view = {"zoom": 2.0, "center": None, "frame_size": (3840, 2160)}

    _pan(view, 0.16, 0.0)

    assert view["center"] == (2227.2, 1080.0)


def test_pan_stays_within_zoomed_source_frame():
    view = {"zoom": 4.0, "center": (3800.0, 2100.0), "frame_size": (3840, 2160)}

    _pan(view, 0.16, 0.16)

    assert view["center"] == (3360.0, 1890.0)


def test_candidate_signature_covers_exact_three_displayed_boxes():
    candidate = {"path": [
        {"frame_number": frame, "box": [frame, 2, 3, 4]} for frame in range(7)
    ]}

    assert _candidate_signature(candidate) == (
        "0:0.00,2.00,3.00,4.00|3:3.00,2.00,3.00,4.00|6:6.00,2.00,3.00,4.00"
    )


def test_review_starts_at_first_unanswered_candidate():
    candidates = [
        {"goal": "A", "start_seconds": 1, "end_seconds": 2},
        {"goal": "B", "start_seconds": 3, "end_seconds": 4},
    ]
    answers = {("A", 1.0, 2.0): {"answer": "keeper"}}

    assert _first_unanswered_index(candidates, answers) == 1
