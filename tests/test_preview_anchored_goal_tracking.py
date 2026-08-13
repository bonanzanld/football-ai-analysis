from tools.preview_anchored_goal_tracking import _spread


def test_spread_keeps_endpoints_and_requested_count():
    records = tuple(range(100))

    selected = _spread(records, 5)

    assert len(selected) == 5
    assert selected[0] == 0
    assert selected[-1] == 99


def test_spread_keeps_short_input_unchanged():
    assert _spread((1, 2), 5) == (1, 2)
