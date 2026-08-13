from tools.qa_goal_window_people import _select_spread_windows


def _record(goal, time):
    return {"goal": goal, "time_seconds": time}


def test_selects_spread_complete_windows_and_drops_short_fragments():
    records = []
    for start in (0, 10, 20, 30, 40):
        records.extend(_record("A", start + offset) for offset in (0, .5, 1.0))
    records.extend((_record("A", 50), _record("A", 50.5)))

    selected = _select_spread_windows(records, 3)

    assert [item["time_seconds"] for item in selected] == [0, .5, 1, 20, 20.5, 21, 40, 40.5, 41]


def test_selects_each_goal_independently():
    records = [
        *(_record("A", time) for time in (0, .5, 1)),
        *(_record("B", time) for time in (10, 10.5, 11)),
    ]

    assert len(_select_spread_windows(records, 1)) == 6
