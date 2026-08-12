import pytest

from football_ai.classification.goalkeeper_window_audit import audit_reviewed_goalkeeper_windows


def _window(proximity, maximum_step, mean_step, appearance):
    return {"quality": {
        "mean_goal_proximity": proximity,
        "maximum_step_ratio": maximum_step,
        "mean_step_ratio": mean_step,
        "appearance_median_distance": appearance,
    }}


def test_audit_preserves_three_of_three_semantics_and_reports_overlap():
    result = audit_reviewed_goalkeeper_windows({
        "accepted_keeper_windows": [_window(.8, .02, .01, .1), _window(.9, .03, .02, .2)],
        "rejected_keeper_windows": [_window(.85, .01, .005, .15)],
        "uncertain_keeper_windows": [],
    })

    assert result["counts"] == {"keeper": 2, "not_three_of_three": 1, "uncertain": 0}
    assert result["feature_summary"]["mean_goal_proximity"]["keeper_mean"] == pytest.approx(.85)
    assert result["feature_summary"]["mean_goal_proximity"]["ranges_overlap"] is True
    assert "individual negative boxes are unknown" in result["review_semantics"]["not_keeper"]


def test_audit_handles_missing_groups_without_inventing_metrics():
    result = audit_reviewed_goalkeeper_windows({"accepted_keeper_windows": []})

    summary = result["feature_summary"]["mean_goal_proximity"]
    assert summary["keeper_mean"] is None
    assert summary["ranges_overlap"] is None
