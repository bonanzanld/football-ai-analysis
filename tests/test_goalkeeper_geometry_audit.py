from football_ai.classification.goalkeeper_geometry_audit import audit_goalkeeper_geometry


def test_geometry_audit_keeps_sources_separate_and_counts_both_labels():
    result = audit_goalkeeper_geometry(({
        "video_name": "a.mp4",
        "examples": [
            {"label": "keeper", "goal_relative_position": [.5, .1]},
            {"label": "not_keeper", "goal_relative_position": [.5, .2]},
            {"label": "not_keeper", "goal_relative_position": [1.5, .2]},
        ],
    },))

    assert result["combined"]["keeper_inside_goal_mouth"] == 1.0
    assert result["combined"]["not_keeper_inside_goal_mouth"] == .5
    assert result["sources"][0]["counts"]["not_keeper"] == {"inside": 1, "total": 2}
    assert result["diagnostic_only"] is True
