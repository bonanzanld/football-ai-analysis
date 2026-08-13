from football_ai.classification.goalkeeper_appearance_evaluation import leave_one_video_out_appearance


def test_leave_one_video_out_reports_each_source_separately():
    examples = tuple(
        {"video_name": video, "label": label, "feature": [float(label), float(index)]}
        for video in ("a", "b", "c")
        for index, label in enumerate((0, 0, 1, 1))
    )

    result = leave_one_video_out_appearance(examples)

    assert [item["held_out_video"] for item in result["folds"]] == ["a", "b", "c"]
    assert all(item["test_positive"] == 2 and item["test_negative"] == 2 for item in result["folds"])
    assert result["diagnostic_only"] is True
