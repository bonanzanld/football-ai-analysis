from football_ai.classification.goal_window_candidates import (
    GoalWindowPerson,
    evaluate_goal_person_path,
    select_continuous_goal_person,
)


def _person(frame, x, proximity):
    return GoalWindowPerson(frame, (x, 100.0), proximity, (x - 5, 50, x + 5, 100))


def test_prefers_continuous_person_over_single_nearer_defender():
    keeper = (_person(1, 100, .8), _person(2, 102, .8), _person(3, 104, .8))
    frames = (
        (keeper[0], _person(1, 250, .9)),
        (keeper[1], _person(2, 20, .99)),
        (keeper[2], _person(3, 260, .95)),
    )
    assert select_continuous_goal_person(frames, frame_diagonal=1000) == keeper


def test_empty_or_incomplete_window_has_no_path():
    assert select_continuous_goal_person((), frame_diagonal=1000) == ()
    assert select_continuous_goal_person(((_person(1, 1, .8),), ()), frame_diagonal=1000) == ()


def test_classifies_smooth_near_goal_path_as_review_candidate():
    path = (_person(1, 100, .9), _person(2, 102, .85), _person(3, 104, .8))
    quality = evaluate_goal_person_path(path, frame_diagonal=1000)
    assert quality.classification == "consistent_review_candidate"
    assert quality.maximum_step_ratio == .002
