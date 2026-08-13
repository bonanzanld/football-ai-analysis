from football_ai.classification.goal_window_candidates import (
    GoalWindowPerson,
    appearance_stability_classification,
    evaluate_goal_person_path,
    select_continuous_goal_person,
)


def _person(frame, x, proximity, shirt=None):
    return GoalWindowPerson(frame, (x, 100.0), proximity, (x - 5, 50, x + 5, 100), shirt)


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


def test_rejects_smooth_switch_to_different_shirt_colour():
    red = (1.0, 0.0)
    blue = (0.0, 1.0)
    keeper = (_person(1, 100, .8, red), _person(2, 112, .75, red), _person(3, 124, .75, red))
    frames = (
        (keeper[0],),
        (_person(2, 102, .99, blue), keeper[1]),
        (_person(3, 104, .99, blue), keeper[2]),
    )

    assert select_continuous_goal_person(frames, frame_diagonal=1000) == keeper


def test_continuity_follows_moving_goal_coordinates_during_camera_pan():
    keeper = (
        GoalWindowPerson(1, (100, 100), .95, (1, 2, 3, 4), None, (.5, .1)),
        GoalWindowPerson(2, (300, 100), .95, (1, 2, 3, 4), None, (.51, .1)),
        GoalWindowPerson(3, (500, 100), .95, (1, 2, 3, 4), None, (.52, .1)),
    )
    wrong = (
        GoalWindowPerson(1, (150, 150), .4, (1, 2, 3, 4), None, (1.5, 1.0)),
        GoalWindowPerson(2, (152, 150), .4, (1, 2, 3, 4), None, (1.0, .8)),
        GoalWindowPerson(3, (154, 150), .4, (1, 2, 3, 4), None, (.5, .6)),
    )

    selected = select_continuous_goal_person(tuple(zip(keeper, wrong)), frame_diagonal=1000)

    assert selected == keeper


def test_prefers_person_between_posts_over_smooth_outside_path():
    keeper = tuple(
        GoalWindowPerson(i, (100 + 100 * i, 100), .9, (1, 2, 3, 4), None, (.5, .2))
        for i in range(3)
    )
    outside = tuple(
        GoalWindowPerson(i, (150 + i, 150), .95, (1, 2, 3, 4), None, (1.8, .8))
        for i in range(3)
    )

    assert select_continuous_goal_person(tuple(zip(keeper, outside)), frame_diagonal=1000) == keeper


def test_goal_mouth_candidate_excludes_field_player_far_in_front():
    keeper = GoalWindowPerson(1, (100, 100), .9, (1, 2, 3, 4), None, (.5, -.1))
    field_player = GoalWindowPerson(1, (101, 100), .99, (1, 2, 3, 4), None, (.5, -.7))

    assert select_continuous_goal_person(((keeper, field_player),), frame_diagonal=1000) == (keeper,)


def test_quality_uses_goal_relative_motion_during_camera_pan():
    path = tuple(
        GoalWindowPerson(i, (100 + 200 * i, 100), .9, (1, 2, 3, 4), None, (.5 + .02 * i, -.1))
        for i in range(3)
    )

    assert evaluate_goal_person_path(path, frame_diagonal=1000).classification == "consistent_review_candidate"


def test_classifies_smooth_near_goal_path_as_review_candidate():
    path = (_person(1, 100, .9), _person(2, 102, .85), _person(3, 104, .8))
    quality = evaluate_goal_person_path(path, frame_diagonal=1000)
    assert quality.classification == "consistent_review_candidate"
    assert quality.maximum_step_ratio == .002


def test_appearance_stability_is_only_available_with_three_samples():
    assert appearance_stability_classification((.1, .2)) == "insufficient_evidence"
    assert appearance_stability_classification((.1, .2, .3, .9)) == "stable"
    assert appearance_stability_classification((.4, .5, .6)) == "unstable"
