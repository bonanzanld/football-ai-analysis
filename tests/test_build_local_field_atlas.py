import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_local_field_atlas.py"
SPEC = importlib.util.spec_from_file_location("build_local_field_atlas", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_endline_reflection_keeps_owning_endline_fixed():
    for goal_id, end_x in (("A", 0.0), ("B", 64.0)):
        reflection = MODULE._reflection_about_endline(goal_id, 64.0)
        points = np.asarray(((end_x, 0.0, 1.0), (end_x, 42.5, 1.0)))

        reflected = (reflection @ points.T).T

        np.testing.assert_allclose(reflected, points)


def test_goal_a_and_b_reflect_depth_around_different_ends():
    goal_a = MODULE._reflection_about_endline("A", 64.0)
    goal_b = MODULE._reflection_about_endline("B", 64.0)

    np.testing.assert_allclose(goal_a @ np.asarray((12.0, 10.0, 1.0)), (-12.0, 10.0, 1.0))
    np.testing.assert_allclose(goal_b @ np.asarray((52.0, 10.0, 1.0)), (76.0, 10.0, 1.0))
