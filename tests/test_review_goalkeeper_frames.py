from tools.review_goalkeeper_frames import _pan, _pan_pixels


def test_pan_moves_source_view_when_zoomed():
    state = {"zoom": 2.0, "center": None}
    _pan(state, (720, 1280, 3), .16, 0)
    assert state["center"] == (742.4, 360.0)


def test_right_drag_pan_uses_rendered_pixel_delta():
    state = {"zoom": 2.0, "center": None, "frame_size": (1280, 720)}
    _pan_pixels(state, 136, 0)
    assert state["center"] == (704.0, 360.0)
