import json

import pytest

from football_ai.calibration.lens_intrinsics_io import load_lens_intrinsics


def test_loads_selected_fixed_zoom_without_inventing_distortion(tmp_path):
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps({
        "selected": {
            "focal_length_px": 248.5,
            "principal_point": [640.0, 360.0],
        }
    }))

    lens, source = load_lens_intrinsics(
        tmp_path / "missing.json", selected_zoom_path=selected
    )

    assert lens.frame_size == (1280, 720)
    assert lens.focal_length_px == pytest.approx(248.5)
    assert lens.radial_distortion == (0.0, 0.0)
    assert source == "selected_fixed_zoom_zero_distortion"


def test_prefers_measured_lens_geometry(tmp_path):
    measured = tmp_path / "lens.json"
    measured.write_text(json.dumps({
        "frame_size": [1920, 1080],
        "focal_length_px": 900.0,
        "principal_point": [960.0, 540.0],
        "radial_distortion": [0.1, -0.02],
    }))

    lens, source = load_lens_intrinsics(measured)

    assert lens.radial_distortion == pytest.approx((0.1, -0.02))
    assert source == "measured_lens_geometry"
