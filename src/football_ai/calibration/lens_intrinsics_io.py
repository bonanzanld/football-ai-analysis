from __future__ import annotations

import json
from pathlib import Path

from football_ai.calibration.lens_geometry import LensIntrinsics


def load_lens_intrinsics(
    lens_path: Path,
    *,
    selected_zoom_path: Path | None = None,
) -> tuple[LensIntrinsics, str]:
    """Load measured lens data, or the explicitly selected fixed-zoom fallback.

    Zoom-segment calibration does not estimate radial distortion.  We therefore
    keep it neutral instead of silently borrowing coefficients from another
    zoom state.
    """
    if lens_path.exists():
        data = json.loads(lens_path.read_text(encoding="utf-8"))
        return (
            LensIntrinsics(
                tuple(data["frame_size"]),
                float(data["focal_length_px"]),
                tuple(data["principal_point"]),
                tuple(data["radial_distortion"]),
            ),
            "measured_lens_geometry",
        )
    if selected_zoom_path is None or not selected_zoom_path.exists():
        raise FileNotFoundError(
            f"Geen lenskalibratie gevonden: {lens_path}"
            + ("" if selected_zoom_path is None else f" of {selected_zoom_path}")
        )
    data = json.loads(selected_zoom_path.read_text(encoding="utf-8"))
    selected = data["selected"]
    frame_size = tuple(data.get("frame_size", ()))
    if len(frame_size) != 2:
        principal = tuple(selected["principal_point"])
        frame_size = (int(round(2.0 * principal[0])), int(round(2.0 * principal[1])))
    return (
        LensIntrinsics(
            frame_size,
            float(selected["focal_length_px"]),
            tuple(selected["principal_point"]),
            (0.0, 0.0),
        ),
        "selected_fixed_zoom_zero_distortion",
    )
