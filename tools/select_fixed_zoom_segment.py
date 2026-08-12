from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.calibration.zoom_segment_intrinsics import (
    ZoomSegmentIntrinsics,
    select_widest_zoom_segment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Selecteer automatisch het verst uitgezoomde stabiele segment.")
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    segments = tuple(
        ZoomSegmentIntrinsics(
            float(item["start_seconds"]),
            float(item["end_seconds"]),
            float(item["reference_seconds"]),
            float(item["focal_length_px"]),
            tuple(map(float, item["principal_point"])),
            float(item["horizontal_fov_degrees"]),
            str(item["evidence"]),
        )
        for item in data.get("segments", ())
    )
    selected = select_widest_zoom_segment(segments)
    payload = {
        "schema_version": 1,
        "video_name": data["video_name"],
        "selection_policy": "minimum_focal_length_most_zoomed_out",
        "fixed_zoom_required": True,
        "selected": asdict(selected),
        "excluded": [asdict(item) for item in segments if item != selected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Geselecteerd: {selected.start_seconds:.1f}-{selected.end_seconds:.1f}s | "
        f"f={selected.focal_length_px:.1f}px | FOV={selected.horizontal_fov_degrees:.1f} graden"
    )


if __name__ == "__main__":
    main()
