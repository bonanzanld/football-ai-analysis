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

from football_ai.calibration.manual_perspective_reference import load_manual_perspective_reference
from football_ai.calibration.zoom_stable_horizon import (
    ZoomStableSegment,
    select_zoom_stable_horizons,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bouw horizons uitsluitend uit zoomstabiele beelden.")
    parser.add_argument("--perspective", type=Path, required=True)
    parser.add_argument("--frame-graph-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary-margin-seconds", type=float, default=1.0)
    args = parser.parse_args()

    perspective = load_manual_perspective_reference(args.perspective)
    graph = json.loads(args.frame_graph_report.read_text(encoding="utf-8"))
    segments = tuple(
        ZoomStableSegment(
            float(item["start_time_seconds"]),
            float(item["end_time_seconds"]),
            int(item["node_count"]),
        )
        for item in graph.get("zoom_stable_segments", ())
    )
    horizons = select_zoom_stable_horizons(
        perspective.views,
        segments,
        boundary_margin_seconds=args.boundary_margin_seconds,
    )
    payload = {
        "schema_version": 1,
        "video_name": perspective.video_name,
        "zoom_transition_frames_excluded": True,
        "boundary_margin_seconds": args.boundary_margin_seconds,
        "horizons": [asdict(item) for item in horizons],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Zoomstabiele horizons: {len(horizons)}")
    print(f"Rapport: {args.output}")


if __name__ == "__main__":
    main()
