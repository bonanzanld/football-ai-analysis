from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.camera_projection_3d import CameraProjection3D
from football_ai.calibration.player_projection_evidence import evaluate_player_footpoints
from football_ai.detector import FootballDetector
from football_ai.filtering.player_filter import PlayerFilter


def _projection(homography) -> CameraProjection3D | None:
    if homography is None:
        return None
    ground = np.asarray(homography, dtype=np.float64)
    matrix = np.zeros((3, 4), dtype=np.float64)
    matrix[:, 0] = ground[:, 0]
    matrix[:, 1] = ground[:, 1]
    matrix[:, 2] = (0.0, 0.0, 1.0)
    matrix[:, 3] = ground[:, 2]
    return CameraProjection3D(matrix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a partial local-field atlas with detected player footpoints."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--atlas-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--player-threshold", type=float, default=0.20)
    parser.add_argument("--minimum-player-containment", type=float, default=0.60)
    args = parser.parse_args()
    report = json.loads(args.atlas_report.read_text(encoding="utf-8"))
    profile = create_detection_profile(args.format)
    detector = FootballDetector(player_threshold=args.player_threshold, ball_threshold=0.05)
    player_filter = PlayerFilter(
        minimum_box_height=24, minimum_aspect_ratio=1.15, maximum_aspect_ratio=6.0,
        minimum_foot_y_ratio=0.15, minimum_green_ratio=0.18, pitch_calibration=None,
    )
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    records = []
    try:
        samples = report["records"][::args.stride]
        for index, item in enumerate(samples, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(item["frame_number"]))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {item['frame_number']}")
            _all, people, _balls = detector.detect(frame)
            people = player_filter.filter(frame, people, int(item["frame_number"]))
            footpoints = tuple(
                ((float(box[0]) + float(box[2])) / 2.0, float(box[3]))
                for box in people.xyxy
            )
            evidence = evaluate_player_footpoints(
                _projection(item.get("ground_homography")), footpoints,
                pitch_length_m=profile.pitch_length_m,
                pitch_width_m=profile.pitch_width_m,
                tolerated_outside_m=profile.boundary_layout_tolerance_m,
                severe_outside_m=max(8.0, 2.5 * profile.boundary_layout_tolerance_m),
                minimum_acceptable_ratio=args.minimum_player_containment,
            )
            records.append({
                "frame_number": item["frame_number"], "time_seconds": item["time_seconds"],
                "atlas_status": item["status"], **asdict(evidence),
            })
            print(f"{index}/{len(samples)} frame {item['frame_number']}: {len(footpoints)} players, {evidence.classification}")
    finally:
        capture.release()
    summary = dict(Counter(item["classification"] for item in records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema_version": 1, "diagnostic_only": True,
        "minimum_player_containment": args.minimum_player_containment,
        "summary": summary, "records": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Samenvatting: {summary}")
    print(f"Rapport: {args.output}")


if __name__ == "__main__":
    main()
