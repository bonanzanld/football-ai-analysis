from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.calibration.manual_parallel_lines import load_manual_parallel_lines
from football_ai.calibration.perspective_parallelism import (
    sideline_support_deviation_degrees,
    sideline_rays_from_confirmed_endline,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--corners", type=Path, required=True)
    parser.add_argument("--parallel-lines", type=Path, required=True)
    parser.add_argument("--supports", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    corners = json.loads(args.corners.read_text(encoding="utf-8"))
    frame_number = int(corners["frame_number"])
    reference = load_manual_parallel_lines(args.parallel_lines)
    vanishing = reference.vanishing_point_at_frame(frame_number)
    rays = sideline_rays_from_confirmed_endline(
        tuple(corners["rear_corner"]), tuple(corners["front_corner"]), vanishing
    )
    away_from_vanishing = float(corners["field_x_m"]) > 0.0
    diagnostics = []
    support_points = []
    if args.supports is not None:
        supports = json.loads(args.supports.read_text(encoding="utf-8"))
        for name, (start, _toward) in zip(("rear", "front"), rays):
            support = tuple(supports[f"{name}_sideline_support"])
            observed = sideline_support_deviation_degrees(
                start,
                vanishing,
                support,
                away_from_vanishing=away_from_vanishing,
            )
            support_points.append(support)
            diagnostics.append((name, observed))
    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Hoekframe kon niet worden gelezen")
    height, width = frame.shape[:2]
    for start, toward in rays:
        end = _ray_in_frame(start, toward, width, height, False)
        cv2.line(frame, tuple(np.rint(start).astype(int)), end, (0, 255, 255), 6, cv2.LINE_AA)
        cv2.circle(frame, tuple(np.rint(start).astype(int)), 12, (0, 140, 255), -1, cv2.LINE_AA)
    rear = tuple(np.rint(corners["rear_corner"]).astype(int))
    front = tuple(np.rint(corners["front_corner"]).astype(int))
    cv2.line(frame, rear, front, (255, 0, 255), 7, cv2.LINE_AA)
    for point in support_points:
        cv2.circle(frame, tuple(np.rint(point).astype(int)), 11, (0, 220, 0), -1, cv2.LINE_AA)
    cv2.putText(frame, "PAARS bevestigde achterlijn | GEEL verplichte zijlijnrichting", (22, 48), cv2.FONT_HERSHEY_SIMPLEX, .78, (255, 255, 255), 2, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), frame):
        raise RuntimeError("Preview kon niet worden opgeslagen")
    print(f"Verdwijnpunt: {vanishing}")
    for name, observed in diagnostics:
        print(f"{name}: hoedjesafwijking {observed:.2f} graden | geometrie niet aangepast")
    print(f"Preview: {args.output}")


def _ray_in_frame(start, toward, width, height, away_from_vanishing):
    start = np.asarray(start, dtype=np.float64)
    direction = np.asarray(toward, dtype=np.float64) - start
    if away_from_vanishing:
        direction *= -1.0
    candidates = []
    for axis, maximum in ((0, width - 1.0), (1, height - 1.0)):
        if abs(float(direction[axis])) < 1e-9:
            continue
        for boundary in (0.0, maximum):
            scale = (boundary - start[axis]) / direction[axis]
            point = start + scale * direction
            if scale > 0 and 0 <= point[0] < width and 0 <= point[1] < height:
                candidates.append((scale, point))
    if not candidates:
        return tuple(np.rint(toward).astype(int))
    return tuple(np.rint(max(candidates, key=lambda item: item[0])[1]).astype(int))


if __name__ == "__main__":
    main()
