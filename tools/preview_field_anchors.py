from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.bootstrap.field_anchor_bank import anchor_visible_polygon, build_field_anchor_bank
from football_ai.calibration.bootstrap.goal_seed import load_goal_seeds
from football_ai.calibration.bootstrap.sideline_anchor import load_sideline_anchors


def main() -> None:
    parser = argparse.ArgumentParser(description="Maak een QA-contactblad van lokale veldankers.")
    parser.add_argument("--video", required=True, help="Bestandsnaam in de videos-map.")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    stem = video.stem
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    goals = load_goal_seeds(output_dir / f"{stem}_{args.format}_goal_seeds.json")
    sidelines = load_sideline_anchors(output_dir / f"{stem}_{args.format}_sideline_anchors.json")
    profile = create_detection_profile(args.format)
    capture = cv2.VideoCapture(str(video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    anchors = build_field_anchor_bank(goals, sidelines, profile.pitch_width_m, (width, height))
    tiles: list[np.ndarray] = []
    for anchor in anchors:
        capture.set(cv2.CAP_PROP_POS_FRAMES, anchor.frame_number)
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Ankerframe {anchor.frame_number} kon niet worden gelezen.")
        polygon = anchor_visible_polygon(anchor, (width, height))
        overlay = frame.copy()
        if len(polygon) >= 3:
            cv2.fillPoly(overlay, [np.round(polygon).astype(np.int32)], (30, 160, 70))
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0.0, frame)
        for line, color in ((anchor.rear_line, (0, 255, 255)), (anchor.front_line, (0, 165, 255)), (anchor.backline, (255, 255, 0))):
            if line is not None:
                cv2.line(frame, tuple(np.round(line[0]).astype(int)), tuple(np.round(line[1]).astype(int)), color, 3, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (width, 44), (18, 18, 18), -1)
        cv2.putText(
            frame,
            f"{anchor.anchor_id} | {anchor.time_seconds:.1f}s | {anchor.observed_boundary_count} vaste grenslijnen",
            (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
        )
        tiles.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))
    capture.release()
    columns = 2
    rows = (len(tiles) + columns - 1) // columns
    blank = np.zeros_like(tiles[0])
    while len(tiles) < rows * columns:
        tiles.append(blank.copy())
    sheet = np.vstack([np.hstack(tiles[row * columns:(row + 1) * columns]) for row in range(rows)])
    output = output_dir / f"{stem}_{args.format}_field_anchors.jpg"
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Contactblad kon niet worden opgeslagen: {output}")
    print(f"Lokale veldankers: {len(anchors)}")
    print(f"QA-contactblad: {output}")


if __name__ == "__main__":
    main()
