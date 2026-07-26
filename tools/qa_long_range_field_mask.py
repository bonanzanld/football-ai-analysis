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

from football_ai.calibration.bootstrap.anchored_mask_tracker import (
    camera_view_distance,
    match_ground_anchor_transform,
)
from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.bootstrap.goal_seed import load_goal_seeds
from football_ai.calibration.bootstrap.sideline_anchor import load_sideline_anchors
from football_ai.calibration.bootstrap.ground_boundary_tracker import (
    GroundBoundaryTracker,
    transform_field_geometry,
)
from football_ai.calibration.bootstrap.visible_field_mask import (
    build_field_boundary_geometry,
    build_visible_field_mask,
    interpolate_sideline_geometry,
    polygon_from_field_boundaries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test veldcontour over een langere camerapan.")
    parser.add_argument("--format", default="8v8", choices=("6v6", "8v8", "11v11"))
    parser.add_argument("--start", type=float, default=None, help="Starttijd in seconden.")
    parser.add_argument("--duration", type=float, default=None, help="Testduur in seconden.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = PROJECT_ROOT / "videos" / "brandevoortbrab.mov"
    seed_path = PROJECT_ROOT / "output" / "pitch_bootstrap" / f"brandevoortbrab_{args.format}_goal_seeds.json"
    output_path = PROJECT_ROOT / "output" / f"brandevoortbrab_{args.format}_field_mask_ground_qa.mp4"
    profile = create_detection_profile(args.format)
    anchors_path = PROJECT_ROOT / "output" / "pitch_bootstrap" / f"brandevoortbrab_{args.format}_sideline_anchors.json"
    if not anchors_path.exists():
        raise RuntimeError(
            "De oude lange lijninterpolatie is uitgeschakeld. Maak eerst de vijf "
            "tussenankers met: .venv/bin/python tools/seed_pitch_sidelines.py"
        )
    sideline_anchors = load_sideline_anchors(anchors_path)
    if not sideline_anchors:
        raise RuntimeError("Het tussenankerbestand bevat geen camerastanden.")
    raise RuntimeError(
        "Tussenankers zijn aanwezig, maar de oude interpolatieroute blijft bewust uitgeschakeld. "
        "Bouw nu eerst de segmenttracker op deze gecontroleerde ankers."
    )
    seeds = sorted(load_goal_seeds(seed_path), key=lambda item: item.frame_number)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video kon niet worden geopend: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = seeds[0].frame_number if args.start is None else int(round(args.start * fps))
    default_end = seeds[-1].frame_number + int(round(2.0 * fps))
    end_frame = default_end if args.duration is None else start_frame + int(round(args.duration * fps))
    end_frame = min(end_frame, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)

    anchor_data: list[tuple[object, np.ndarray, np.ndarray]] = []
    for seed in seeds:
        capture.set(cv2.CAP_PROP_POS_FRAMES, seed.frame_number)
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Ankerframe {seed.frame_number} kon niet worden gelezen.")
        geometry = build_field_boundary_geometry(seed, profile.pitch_width_m)
        full_mask = build_visible_field_mask(seed, profile.pitch_width_m, (width, height))
        anchor_data.append((seed, frame, geometry, full_mask))

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    success, first_frame = capture.read()
    if not success:
        raise RuntimeError("Startframe kon niet worden gelezen.")
    first_seed, first_anchor_frame, first_geometry, first_full_mask = min(
        anchor_data, key=lambda item: abs(item[0].frame_number - start_frame)
    )
    if first_seed.frame_number == start_frame:
        initial_visible_polygon = first_full_mask.polygon
    else:
        raise ValueError("Kies als start het tijdstip van een bestaand doelanker.")
    tracker = GroundBoundaryTracker(first_frame, first_geometry, include_backline=False)
    anchors_by_frame = {seed.frame_number: (seed, geometry) for seed, _frame, geometry, _mask in anchor_data}

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"QA-video kon niet worden aangemaakt: {output_path}")

    counts = {"local": 0, "anchor": 0, "hold": 0}
    switches: list[tuple[int, str]] = []
    active_anchor = f"doel-{first_seed.goal_id}"
    previous_anchor = active_anchor
    frame_number = start_frame
    frame = first_frame
    while frame_number <= end_frame:
        if frame_number == start_frame:
            polygon = initial_visible_polygon
            mode = "anchor"
            anchor_id = active_anchor
            ratio = 1.0
            boundary_mode = "doellijn verankerd"
        else:
            if frame_number in anchors_by_frame:
                anchor_seed, anchor_geometry = anchors_by_frame[frame_number]
                tracker = GroundBoundaryTracker(frame, anchor_geometry, include_backline=False)
                active_anchor = f"doel-{anchor_seed.goal_id}"
                geometry = anchor_geometry
                mode, ratio = "anchor", 1.0
            else:
                tracked = tracker.update(frame)
                geometry = tracked.geometry
                mode = "local" if tracked.reliable else "hold"
                ratio = tracked.inlier_ratio
                recognition_interval = max(5, int(round(fps / 2.0)))
                if (frame_number - start_frame) % recognition_interval == 0:
                    closest = min(
                        anchor_data,
                        key=lambda item: camera_view_distance(item[1], frame),
                    )
                    matched = match_ground_anchor_transform(
                        closest[1], closest[3].tracking_polygon, frame
                    )
                    if matched is not None:
                        matrix, _matched_points, ratio = matched
                        geometry = transform_field_geometry(closest[2], matrix)
                        tracker = GroundBoundaryTracker(frame, geometry, include_backline=False)
                        active_anchor = f"doel-{closest[0].goal_id}"
                        mode = "anchor"
            distances = [camera_view_distance(item[1], frame) for item in anchor_data]
            closest_distance = min(distances)
            closest_index = int(np.argmin(distances))
            closest_anchor = anchor_data[closest_index]
            confirmed_backline = None
            if closest_distance <= 0.115:
                confirmed_backline = match_ground_anchor_transform(
                    closest_anchor[1], closest_anchor[3].tracking_polygon, frame
                )
            if confirmed_backline is not None:
                matrix, _matched_points, anchor_ratio = confirmed_backline
                geometry = transform_field_geometry(closest_anchor[2], matrix)
                ratio = max(ratio, anchor_ratio)
                boundary_mode = f"doellijn {closest_anchor[0].goal_id} bevestigd"
                include_backline = True
            else:
                boundary_mode = "doellijn buiten beeld"
                include_backline = False
                denominator = max(distances[0] + distances[-1], 1e-9)
                fraction = distances[0] / denominator
                geometry = interpolate_sideline_geometry(
                    anchor_data[0][2], anchor_data[-1][2], fraction, (width, height)
                )
            polygon = polygon_from_field_boundaries(
                geometry, (width, height), include_backline=include_backline
            )
            anchor_id = active_anchor
        counts[mode] += 1
        if anchor_id != previous_anchor:
            switches.append((frame_number, anchor_id))
            previous_anchor = anchor_id

        overlay = frame.copy()
        points = np.round(polygon).astype(np.int32)
        cv2.fillPoly(overlay, [points], (30, 160, 70))
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0.0, frame)
        color = (0, 255, 255) if mode != "hold" else (0, 80, 255)
        _draw_field_boundaries(frame, points, color)
        cv2.rectangle(frame, (0, 0), (width, 54), (20, 20, 20), -1)
        label = f"LANGE CONTOUR-QA | {frame_number / fps:.1f}s | {mode.upper()} | {boundary_mode} | steun {ratio:.0%}"
        cv2.putText(frame, label, (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)

        frame_number += 1
        if frame_number > end_frame:
            break
        success, frame = capture.read()
        if not success:
            break
        if (frame_number - start_frame) % max(1, int(round(5.0 * fps))) == 0:
            print(f"Verwerkt: {(frame_number - start_frame) / fps:.0f}s / {(end_frame - start_frame) / fps:.0f}s")

    capture.release()
    writer.release()
    print(f"Tracking: lokaal {counts['local']} | anker {counts['anchor']} | hold {counts['hold']}")
    if switches:
        print("Ankerwissels: " + " | ".join(f"{number / fps:.1f}s -> {anchor}" for number, anchor in switches))
    else:
        print("Ankerwissels: geen")
    print(f"QA-video: {output_path}")
def _draw_field_boundaries(frame: np.ndarray, polygon: np.ndarray, color: tuple[int, int, int]) -> None:
    """Draw real field boundaries, not artificial clipping edges at the image border."""
    height, width = frame.shape[:2]
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        on_left = first[0] <= 2 and second[0] <= 2
        on_right = first[0] >= width - 3 and second[0] >= width - 3
        on_top = first[1] <= 2 and second[1] <= 2
        on_bottom = first[1] >= height - 3 and second[1] >= height - 3
        if on_left or on_right or on_top or on_bottom:
            continue
        cv2.line(frame, tuple(first), tuple(second), color, 3, cv2.LINE_AA)


if __name__ == "__main__":
    main()
