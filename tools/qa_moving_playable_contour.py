from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.camera_anchor_bank_3d import load_camera_anchor_bank
from football_ai.calibration.reference_3d import create_field_reference_3d
from football_ai.calibration.video_projection_plan import load_video_projection_plan


FIELD_IDS = ("corner_a_rear", "corner_b_rear", "corner_b_front", "corner_a_front")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Doorlopende QA-video voor een speelveldcontour bij een bewegende camera."
    )
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--start", type=float, default=900.0)
    parser.add_argument("--duration", type=float, default=110.0)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    args = parser.parse_args()

    video_path = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{video_path.stem}_{args.format}"
    plan_path = output_dir / f"{prefix}_projection_plan.json"
    if not plan_path.exists():
        raise RuntimeError(
            "Projectieplan ontbreekt. Voer eerst tools/analyze_pitch_tracking.py uit; "
            "de QA-video bepaalt niet langer zelf het speelveld."
        )
    plan = load_video_projection_plan(plan_path)
    if (
        abs(plan.start_seconds - args.start) > 1e-6
        or abs(plan.duration_seconds - args.duration) > 1e-6
        or abs(plan.interval_seconds - args.interval) > 1e-6
    ):
        raise RuntimeError(
            "Projectieplan hoort bij andere start-, duur- of intervalwaarden. "
            "Voer de vooranalyse opnieuw uit met exact dezelfde opties."
        )
    reference = create_field_reference_3d(create_detection_profile(args.format))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video kon niet worden geopend: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_panel_width = max(360, int(round(width * 0.34)))
    panel_width = int(np.ceil(raw_panel_width / 16.0) * 16)
    output_fps = max(1.0, 1.0 / args.interval)
    output_path = output_dir / f"{prefix}_moving_contour_qa.mp4"
    raw_output_path = output_dir / f"{prefix}_moving_contour_qa_raw.mp4"
    writer = cv2.VideoWriter(
        str(raw_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width + panel_width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"QA-video kon niet worden aangemaakt: {raw_output_path}")

    records: list[dict] = []
    counts = {"valid": 0, "hold": 0, "unknown": 0}
    switches: list[dict] = []
    last_anchor: str | None = None
    last_valid_polygon: np.ndarray | None = None
    last_valid_footprint: np.ndarray | None = None
    last_valid_time: float | None = None
    previous_polygon: np.ndarray | None = None
    frame_diagonal = float(np.hypot(width, height))
    sample_count = len(plan.records)
    projected_records = tuple(item for item in plan.records if item.projection is not None)
    if not projected_records:
        raise RuntimeError("Projectieplan bevat geen enkele bruikbare kandidaatprojectie.")

    for sample_index in range(sample_count):
        planned = plan.records[sample_index]
        time_seconds = planned.time_seconds
        frame_number = planned.frame_number
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        if not success:
            break
        polygon = None
        footprint = None
        anchor_id = planned.anchor_id
        reason = planned.reason
        static_trusted = planned.status == "valid"
        fallback_projection = False
        jump_ratio = None
        status = "unknown"

        candidate_polygon = None
        projection = planned.projection
        render_source = planned
        if projection is None:
            render_source = min(
                projected_records,
                key=lambda item: abs(item.time_seconds - time_seconds),
            )
            projection = render_source.projection
            anchor_id = f"fallback-{render_source.frame_number}"
            fallback_projection = True
            reason = (
                f"Rode fallback uit dichtstbijzijnd analysebeeld "
                f"({abs(render_source.time_seconds - time_seconds):.1f}s verschil); niet meten."
            )
        if projection is not None:
            candidate_polygon = np.asarray(
                [projection.project(reference.landmark(item).point) for item in FIELD_IDS],
                dtype=np.float64,
            )

        if projection is not None and static_trusted:
            polygon = candidate_polygon.copy()
            if previous_polygon is not None:
                jump_ratio = float(
                    np.mean(np.linalg.norm(polygon - previous_polygon, axis=1))
                    / max(frame_diagonal, 1.0)
                )
            if jump_ratio is None or jump_ratio <= 0.12:
                status = "valid"
                previous_polygon = polygon.copy()
                last_valid_polygon = polygon.copy()
                last_valid_time = time_seconds
                footprint = _visible_ground_footprint(
                    polygon,
                    projection.ground_homography(),
                    width,
                    height,
                    reference.pitch_length_m,
                    reference.pitch_width_m,
                )
                last_valid_footprint = None if footprint is None else footprint.copy()
            else:
                reason = f"Onrealistische contoursprong ({jump_ratio:.1%} van beelddiagonaal)."
                polygon = None
        elif projection is not None and not static_trusted:
            polygon = None if candidate_polygon is None else candidate_polygon.copy()

        if (
            status == "unknown"
            and last_valid_polygon is not None
            and last_valid_time is not None
            and time_seconds - last_valid_time <= args.hold_seconds
        ):
            status = "hold"
            polygon = last_valid_polygon.copy()
            footprint = None if last_valid_footprint is None else last_valid_footprint.copy()
            reason = "Kort behoud van laatste geldige contour; niet gebruiken voor metingen."

        if status == "unknown" and polygon is None and candidate_polygon is not None:
            polygon = candidate_polygon.copy()
            if not fallback_projection:
                reason = f"Rode kandidaat: {reason} Niet gebruiken voor metingen."

        counts[status] += 1
        if status == "valid" and anchor_id != last_anchor:
            switches.append({"time_seconds": time_seconds, "anchor_id": anchor_id})
            last_anchor = anchor_id
        local_metrics = SimpleNamespace(
            inliers=render_source.inliers,
            inlier_ratio=render_source.inlier_ratio,
            anchor_coverage=render_source.coverage,
            frame_coverage=render_source.coverage,
            supporting_line_count=render_source.supporting_line_count,
            supporting_line_length_m=render_source.supporting_line_length_m,
        )
        resolved = SimpleNamespace(
            local=None if projection is None else local_metrics,
            recognition=SimpleNamespace(status=SimpleNamespace(value="offline-plan")),
        )
        rendered = _render_frame(
            frame,
            polygon,
            footprint,
            status,
            time_seconds,
            anchor_id,
            reason,
            resolved,
            jump_ratio,
            panel_width,
            reference.pitch_length_m,
            reference.pitch_width_m,
        )
        writer.write(rendered)
        records.append(
            {
                "time_seconds": time_seconds,
                "frame_number": frame_number,
                "status": status,
                "anchor_id": anchor_id,
                "static_anchor_trusted": static_trusted,
                "jump_ratio": jump_ratio,
                "reason": reason,
                "candidate_projection_shown": bool(status == "unknown" and polygon is not None),
                "fallback_projection": fallback_projection,
                "fallback_source_time_seconds": (
                    render_source.time_seconds if fallback_projection else None
                ),
                "recognition": resolved.recognition.status.value,
                "supporting_line_count": planned.supporting_line_count,
                "supporting_line_length_m": planned.supporting_line_length_m,
                "local": None
                if resolved.local is None
                else {
                    "inliers": resolved.local.inliers,
                    "inlier_ratio": resolved.local.inlier_ratio,
                    "anchor_coverage": resolved.local.anchor_coverage,
                    "frame_coverage": resolved.local.frame_coverage,
                },
            }
        )
        if (sample_index + 1) % max(int(round(5.0 / args.interval)), 1) == 0:
            print(f"Verwerkt: {time_seconds - args.start + args.interval:.0f}s / {args.duration:.0f}s")

    capture.release()
    writer.release()
    _transcode_h264(raw_output_path, output_path)
    report_path = output_dir / f"{prefix}_moving_contour_qa.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "video": str(video_path),
                "start_seconds": args.start,
                "duration_seconds": args.duration,
                "sample_interval_seconds": args.interval,
                "projection_plan": str(plan_path),
                "preanalysis_resolved_ratio": plan.resolved_ratio,
                "preanalysis_trusted_ratio": plan.trusted_ratio,
                "counts": counts,
                "anchor_switches": switches,
                "samples": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"Tracking: VALID {counts['valid']} | HOLD {counts['hold']} | "
        f"UNKNOWN {counts['unknown']}"
    )
    print(f"QA-video: {output_path}")
    print(f"QA-rapport: {report_path}")


def _transcode_h264(raw_path: Path, output_path: Path) -> None:
    command = (
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(raw_path),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"H.264-conversie mislukt: {completed.stderr.strip()}")
    raw_path.unlink(missing_ok=True)


def _load_static_anchor_quality(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for anchor_id in report.get("projection_quality", {}):
        result[anchor_id] = all(
            section.get(anchor_id, {}).get("confirmed", section.get(anchor_id, {}).get("valid", False))
            for section in (
                report.get("projection_quality", {}),
                report.get("parallelism_quality", {}),
                report.get("support_alignment_quality", {}),
                report.get("orthogonality_quality", {}),
            )
        )
    return result


def _anchor_is_trusted(anchor_id: str | None, anchor_by_id: dict, quality: dict[str, bool]) -> bool:
    if anchor_id is None:
        return False
    anchor = anchor_by_id[anchor_id]
    primary_id = anchor.parent_anchor_id if anchor.anchor_type == "intermediate" else anchor.anchor_id
    return bool(quality.get(primary_id, False))


def _render_frame(
    frame: np.ndarray,
    polygon: np.ndarray | None,
    footprint: np.ndarray | None,
    status: str,
    time_seconds: float,
    anchor_id: str | None,
    reason: str,
    resolved,
    jump_ratio: float | None,
    panel_width: int,
    pitch_length_m: float,
    pitch_width_m: float,
) -> np.ndarray:
    colors = {"valid": (0, 220, 0), "hold": (0, 165, 255), "unknown": (0, 0, 255)}
    color = colors[status]
    if polygon is not None and np.all(np.isfinite(polygon)):
        points = np.round(polygon).astype(np.int32)
        contour_color = (0, 255, 255) if status == "valid" else color
        cv2.polylines(frame, [points], True, contour_color, 5, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 92), (18, 18, 18), -1)
    cv2.putText(
        frame,
        f"BEWEGENDE CONTOUR-QA | {time_seconds:.1f}s | {status.upper()} | anker {anchor_id or '-'}",
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    local = resolved.local
    metrics = (
        "geen geldige lokale koppeling"
        if local is None
        else f"inliers {local.inliers} | ratio {local.inlier_ratio:.0%} | dekking {local.frame_coverage:.0%}"
    )
    if local is not None:
        metrics += (
            f" | witte lijnsteun {local.supporting_line_count} "
            f"({local.supporting_line_length_m:.1f}m)"
        )
    if jump_ratio is not None:
        metrics += f" | sprong {jump_ratio:.1%}"
    cv2.putText(frame, metrics, (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, reason[:115], (14, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
    panel = _render_topdown_panel(
        panel_width,
        frame.shape[0],
        pitch_length_m,
        pitch_width_m,
        footprint,
        status,
        anchor_id,
    )
    return np.hstack((frame, panel))


def _visible_ground_footprint(
    projected_field: np.ndarray,
    ground_homography: np.ndarray,
    frame_width: int,
    frame_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
) -> np.ndarray | None:
    inverse = np.linalg.inv(np.asarray(ground_homography, dtype=np.float64))
    image_frame = np.asarray(
        ((0.0, 0.0), (frame_width - 1.0, 0.0), (frame_width - 1.0, frame_height - 1.0), (0.0, frame_height - 1.0)),
        dtype=np.float32,
    )
    field_image = np.asarray(projected_field, dtype=np.float32)
    if not cv2.isContourConvex(field_image.reshape(-1, 1, 2)):
        return None
    area, visible_image = cv2.intersectConvexConvex(field_image, image_frame)
    if area <= 0.0 or visible_image is None:
        return None
    visible_image = visible_image.reshape(-1, 2).astype(np.float64)
    homogeneous = np.column_stack((visible_image, np.ones(len(visible_image))))
    projected = (inverse @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) < 1e-9):
        return None
    ground = projected[:, :2] / projected[:, 2:3]
    if not np.all(np.isfinite(ground)):
        return None
    field = np.asarray(
        ((0.0, 0.0), (pitch_length_m, 0.0), (pitch_length_m, pitch_width_m), (0.0, pitch_width_m)),
        dtype=np.float32,
    )
    footprint = cv2.convexHull(ground.astype(np.float32)).reshape(-1, 2)
    clipped_area, intersection = cv2.intersectConvexConvex(footprint, field)
    if clipped_area <= 0.0 or intersection is None:
        return None
    return intersection.reshape(-1, 2).astype(np.float64)


def _render_topdown_panel(
    panel_width: int,
    panel_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    footprint: np.ndarray | None,
    status: str,
    anchor_id: str | None,
) -> np.ndarray:
    panel = np.full((panel_height, panel_width, 3), (24, 64, 28), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, panel_height - 1), (12, 24, 14), 8)
    top = 96
    margin_x = 28
    available_width = panel_width - 2 * margin_x
    available_height = panel_height - top - 40
    scale = min(available_width / pitch_length_m, available_height / pitch_width_m)
    draw_width = pitch_length_m * scale
    draw_height = pitch_width_m * scale
    left = (panel_width - draw_width) / 2.0
    field_top = top + (available_height - draw_height) / 2.0

    def map_points(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        mapped = np.column_stack(
            (left + values[:, 0] * scale, field_top + values[:, 1] * scale)
        )
        return np.round(mapped).astype(np.int32)

    field = np.asarray(
        ((0.0, 0.0), (pitch_length_m, 0.0), (pitch_length_m, pitch_width_m), (0.0, pitch_width_m)),
        dtype=np.float64,
    )
    field_px = map_points(field)
    cv2.fillPoly(panel, [field_px], (34, 112, 45))
    if footprint is not None and status in {"valid", "hold"}:
        footprint_px = map_points(footprint)
        overlay = panel.copy()
        fill = (80, 190, 210) if status == "valid" else (20, 130, 230)
        cv2.fillPoly(overlay, [footprint_px], fill)
        cv2.addWeighted(overlay, 0.42, panel, 0.58, 0.0, panel)
        cv2.polylines(panel, [footprint_px], True, fill, 3, cv2.LINE_AA)

    white = (235, 235, 235)
    yellow = (0, 255, 255)
    cv2.polylines(panel, [field_px], True, yellow, 4, cv2.LINE_AA)
    middle_x = int(round(left + pitch_length_m * scale / 2.0))
    cv2.line(
        panel,
        (middle_x, int(round(field_top))),
        (middle_x, int(round(field_top + draw_height))),
        white,
        2,
        cv2.LINE_AA,
    )
    center = (middle_x, int(round(field_top + draw_height / 2.0)))
    cv2.circle(panel, center, max(3, int(round(6.0 * scale))), white, 2, cv2.LINE_AA)
    goal_half = 2.5 * scale
    goal_depth = max(6, int(round(1.5 * scale)))
    center_y = field_top + draw_height / 2.0
    for x, direction in ((left, -1), (left + draw_width, 1)):
        x_px = int(round(x))
        y1, y2 = int(round(center_y - goal_half)), int(round(center_y + goal_half))
        cv2.rectangle(
            panel,
            (min(x_px, x_px + direction * goal_depth), y1),
            (max(x_px, x_px + direction * goal_depth), y2),
            white,
            2,
        )

    colors = {"valid": (0, 220, 0), "hold": (0, 165, 255), "unknown": (0, 0, 255)}
    cv2.putText(panel, "TOP-DOWN 8v8 | 64 x 42.5 m", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, white, 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"{status.upper()} | anker {anchor_id or '-'}",
        (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        colors[status],
        2,
        cv2.LINE_AA,
    )
    legend = "gekleurd vlak = zichtbaar cameragebied" if footprint is not None else "geen betrouwbaar cameragebied"
    cv2.putText(panel, legend, (18, panel_height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.39, white, 1, cv2.LINE_AA)
    return panel


if __name__ == "__main__":
    main()
