from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.bootstrap.white_line_detection import (
    detect_white_field_lines,
    extract_white_pitch_mask,
)
from football_ai.calibration.image_line_perspective import estimate_sideline_perspective
from football_ai.calibration.lens_geometry import LensIntrinsics
from football_ai.calibration.local_field_atlas import load_local_field_atlas
from football_ai.calibration.local_field_atlas_runtime import (
    FixedPatchTracker,
    LocalFieldAtlasTracker,
    LocalFieldAtlasRuntime,
    sideline_vanishing_error_degrees,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maak een bewegende QA vanuit de lokale veldatlas en parallelle 11v11-lijnen."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, help="Standaard: tot einde video.")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--maximum-vp-error", type=float, default=4.0)
    args = parser.parse_args()

    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    atlas = load_local_field_atlas(output / f"{prefix}_local_field_atlas.json")
    lens = _load_lens(output / f"{prefix}_lens_geometry_qa.json")
    profile = create_detection_profile(args.format)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Video kon niet worden geopend: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = args.duration
    requested_partial_run = args.duration is not None or args.start > 0.0
    if duration is None:
        duration = max(frame_count / fps - args.start, 0.0)
    anchor_frames = {}
    for patch in atlas.patches:
        capture.set(cv2.CAP_PROP_POS_FRAMES, patch.anchor_frame)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Atlasankerframe {patch.anchor_frame} kon niet worden gelezen.")
        anchor_frames[patch.patch_id] = cv2.undistort(
            frame, lens.camera_matrix, lens.distortion_coefficients
        )
    runtime = LocalFieldAtlasRuntime(atlas, anchor_frames)
    tracker = LocalFieldAtlasTracker(
        runtime,
        lambda frame, candidate: _semantic_switch_supported(
            frame, candidate, runtime, atlas, (width, height)
        ),
    )
    boundary_owners = {
        "sideline_rear": "midfield-rear",
        "sideline_front": "midfield-front",
        "end_line_a": "goal-a",
        "end_line_b": "goal-b",
    }

    width, height = lens.frame_size
    suffix = "_sample" if requested_partial_run else ""
    raw_path = output / f"{prefix}_moving_local_atlas_qa{suffix}_raw.mp4"
    video_path = output / f"{prefix}_moving_local_atlas_qa{suffix}.mp4"
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, 1.0 / args.interval), (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("De tijdelijke atlas-QA-video kon niet worden aangemaakt.")

    records = []
    counts = {"valid": 0, "candidate": 0, "unknown": 0}
    samples = int(np.floor(duration / args.interval)) + 1
    sample_frame_numbers = tuple(
        int(round((args.start + index * args.interval) * fps))
        for index in range(samples)
    )
    maximum_tracking_step = max(int(round(0.10 * fps)), 1)
    boundary_projection_plan = _bidirectional_boundary_plan(
        video, sample_frame_numbers, maximum_tracking_step, lens,
        runtime, boundary_owners,
    )
    last_tracking_number = None
    for index in range(samples):
        time_seconds = args.start + index * args.interval
        frame_number = int(round(time_seconds * fps))
        tracking_numbers = _tracking_frame_numbers(
            last_tracking_number, frame_number, maximum_tracking_step
        )
        projection = None
        boundary_projections = boundary_projection_plan.get(frame_number, {})
        corrected = None
        for tracking_number in tracking_numbers:
            capture.set(cv2.CAP_PROP_POS_FRAMES, tracking_number)
            ok, frame = capture.read()
            if not ok:
                break
            corrected = cv2.undistort(frame, lens.camera_matrix, lens.distortion_coefficients)
            projection = tracker.update(corrected)
            last_tracking_number = tracking_number
        if corrected is None or projection is None or last_tracking_number != frame_number:
            break
        status = "unknown"
        polygon = np.asarray(projection.polygon, dtype=np.float64) if projection.polygon else None
        # Physical boundaries have independent, immutable semantic owners.
        # Keep rendering every boundary that is still reliably tracked even
        # when the aggregate field-plane tracker temporarily rejects a frame.
        visible_segments = _compose_verified_boundaries(
            boundary_projections, runtime, atlas, (width, height)
        )
        observed_vp = None
        vp_error = None
        supporting = 0
        manual_support = {}
        boundary_support = _verified_boundary_support(corrected, visible_segments)
        direction_confirmed = False
        reason = projection.reason
        if projection.valid and polygon is not None:
            status = "candidate"
            patch = runtime.patch_by_id[projection.patch_id]
            anchor_to_frame = projection.ground_to_frame @ np.linalg.inv(patch.ground_to_anchor)
            evidence = atlas.visible_evidence(
                projection.patch_id, (width, height), anchor_to_frame
            )
            manual_support = _manual_line_support(
                corrected, atlas, projection.patch_id, anchor_to_frame, lens
            )
            visual = detect_white_field_lines(corrected, profile)
            perspective = estimate_sideline_perspective(
                visual.candidates, polygon, (width, height)
            )
            supporting = len(perspective.supporting_lines)
            if perspective.valid and perspective.vanishing_point is not None:
                observed_vp = perspective.vanishing_point
                vp_error = sideline_vanishing_error_degrees(
                    projection.predicted_vanishing_point,
                    observed_vp,
                    lens.principal_point,
                )
                if vp_error <= args.maximum_vp_error:
                    direction_confirmed = True
                    reason = (
                        "Zijlijnrichting bevestigd; positie wacht nog op achterlijn- of doelsteun."
                    )
                else:
                    reason = (
                        f"Perspectiefafwijking {vp_error:.1f} graden; maximaal "
                        f"{args.maximum_vp_error:.1f}."
                    )
            else:
                reason = "Atlas gekoppeld, maar in dit beeld ontbreken twee lange witte controlelijnen."
            # The support band deliberately covers chalk width, lens correction
            # and a few pixels of registration uncertainty. A solid 8% within
            # that 18px band is meaningful white-paint evidence at 4K.
            confirmed_manual = sum(value >= 0.08 for value in manual_support.values())
            confirmed_end_line = any(
                name.startswith("end_line_") and value >= 0.08
                for name, value in boundary_support.items()
            )
            if confirmed_manual >= 2 and confirmed_end_line:
                status = "valid"
                reason = "Perspectief en witte achterlijn bevestigen samen het atlasvlak."
            elif confirmed_manual >= 2:
                reason = "5m- en 16m-richting bevestigd, maar de witte achterlijn niet."
            elif direction_confirmed and confirmed_end_line:
                status = "valid"
                reason = "Zijlijnrichting en witte achterlijn bevestigen samen het atlasvlak."
        visible_segments = _filter_supported_boundaries(
            visible_segments, boundary_support, direction_confirmed
        )
        counts[status] += 1
        writer.write(
            _render(
                corrected, polygon, visible_segments, status, time_seconds, projection.patch_id,
                projection.inliers, projection.inlier_ratio, projection.coverage,
                supporting, vp_error, reason,
            )
        )
        records.append(
            {
                "time_seconds": time_seconds,
                "frame_number": frame_number,
                "status": status,
                "patch_id": projection.patch_id,
                "inliers": projection.inliers,
                "inlier_ratio": projection.inlier_ratio,
                "coverage": projection.coverage,
                "predicted_vanishing_point": projection.predicted_vanishing_point,
                "observed_vanishing_point": observed_vp,
                "vanishing_point_error_degrees": vp_error,
                "supporting_white_lines": supporting,
                "manual_line_white_support": manual_support,
                "verified_boundary_white_support": boundary_support,
                "tracked_boundaries": [segment.name for segment in visible_segments],
                "reason": reason,
            }
        )
        progress_step = max(int(round(5.0 / args.interval)), 1)
        if (index + 1) % progress_step == 0:
            print(f"Verwerkt: {(index + 1) * args.interval:.0f}s / {duration:.0f}s")
    capture.release()
    writer.release()
    _transcode(raw_path, video_path)

    report_path = output / f"{prefix}_moving_local_atlas_qa{suffix}.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "video": str(video),
                "start_seconds": args.start,
                "duration_seconds": duration,
                "interval_seconds": args.interval,
                "maximum_vanishing_point_error_degrees": args.maximum_vp_error,
                "counts": counts,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"Atlastracking: GELDIG {counts['valid']} | KANDIDAAT {counts['candidate']} | "
        f"ONBEKEND {counts['unknown']}"
    )
    print(f"QA-video: {video_path}")
    print(f"QA-rapport: {report_path}")


def _load_lens(path: Path) -> LensIntrinsics:
    data = json.loads(path.read_text(encoding="utf-8"))
    return LensIntrinsics(
        tuple(data["frame_size"]), float(data["focal_length_px"]),
        tuple(data["principal_point"]), tuple(data["radial_distortion"]),
    )


def _manual_line_support(frame, atlas, patch_id, anchor_to_frame, lens):
    reference = atlas.manual_parallel_lines
    patch = next(item for item in atlas.patches if item.patch_id == patch_id)
    if reference is None or anchor_to_frame is None:
        return {}
    lines = tuple(
        item for item in reference.lines
        if item.line_type in ("goal_area_5m", "penalty_area_16m")
        and item.frame_number == patch.anchor_frame
    )
    if not lines:
        return {}
    _grass, white = extract_white_pitch_mask(frame)
    result = {}
    for line in lines:
        undistorted = lens.undistort_points(
            np.asarray(line.points, dtype=np.float64)
        ).astype(np.float32)
        points = cv2.perspectiveTransform(
            undistorted.reshape(1, -1, 2),
            np.asarray(anchor_to_frame, dtype=np.float64),
        ).reshape(-1, 2)
        center = np.mean(points, axis=0)
        _u, _s, vh = np.linalg.svd(points - center)
        direction = vh[0]
        projection = (points - center) @ direction
        first = center + direction * float(np.min(projection))
        second = center + direction * float(np.max(projection))
        mask = np.zeros_like(white)
        cv2.line(
            mask, tuple(np.rint(first).astype(int)), tuple(np.rint(second).astype(int)),
            255, 18, cv2.LINE_AA,
        )
        pixels = mask > 0
        result[line.line_type] = (
            float(np.count_nonzero(white[pixels])) / max(float(np.count_nonzero(pixels)), 1.0)
        )
    return result


def _verified_boundary_support(frame, segments):
    if not segments:
        return {}
    _grass, white = extract_white_pitch_mask(frame)
    result = {}
    for segment in segments:
        if segment.status != "VISIBLE":
            continue
        mask = np.zeros_like(white)
        cv2.line(
            mask,
            tuple(np.rint(segment.image_start).astype(int)),
            tuple(np.rint(segment.image_end).astype(int)),
            255, 18, cv2.LINE_AA,
        )
        pixels = mask > 0
        result[segment.name] = (
            float(np.count_nonzero(white[pixels])) / max(float(np.count_nonzero(pixels)), 1.0)
        )
    return result


def _semantic_switch_supported(frame, projection, runtime, atlas, frame_size):
    """Validate patch identity; never alter its metric field geometry."""
    if not projection.valid:
        return False
    if projection.patch_id.startswith("midfield"):
        return True
    if projection.patch_id not in ("goal-a", "goal-b"):
        return False
    support = _projection_boundary_support(
        frame, projection, runtime, atlas, frame_size
    )
    own_name = "end_line_a" if projection.patch_id == "goal-a" else "end_line_b"
    return support.get(own_name, 0.0) >= 0.25


def _projection_boundary_support(frame, projection, runtime, atlas, frame_size):
    patch = runtime.patch_by_id[projection.patch_id]
    anchor_to_frame = projection.ground_to_frame @ np.linalg.inv(
        patch.ground_to_anchor
    )
    evidence = atlas.visible_evidence(
        projection.patch_id, frame_size, anchor_to_frame
    )
    return _verified_boundary_support(frame, evidence.boundary_segments)


def _compose_verified_boundaries(projections, runtime, atlas, frame_size):
    """Project every physical boundary only from its own observed local patch."""
    owners = {
        "sideline_rear": "midfield-rear",
        "sideline_front": "midfield-front",
        "end_line_a": "goal-a",
        "end_line_b": "goal-b",
    }
    segments = []
    for boundary_name, patch_id in owners.items():
        if patch_id not in runtime.patch_by_id:
            continue
        candidate = projections.get(boundary_name)
        if candidate is None:
            continue
        if not candidate.valid:
            continue
        patch = runtime.patch_by_id[patch_id]
        if boundary_name not in patch.verified_boundaries:
            continue
        anchor_to_frame = candidate.ground_to_frame @ np.linalg.inv(
            patch.ground_to_anchor
        )
        evidence = atlas.visible_evidence(patch_id, frame_size, anchor_to_frame)
        segments.extend(
            segment for segment in evidence.boundary_segments
            if segment.name == boundary_name and segment.status == "VISIBLE"
        )
    return tuple(segments)


def _filter_supported_boundaries(segments, white_support, sideline_direction_confirmed):
    """Never render a boundary that only looks visually trackable.

    End lines are painted 11v11 lines and therefore need local white-paint
    support.  Cone-based 8v8 sidelines need the independently established
    11v11 perspective direction.  These checks affect rendering only; they do
    not alter the immutable metric field geometry.
    """
    supported = []
    for segment in segments:
        if segment.name.startswith("end_line_"):
            if white_support.get(segment.name, 0.0) >= 0.08:
                supported.append(segment)
        elif segment.name.startswith("sideline_") and sideline_direction_confirmed:
            supported.append(segment)
    return tuple(supported)


def _tracking_frame_numbers(previous, target, maximum_step):
    """Include intermediate frames so temporal geometry never makes a large jump."""
    if previous is None or target <= previous:
        return (target,)
    numbers = list(range(previous + maximum_step, target, maximum_step))
    numbers.append(target)
    return tuple(numbers)


def _bidirectional_boundary_plan(video, sample_frames, maximum_step, lens, runtime, owners):
    """Fill a forward gap only with a backward path proven in overlap."""
    if not sample_frames:
        return {}
    dense = _dense_frame_numbers(sample_frames[0], sample_frames[-1], maximum_step)
    forward = _track_fixed_boundaries(video, dense, lens, runtime, owners)
    backward = _track_fixed_boundaries(video, tuple(reversed(dense)), lens, runtime, owners)
    trusted = set()
    for frame_number in dense:
        for name in owners:
            if _boundary_projections_agree(
                name,
                forward.get(frame_number, {}).get(name),
                backward.get(frame_number, {}).get(name),
                runtime,
                runtime.atlas,
            ):
                trusted.add(name)
    plan = {}
    for frame_number in sample_frames:
        selected = {}
        for name in owners:
            first = forward.get(frame_number, {}).get(name)
            second = backward.get(frame_number, {}).get(name)
            if first is not None and first.valid:
                selected[name] = first
            elif name in trusted and second is not None and second.valid:
                selected[name] = second
        plan[frame_number] = selected
    return plan


def _dense_frame_numbers(first, last, maximum_step):
    numbers = list(range(first, last + 1, maximum_step))
    if not numbers or numbers[-1] != last:
        numbers.append(last)
    return tuple(numbers)


def _track_fixed_boundaries(video, frame_numbers, lens, runtime, owners):
    trackers = {
        name: FixedPatchTracker(runtime, patch_id)
        for name, patch_id in owners.items()
        if patch_id in runtime.patch_by_id
    }
    capture = cv2.VideoCapture(str(video))
    result = {}
    for frame_number in frame_numbers:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        corrected = cv2.undistort(frame, lens.camera_matrix, lens.distortion_coefficients)
        result[frame_number] = {
            name: tracker.update(corrected) for name, tracker in trackers.items()
        }
    capture.release()
    return result


def _boundary_projections_agree(name, first, second, runtime, atlas):
    if first is None or second is None or not first.valid or not second.valid:
        return False
    size = (10_000, 10_000)
    first_segments = _compose_verified_boundaries({name: first}, runtime, atlas, size)
    second_segments = _compose_verified_boundaries({name: second}, runtime, atlas, size)
    if not first_segments or not second_segments:
        return False
    a, b = first_segments[0], second_segments[0]
    endpoint_error = 0.5 * (
        np.linalg.norm(np.asarray(a.image_start) - np.asarray(b.image_start))
        + np.linalg.norm(np.asarray(a.image_end) - np.asarray(b.image_end))
    )
    return float(endpoint_error) <= 24.0


def _render(
    frame, polygon, visible_segments, status, time_seconds, patch_id, inliers, ratio, coverage,
    supporting, vp_error, reason,
):
    colors = {"valid": (0, 255, 255), "candidate": (0, 0, 255), "unknown": (0, 0, 255)}
    color = colors[status]
    if visible_segments:
        for segment in visible_segments:
            segment_color = color if segment.status == "VISIBLE" else (0, 165, 255)
            cv2.line(
                frame,
                tuple(np.rint(segment.image_start).astype(int)),
                tuple(np.rint(segment.image_end).astype(int)),
                segment_color, 7 if segment.status == "VISIBLE" else 4, cv2.LINE_AA,
            )
            midpoint = np.rint(
                (np.asarray(segment.image_start) + np.asarray(segment.image_end)) / 2.0
            ).astype(int)
            label = {
                "end_line_a": "ACHTERLIJN A",
                "end_line_b": "ACHTERLIJN B",
                "sideline_front": "ZIJLIJN VOOR",
                "sideline_rear": "ZIJLIJN ACHTER",
            }.get(segment.name, segment.name)
            cv2.putText(
                frame, label, tuple(midpoint), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                segment_color, 2, cv2.LINE_AA,
            )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 132), (15, 15, 15), -1)
    heading_color = (0, 220, 0) if status == "valid" else (0, 0, 255)
    cv2.putText(
        frame,
        f"LOKALE VELDATLAS | {time_seconds:.1f}s | {status.upper()} | vlak {patch_id or '-'}",
        (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, heading_color, 2, cv2.LINE_AA,
    )
    metrics = (
        f"inliers {inliers} | ratio {ratio:.0%} | dekking {coverage:.0%} | "
        f"witte dwarslijnen {supporting}"
    )
    if vp_error is not None:
        metrics += f" | perspectieffout {vp_error:.2f} graden"
    cv2.putText(frame, metrics, (18, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(frame, reason[:145], (18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def _transcode(raw_path: Path, output_path: Path) -> None:
    completed = subprocess.run(
        (
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
        ),
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"H.264-conversie mislukt: {completed.stderr.strip()}")
    raw_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
