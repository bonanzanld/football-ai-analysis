from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detection.ball_tracking import (
    BallCandidate,
    BallObservation,
    BallTracker,
    PlayerContext,
    best_local_search_anchor,
    candidates_from_detections,
    exclude_candidates_inside_people,
    hold_stationary_detected_gaps,
    interpolate_detected_gaps,
    offset_ball_candidate,
    save_ball_observations,
)
from football_ai.detector import FootballDetector
from football_ai.privacy import anonymize_people_heads
from football_ai.classification.team_classifier import TeamClassifier
from football_ai.tracker import FootballTracker
from football_ai.tracking.online_camera_motion import (
    OnlineCameraMotion,
    transform_box,
    transform_point,
)
from football_ai.visualizer import draw_ball_observation


def _candidate_to_reference(
    candidate: BallCandidate,
    current_to_reference: np.ndarray,
) -> BallCandidate:
    return BallCandidate(
        box=transform_box(candidate.box, current_to_reference),
        confidence=candidate.confidence,
    )


def _local_ball_candidates(
    detector: FootballDetector,
    frame: np.ndarray,
    center: tuple[float, float],
    crop_width: int | None = None,
    crop_height: int | None = None,
) -> list[BallCandidate]:
    """Run a higher-detail detector pass around the last credible ball area."""

    height, width = frame.shape[:2]
    if crop_width is None:
        crop_width = min(1650, max(480, int(round(width * 0.55))))
    if crop_height is None:
        crop_height = min(850, max(320, int(round(height * 0.60))))
    actual_width = min(width, max(1, int(crop_width)))
    actual_height = min(height, max(1, int(crop_height)))
    x1 = int(round(center[0] - actual_width / 2.0))
    y1 = int(round(center[1] - actual_height / 2.0))
    x1 = max(0, min(width - actual_width, x1))
    y1 = max(0, min(height - actual_height, y1))
    crop = frame[y1 : y1 + actual_height, x1 : x1 + actual_width]
    _, _, ball_detections = detector.detect(crop)
    return [
        offset_ball_candidate(candidate, x1, y1)
        for candidate in candidates_from_detections(ball_detections)
        # In a detail crop the real ball in this 4K source spans roughly
        # 18-23 pixels. Tiny 5-10 pixel hits are predominantly fixed field or
        # background texture and must not reach the temporal tracker.
        if min(candidate.size) >= 12.0
    ]


def _observations_to_image_space(
    observations: list[BallObservation] | tuple[BallObservation, ...],
    frame_transforms: list[np.ndarray],
) -> list[BallObservation]:
    converted: list[BallObservation] = []
    for observation in observations:
        if observation.frame_number >= len(frame_transforms):
            continue
        try:
            reference_to_image = np.linalg.inv(
                frame_transforms[observation.frame_number]
            )
        except np.linalg.LinAlgError:
            reference_to_image = np.eye(3, dtype=np.float64)
        converted.append(
            BallObservation(
                frame_number=observation.frame_number,
                center=transform_point(observation.center, reference_to_image),
                box=transform_box(observation.box, reference_to_image),
                confidence=observation.confidence,
                source=observation.source,
                track_segment=observation.track_segment,
            )
        )
    return converted


def _boxes_to_image_space(
    boxes: tuple[tuple[float, float, float, float], ...],
    current_to_reference: np.ndarray,
) -> np.ndarray:
    try:
        reference_to_image = np.linalg.inv(current_to_reference)
    except np.linalg.LinAlgError:
        reference_to_image = np.eye(3, dtype=np.float64)
    return np.asarray(
        [transform_box(box, reference_to_image) for box in boxes],
        dtype=np.float64,
    ).reshape(-1, 4)


def _save_candidate_cache(
    path: Path,
    *,
    source_video: Path,
    fps: float,
    frame_candidates: list[tuple[BallCandidate, ...]],
    frame_player_footpoints: list[tuple[tuple[float, float], ...]],
    frame_player_boxes: list[tuple[tuple[float, float, float, float], ...]],
    frame_transforms: list[np.ndarray],
    accepted_camera_updates: int,
    rejected_camera_updates: int,
    frame_player_contexts: list[tuple[PlayerContext, ...]] | None = None,
) -> None:
    if frame_player_contexts is None:
        frame_player_contexts = [tuple() for _ in frame_candidates]
    payload = {
        "schema_version": 2,
        "source_video": str(source_video),
        "fps": float(fps),
        "accepted_camera_updates": int(accepted_camera_updates),
        "rejected_camera_updates": int(rejected_camera_updates),
        "frames": [
            {
                "candidates": [
                    {"box": list(candidate.box), "confidence": candidate.confidence}
                    for candidate in candidates
                ],
                "player_footpoints": [list(point) for point in footpoints],
                "player_boxes": [list(box) for box in boxes],
                "players": [
                    {
                        "track_id": player.track_id,
                        "team_id": player.team_id,
                        "footpoint": list(player.footpoint),
                        "box": list(player.box),
                    }
                    for player in players
                ],
                "transform": transform.tolist(),
            }
            for candidates, footpoints, boxes, players, transform in zip(
                frame_candidates,
                frame_player_footpoints,
                frame_player_boxes,
                frame_player_contexts,
                frame_transforms,
                strict=True,
            )
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_candidate_cache(
    path: Path,
) -> tuple[
    str,
    float,
    list[tuple[BallCandidate, ...]],
    list[tuple[tuple[float, float], ...]],
    list[tuple[tuple[float, float, float, float], ...]],
    list[tuple[PlayerContext, ...]],
    list[np.ndarray],
    int,
    int,
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in (1, 2):
        raise ValueError(f"Onbekende kandidaatcache-versie: {payload.get('schema_version')}")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Kandidaatcache bevat geen frames")
    frame_candidates = [
        tuple(
            BallCandidate(
                box=tuple(float(value) for value in candidate["box"]),
                confidence=float(candidate["confidence"]),
            )
            for candidate in frame.get("candidates", [])
        )
        for frame in frames
    ]
    frame_player_footpoints = [
        tuple(
            tuple(float(value) for value in point)
            for point in frame.get("player_footpoints", [])
        )
        for frame in frames
    ]
    frame_player_boxes = [
        tuple(
            tuple(float(value) for value in box)
            for box in frame.get("player_boxes", [])
        )
        for frame in frames
    ]
    frame_player_contexts = [
        tuple(
            PlayerContext(
                track_id=(
                    None if player.get("track_id") is None else int(player["track_id"])
                ),
                team_id=(
                    None if player.get("team_id") is None else int(player["team_id"])
                ),
                footpoint=tuple(float(value) for value in player["footpoint"]),
                box=tuple(float(value) for value in player["box"]),
            )
            for player in frame.get("players", [])
        )
        for frame in frames
    ]
    frame_transforms = [
        np.asarray(frame["transform"], dtype=np.float64)
        for frame in frames
    ]
    return (
        str(payload.get("source_video", "")),
        float(payload["fps"]),
        frame_candidates,
        frame_player_footpoints,
        frame_player_boxes,
        frame_player_contexts,
        frame_transforms,
        int(payload.get("accepted_camera_updates", 0)),
        int(payload.get("rejected_camera_updates", 0)),
    )


def _build_tracker(
    fps: float,
    threshold: float,
) -> BallTracker:
    return BallTracker(
        maximum_gap_frames=5,
        maximum_jump_pixels=70.0,
        confidence_weight=0.25,
        acquisition_confidence=0.50,
        # A real ball in the reference opening reappears at 41% after a short
        # gap, so a nearby continuation may anchor below the independent 50%
        # acquisition threshold.
        strong_reacquisition_confidence=0.30,
        supporting_confidence=0.15,
        weak_reacquisition_confidence=threshold,
        # A distant restart remains stricter because low-confidence heads,
        # shoes, and cones also persist near players.
        remote_weak_reacquisition_confidence=max(threshold, 0.30),
        remote_weak_reacquisition_after_frames=max(1, int(round(fps * 0.5))),
        weak_reacquisition_minimum_players=2,
        weak_reacquisition_minimum_size=8.0,
        weak_support_radius_pixels=35.0,
        maximum_trajectory_support_frames=max(1, int(round(fps))),
        player_activity_radius_pixels=90.0,
        minimum_activity_players=2,
        remote_weak_player_contact_lock_frames=max(1, int(round(fps * 3.0))),
        player_proximity_weight=0.25,
        maximum_player_activity_support_frames=max(1, int(round(fps * 0.75))),
        contact_speed_multiplier=1.25,
        unrestricted_reacquisition_after_frames=max(300, int(round(fps * 10.0))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Eerste QA-versie voor baldetectie en korte baltracking.")
    parser.add_argument("--video", required=True, help="Bestandsnaam in videos/ of volledig pad.")
    parser.add_argument("--seconds", type=float, default=30.0, help="Aantal testseconden.")
    parser.add_argument("--threshold", type=float, default=0.05, help="Minimale ruwe balconfidence.")
    parser.add_argument(
        "--reuse-candidates",
        action="store_true",
        help="Sla detectorinferentie over en laad de eerder opgeslagen kandidaatcache.",
    )
    parser.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Schakel hoofdvervaging alleen uit voor intern diagnosewerk.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / "videos" / video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Video niet gevonden: {video_path}")

    output_dir = PROJECT_ROOT / "output" / "ball"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_ball_qa.mp4"
    raw_path = output_dir / f"{video_path.stem}_ball_qa_raw.mp4"
    report_path = output_dir / f"{video_path.stem}_ball_tracking.json"
    candidate_cache_path = output_dir / f"{video_path.stem}_ball_candidates.json"

    if args.reuse_candidates:
        if not candidate_cache_path.exists():
            raise FileNotFoundError(f"Kandidaatcache niet gevonden: {candidate_cache_path}")
        (
            cached_source_video,
            fps,
            frame_candidates,
            frame_player_footpoints,
            frame_player_boxes,
            frame_player_contexts,
            frame_transforms,
            accepted_camera_updates,
            rejected_camera_updates,
        ) = _load_candidate_cache(candidate_cache_path)
        if Path(cached_source_video).resolve() != video_path.resolve():
            raise ValueError("Kandidaatcache hoort bij een andere bronvideo")
        frame_number = len(frame_candidates)
        print(f"FASE 1/2 - {frame_number} frames uit kandidaatcache geladen")
    else:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Video kon niet worden geopend: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        maximum_frames = int(max(0.0, args.seconds) * fps)
        detector = FootballDetector(player_threshold=0.20, ball_threshold=args.threshold)
        frame_candidates = []
        frame_player_footpoints = []
        frame_player_boxes = []
        frame_player_contexts = []
        frame_transforms = []
        camera_motion = OnlineCameraMotion()
        player_tracker = FootballTracker(frame_rate=fps)
        team_classifier = TeamClassifier()
        local_search_center: tuple[float, float] | None = None
        frame_number = 0

        print("FASE 1/2 - Video analyseren en balkandidaten verzamelen")
        try:
            while frame_number < maximum_frames:
                success, frame = capture.read()
                if not success:
                    break
                current_to_reference = camera_motion.update(frame)
                frame_transforms.append(current_to_reference.copy())
                _, people, ball_detections = detector.detect(frame)
                tracked_people = player_tracker.update(people)
                teams = team_classifier.update(frame, tracked_people)
                candidates = candidates_from_detections(ball_detections)
                local_candidates: list[BallCandidate] = []
                if local_search_center is not None:
                    local_candidates = _local_ball_candidates(
                        detector,
                        frame,
                        local_search_center,
                    )
                    local_anchor = best_local_search_anchor(
                        local_candidates,
                        local_search_center,
                    )
                    if local_anchor is not None:
                        local_search_center = local_anchor.center
                candidates.extend(local_candidates)
                candidates = exclude_candidates_inside_people(candidates, people.xyxy)
                if local_search_center is None:
                    initial_anchors = [
                        candidate
                        for candidate in candidates
                        if candidate.confidence >= 0.50
                    ]
                    if initial_anchors:
                        local_search_center = max(
                            initial_anchors,
                            key=lambda candidate: candidate.confidence,
                        ).center
                frame_candidates.append(
                    tuple(
                        _candidate_to_reference(candidate, current_to_reference)
                        for candidate in candidates
                    )
                )
                frame_player_footpoints.append(
                    tuple(
                        transform_point(
                            ((float(x1) + float(x2)) / 2.0, float(y2)),
                            current_to_reference,
                        )
                        for x1, y1, x2, y2 in people.xyxy
                    )
                )
                frame_player_boxes.append(
                    tuple(
                        transform_box(
                            tuple(float(value) for value in box),
                            current_to_reference,
                        )
                        for box in people.xyxy
                    )
                )
                frame_player_contexts.append(
                    tuple(
                        PlayerContext(
                            track_id=int(track_id),
                            team_id=teams.get(int(track_id)),
                            footpoint=transform_point(
                                ((float(x1) + float(x2)) / 2.0, float(y2)),
                                current_to_reference,
                            ),
                            box=transform_box(
                                (float(x1), float(y1), float(x2), float(y2)),
                                current_to_reference,
                            ),
                        )
                        for (x1, y1, x2, y2), track_id in zip(
                            tracked_people.xyxy,
                            tracked_people.tracker_id
                            if tracked_people.tracker_id is not None
                            else (),
                            strict=True,
                        )
                    )
                )
                frame_number += 1
                if frame_number % 30 == 0:
                    print(f"Analyse {frame_number}/{maximum_frames} frames")
        finally:
            capture.release()
        accepted_camera_updates = camera_motion.accepted_updates
        rejected_camera_updates = camera_motion.rejected_updates
        _save_candidate_cache(
            candidate_cache_path,
            source_video=video_path,
            fps=fps,
            frame_candidates=frame_candidates,
            frame_player_footpoints=frame_player_footpoints,
            frame_player_boxes=frame_player_boxes,
            frame_player_contexts=frame_player_contexts,
            frame_transforms=frame_transforms,
            accepted_camera_updates=accepted_camera_updates,
            rejected_camera_updates=rejected_camera_updates,
        )
        print(f"Kandidaatcache: {candidate_cache_path}")

    if frame_number == 0:
        raise RuntimeError("Geen videoframes verwerkt.")

    tracker = _build_tracker(
        fps,
        args.threshold,
    )
    observations_by_frame = {}
    for analyzed_frame, candidates in enumerate(frame_candidates):
        observation = tracker.update(
            analyzed_frame,
            candidates,
            player_footpoints=frame_player_footpoints[analyzed_frame],
            player_boxes=frame_player_boxes[analyzed_frame],
            player_contexts=frame_player_contexts[analyzed_frame],
        )
        if observation is not None:
            observations_by_frame[analyzed_frame] = observation

    final_observations = interpolate_detected_gaps(
        tracker.observations,
        # Een lange boogvlucht kan op bijna dezelfde beeldpositie beginnen en
        # eindigen. Lineaire interpolatie zou de bal dan ten onrechte stil of
        # laag door het beeld laten lopen. Beperk invulling tot korte hiaten.
        maximum_gap_frames=max(1, int(round(fps * 0.5))),
        maximum_speed_pixels_per_frame=45.0,
    )
    final_observations = hold_stationary_detected_gaps(
        final_observations,
        maximum_gap_frames=max(1, int(round(fps * 8.0))),
        maximum_displacement_pixels=35.0,
        minimum_endpoint_confidence=0.50,
    )
    final_observations = _observations_to_image_space(
        final_observations,
        frame_transforms,
    )
    observations_by_frame = {
        observation.frame_number: observation
        for observation in final_observations
    }

    detected_frames = sum(
        observation.source == "detected"
        for observation in observations_by_frame.values()
    )
    predicted_frames = sum(
        observation.source == "predicted"
        for observation in observations_by_frame.values()
    )
    interpolated_frames = sum(
        observation.source == "interpolated"
        for observation in observations_by_frame.values()
    )
    stationary_frames = sum(
        observation.source == "stationary_hold"
        for observation in observations_by_frame.values()
    )

    print("FASE 2/2 - Gekozen baltraject in QA-video renderen")
    render_capture = cv2.VideoCapture(str(video_path))
    if not render_capture.isOpened():
        raise RuntimeError(f"Video kon niet opnieuw worden geopend: {video_path}")
    writer = None
    try:
        for render_frame in range(frame_number):
            success, frame = render_capture.read()
            if not success:
                break
            if not args.no_anonymize:
                frame = anonymize_people_heads(
                    frame,
                    _boxes_to_image_space(
                        frame_player_boxes[render_frame],
                        frame_transforms[render_frame],
                    ),
                )
            annotated = draw_ball_observation(
                frame,
                observations_by_frame.get(render_frame),
            )
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(raw_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("QA-video kon niet worden aangemaakt.")
            writer.write(annotated)
            if (render_frame + 1) % 30 == 0:
                print(f"Render {render_frame + 1}/{frame_number} frames")
    finally:
        render_capture.release()
        if writer is not None:
            writer.release()

    if writer is None:
        raise RuntimeError("De renderpass bevatte geen videoframes.")
    save_ball_observations(final_observations, report_path, str(video_path), fps)
    _transcode(raw_path, output_path)
    coverage = (
        detected_frames + predicted_frames + interpolated_frames + stationary_frames
    ) / frame_number
    print(f"Bal gedetecteerd: {detected_frames}/{frame_number} frames")
    print(f"Kort voorspeld: {predicted_frames}/{frame_number} frames")
    print(f"Achteraf geïnterpoleerd: {interpolated_frames}/{frame_number} frames")
    print(f"Stilstaand vastgehouden: {stationary_frames}/{frame_number} frames")
    print(
        "Camerastabilisatie: "
        f"{accepted_camera_updates} gekoppeld | "
        f"{rejected_camera_updates} overgeslagen"
    )
    print(f"Totale zichtbaarheid: {coverage:.1%}")
    print(f"QA-video: {output_path}")
    print(f"Balrapport: {report_path}")


def _transcode(raw_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
        raw_path.unlink()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError("QA-video kon niet naar H.264 worden omgezet.") from error


if __name__ == "__main__":
    main()
