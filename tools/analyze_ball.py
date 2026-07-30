from __future__ import annotations

import argparse
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
    candidates_from_detections,
    exclude_candidates_inside_people,
    hold_stationary_detected_gaps,
    interpolate_detected_gaps,
    save_ball_observations,
)
from football_ai.detector import FootballDetector
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
            )
        )
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Eerste QA-versie voor baldetectie en korte baltracking.")
    parser.add_argument("--video", required=True, help="Bestandsnaam in videos/ of volledig pad.")
    parser.add_argument("--seconds", type=float, default=30.0, help="Aantal testseconden.")
    parser.add_argument("--threshold", type=float, default=0.05, help="Minimale ruwe balconfidence.")
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

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video kon niet worden geopend: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    maximum_frames = int(max(0.0, args.seconds) * fps)
    detector = FootballDetector(player_threshold=0.20, ball_threshold=args.threshold)
    frame_candidates = []
    frame_player_footpoints = []
    frame_transforms: list[np.ndarray] = []
    camera_motion = OnlineCameraMotion()
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
            candidates = candidates_from_detections(ball_detections)
            candidates = exclude_candidates_inside_people(candidates, people.xyxy)
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
            frame_number += 1
            if frame_number % 30 == 0:
                print(f"Analyse {frame_number}/{maximum_frames} frames")
    finally:
        capture.release()

    if frame_number == 0:
        raise RuntimeError("Geen videoframes verwerkt.")

    tracker = BallTracker(
        maximum_gap_frames=5,
        maximum_jump_pixels=70.0,
        confidence_weight=0.25,
        acquisition_confidence=0.50,
        strong_reacquisition_confidence=0.55,
        supporting_confidence=0.15,
        weak_support_radius_pixels=35.0,
        # Een kleine bal op afstand mag ongeveer één seconde lang met zwakke,
        # maar baan-consistente detecties zichtbaar blijven. Die detecties
        # kunnen het bewezen traject niet verplaatsen of opnieuw starten.
        maximum_trajectory_support_frames=max(1, int(round(fps))),
        # Gebruik de al beschikbare spelersposities alleen als goedkope steun
        # voor een bestaande balbaan. Hiervoor draait geen extra AI-model.
        player_activity_radius_pixels=90.0,
        minimum_activity_players=2,
        maximum_player_activity_support_frames=max(1, int(round(fps * 0.75))),
        contact_speed_multiplier=1.25,
        # Tijdens een korte analyse mag een verre, nieuwe kandidaat de eenmaal
        # bewezen bal nooit stilzwijgend vervangen. Na langdurig verlies is een
        # gecontroleerde herstart nog steeds mogelijk.
        unrestricted_reacquisition_after_frames=max(300, int(round(fps * 10.0))),
    )
    observations_by_frame = {}
    for analyzed_frame, candidates in enumerate(frame_candidates):
        observation = tracker.update(
            analyzed_frame,
            candidates,
            player_footpoints=frame_player_footpoints[analyzed_frame],
        )
        if observation is not None:
            observations_by_frame[analyzed_frame] = observation

    final_observations = interpolate_detected_gaps(
        tracker.observations,
        # Achtergrondcamouflage (bijvoorbeeld een witte bal voor een gebouw of
        # reclamebord) kan langer duren dan een gewone spelersocclusie. Werk
        # daarom in seconden, zodat dezelfde grens bij iedere framerate geldt.
        maximum_gap_frames=max(1, int(round(fps * 1.5))),
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
        f"{camera_motion.accepted_updates} gekoppeld | "
        f"{camera_motion.rejected_updates} overgeslagen"
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
