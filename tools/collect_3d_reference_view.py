from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.reference_3d import create_field_reference_3d
from football_ai.calibration.reference_observation_app import (
    ReferenceObservationApp,
    create_projection_preview,
    load_goal_seed,
    save_observation_result,
)
from football_ai.calibration.reference_observation import CameraViewObservations
from football_ai.calibration.camera_projection_3d import CameraProjection3D


def main() -> None:
    parser = argparse.ArgumentParser(description="Schat één 3D-naar-2D camerareferentie vanuit een bekend doelvlak.")
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4", help="Bestandsnaam in videos-map.")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--goal", choices=("A", "B", "a", "b"), default="B")
    parser.add_argument(
        "--resume-missing-corners",
        action="store_true",
        help="Behoud bestaande observaties en vraag alleen ontbrekende hoeken van deze achterlijn.",
    )
    parser.add_argument(
        "--redo-existing",
        action="store_true",
        help="Klik alle bestaande punten opnieuw aan op exact hetzelfde referentieframe; oude uitvoer blijft bij mislukking behouden.",
    )
    parser.add_argument(
        "--redo-goal-only",
        action="store_true",
        help="Klik alleen de vier doelpunten opnieuw aan en behoud de bestaande hoekpunten.",
    )
    parser.add_argument(
        "--recompute-existing",
        action="store_true",
        help="Bereken de bestaande handmatige punten opnieuw zonder de interactieve UI te openen.",
    )
    args = parser.parse_args()

    video = PROJECT_ROOT / "videos" / args.video
    seeds_path = PROJECT_ROOT / "output" / "pitch_bootstrap" / f"{video.stem}_{args.format}_goal_seeds.json"
    if not video.exists():
        raise FileNotFoundError(f"Video niet gevonden: {video}")
    if not seeds_path.exists():
        raise FileNotFoundError(f"Doel-seeds niet gevonden: {seeds_path}")
    profile = create_detection_profile(args.format)
    reference = create_field_reference_3d(profile)
    seed = load_goal_seed(seeds_path, args.goal)
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    output_path = output_dir / f"{video.stem}_{args.format}_view_{args.goal.upper()}_3d.json"
    existing_view = None
    requested_landmarks = None
    replace_requested = False
    if args.resume_missing_corners:
        if not output_path.exists():
            raise FileNotFoundError(f"Bestaande 3D-observaties ontbreken: {output_path}")
        existing_data = json.loads(output_path.read_text(encoding="utf-8"))
        existing_view = CameraViewObservations.from_dict(existing_data["view"])
        goal = args.goal.lower()
        existing_ids = {item.landmark_id for item in existing_view.observations}
        requested_landmarks = tuple(
            landmark_id
            for landmark_id in (f"corner_{goal}_rear", f"corner_{goal}_front")
            if landmark_id not in existing_ids
        )
        if not requested_landmarks:
            print(f"Geen ontbrekende hoeken voor Doel {args.goal.upper()}.")
            return
    elif args.redo_existing or args.redo_goal_only or args.recompute_existing:
        if not output_path.exists():
            raise FileNotFoundError(f"Bestaande 3D-observaties ontbreken: {output_path}")
        existing_data = json.loads(output_path.read_text(encoding="utf-8"))
        existing_view = CameraViewObservations.from_dict(existing_data["view"])
        by_id = {item.landmark_id: item.image_point for item in existing_view.observations}
        goal = args.goal.lower()
        required = (
            f"goal_{goal}_rear_bottom", f"goal_{goal}_front_bottom",
            f"goal_{goal}_rear_top", f"goal_{goal}_front_top",
            f"corner_{goal}_rear", f"corner_{goal}_front",
        )
        missing = [item for item in required if item not in by_id]
        if missing:
            raise ValueError(f"Bestaande referentie mist punten: {missing}")
        old_projection = CameraProjection3D(existing_data["projection"]["matrix"])
        other_goal = "a" if goal == "b" else "b"
        rear_support = old_projection.project(reference.landmark(f"corner_{other_goal}_rear").point)
        front_support = old_projection.project(reference.landmark(f"corner_{other_goal}_front").point)
        fps_capture = cv2.VideoCapture(str(video))
        fps = float(fps_capture.get(cv2.CAP_PROP_FPS))
        fps_capture.release()
        seed = replace(
            seed,
            frame_number=existing_view.frame_number,
            time_seconds=existing_view.frame_number / fps,
            camera_state=existing_view.camera_state,
            first_ground=by_id[f"goal_{goal}_rear_bottom"],
            second_ground=by_id[f"goal_{goal}_front_bottom"],
            rear_corner=by_id[f"corner_{goal}_rear"],
            front_corner=by_id[f"corner_{goal}_front"],
            rear_sideline_support=rear_support,
            front_sideline_support=front_support,
        )
        requested_landmarks = (
            () if args.recompute_existing
            else required[:4] if args.redo_goal_only
            else required
        )
        replace_requested = not args.recompute_existing
    app = ReferenceObservationApp(
        video,
        seed,
        reference,
        existing_view=existing_view,
        requested_landmarks=requested_landmarks,
        replace_requested=replace_requested,
    )
    result = app._build_result() if args.recompute_existing else app.run()

    preview_path = output_dir / f"{video.stem}_{args.format}_view_{args.goal.upper()}_3d.jpg"
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, seed.frame_number)
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"Frame {seed.frame_number} kon niet voor QA worden gelezen.")
    preview = create_projection_preview(frame, reference, result)
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"QA-preview kon niet worden opgeslagen: {preview_path}")

    if (args.redo_existing or args.redo_goal_only or args.recompute_existing) and result.estimate is None:
        print(f"OUDE REFERENTIE BEHOUDEN: {result.failure_reason}")
        return
    save_observation_result(result, output_path)

    print(f"3D-observaties opgeslagen: {output_path}")
    print(f"QA-preview: {preview_path}")
    if result.estimate is None:
        print(f"GEEN GELDIGE PROJECTIE: {result.failure_reason}")
    else:
        print(f"Projectie geldig | RMS {result.estimate.rms_error_px:.2f}px | max {result.estimate.maximum_error_px:.2f}px")


if __name__ == "__main__":
    main()
