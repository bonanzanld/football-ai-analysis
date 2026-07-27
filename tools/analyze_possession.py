from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.analysis.entity_timeline import load_entity_timeline
from football_ai.analysis.possession import (
    PossessionState,
    PossessionTracker,
    save_possession_report,
    should_render_inferred_ball,
)
from football_ai.detection.ball_tracking import BallObservation


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyseer voorzichtig balbezit en mogelijke passes.")
    parser.add_argument("--video", required=True, help="Bestandsnaam in videos/ of volledig pad.")
    args = parser.parse_args()
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / "videos" / video_path
    prefix = video_path.stem
    entity_dir = PROJECT_ROOT / "output" / "entities"
    ball_dir = PROJECT_ROOT / "output" / "ball"
    combined_dir = PROJECT_ROOT / "output" / "combined"
    timeline_path = entity_dir / f"{prefix}_entity_timeline.json"
    ball_path = ball_dir / f"{prefix}_ball_tracking.json"
    base_video = combined_dir / f"{prefix}_all_detections_qa.mp4"
    if not timeline_path.exists():
        raise FileNotFoundError(
            f"Entiteitentijdlijn ontbreekt: {timeline_path}. Draai eerst tools/analyze_entities.py opnieuw."
        )
    if not ball_path.exists():
        raise FileNotFoundError(f"Balrapport ontbreekt: {ball_path}. Draai eerst tools/analyze_ball.py.")
    if not base_video.exists():
        raise FileNotFoundError(f"Gecombineerde QA-video ontbreekt: {base_video}.")

    timeline = load_entity_timeline(timeline_path)
    entities_by_frame = {}
    for item in timeline.observations:
        entities_by_frame.setdefault(item.frame_number, []).append(item)
    ball_payload = json.loads(ball_path.read_text(encoding="utf-8"))
    balls = {
        int(item["frame_number"]): BallObservation(
            int(item["frame_number"]), tuple(item["center"]), tuple(item["box"]),
            float(item["confidence"]), str(item["source"]),
        )
        for item in ball_payload.get("observations", [])
    }
    tracker = PossessionTracker()
    maximum_frame = max(
        max(entities_by_frame, default=-1),
        max(balls, default=-1),
    )
    observations = [
        tracker.update(frame, balls.get(frame), entities_by_frame.get(frame, []))
        for frame in range(maximum_frame + 1)
    ]

    output_dir = PROJECT_ROOT / "output" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{prefix}_possession.json"
    raw_path = output_dir / f"{prefix}_possession_qa_raw.mp4"
    output_path = output_dir / f"{prefix}_possession_qa.mp4"
    save_possession_report(report_path, timeline.source_video, timeline.fps, observations, tracker.passes)
    _render(base_video, raw_path, observations, entities_by_frame, balls)
    _transcode(raw_path, output_path)

    controlled = sum(item.state is PossessionState.CONTROLLED for item in observations)
    inferred = sum(item.state is PossessionState.INFERRED for item in observations)
    contested = sum(item.state is PossessionState.CONTESTED for item in observations)
    print(f"Bevestigd balbezit: {controlled}/{len(observations)} frames")
    print(f"Vermoedelijk behouden bezit: {inferred}/{len(observations)} frames")
    print(f"Duel/onzeker bezit: {contested}/{len(observations)} frames")
    print(f"Voorlopige passkandidaten: {len(tracker.passes)}")
    print(f"QA-video: {output_path}")
    print(f"Balbezitrapport: {report_path}")


def _render(
    base_video: Path,
    raw_path: Path,
    observations,
    entities_by_frame,
    balls,
) -> None:
    capture = cv2.VideoCapture(str(base_video))
    if not capture.isOpened():
        raise RuntimeError(f"Video kon niet worden geopend: {base_video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    frame_number = 0
    try:
        while frame_number < len(observations):
            success, frame = capture.read()
            if not success:
                break
            observation = observations[frame_number]
            color = {
                PossessionState.CONTROLLED: (0, 255, 0),
                PossessionState.INFERRED: (0, 215, 255),
            }.get(observation.state, (0, 165, 255))
            label = {
                PossessionState.CONTROLLED: f"BALBEZIT: {observation.label}",
                PossessionState.INFERRED: f"BALBEZIT VERMOEDELIJK: {observation.label}",
                PossessionState.CONTESTED: "BALBEZIT: DUEL / ONZEKER",
                PossessionState.LOOSE: "BALBEZIT: LOSSE BAL",
                PossessionState.UNKNOWN: "BALBEZIT: BAL NIET BETROUWBAAR ZICHTBAAR",
            }[observation.state]
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (15, 15, 15), -1)
            cv2.putText(frame, label, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
            if observation.track_id is not None:
                owner = next(
                    (item for item in entities_by_frame.get(frame_number, []) if item.track_id == observation.track_id),
                    None,
                )
                if owner is not None:
                    point = tuple(int(round(value)) for value in owner.footpoint)
                    draw_inferred_ball = should_render_inferred_ball(
                        observation,
                        balls.get(frame_number),
                    )
                    if (
                        observation.state is PossessionState.CONTROLLED
                        or draw_inferred_ball
                    ):
                        cv2.circle(frame, point, 13, color, 3, cv2.LINE_AA)
                    if draw_inferred_ball:
                        cv2.circle(frame, point, 9, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.circle(frame, point, 9, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(
                            frame,
                            "BAL vermoedelijk",
                            (point[0] + 13, max(22, point[1] - 9)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
            if writer is None:
                writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            frame_number += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def _transcode(raw_path: Path, output_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", str(output_path),
    ], check=True)
    raw_path.unlink()


if __name__ == "__main__":
    main()
