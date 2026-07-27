from __future__ import annotations

import argparse
from collections import deque
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
    save_possession_report(
        report_path,
        timeline.source_video,
        timeline.fps,
        observations,
        tracker.passes,
        tracker.turnovers,
    )
    _render(
        base_video,
        raw_path,
        observations,
        entities_by_frame,
        balls,
        tracker.passes,
        tracker.turnovers,
    )
    _transcode(raw_path, output_path)

    controlled = sum(item.state is PossessionState.CONTROLLED for item in observations)
    inferred = sum(item.state is PossessionState.INFERRED for item in observations)
    contested = sum(item.state is PossessionState.CONTESTED for item in observations)
    print(f"Bevestigd balbezit: {controlled}/{len(observations)} frames")
    print(f"Vermoedelijk behouden bezit: {inferred}/{len(observations)} frames")
    print(f"Duel/onzeker bezit: {contested}/{len(observations)} frames")
    print(f"Voorlopige passkandidaten: {len(tracker.passes)}")
    print(f"Bevestigd balverlies: {len(tracker.turnovers)}")
    print(f"QA-video: {output_path}")
    print(f"Balbezitrapport: {report_path}")


def _render(
    base_video: Path,
    raw_path: Path,
    observations,
    entities_by_frame,
    balls,
    passes,
    turnovers,
) -> None:
    capture = cv2.VideoCapture(str(base_video))
    if not capture.isOpened():
        raise RuntimeError(f"Video kon niet worden geopend: {base_video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    frame_number = 0
    team_names = _team_names(observations)
    team_frames = {team: 0 for team in team_names}
    pass_counts = {team: 0 for team in team_names}
    loss_counts = {team: 0 for team in team_names}
    passes_at = {item.end_frame: item for item in passes}
    turnovers_at = {item.end_frame: item for item in turnovers}
    recent_events: deque[tuple[int, str, tuple[int, int, int]]] = deque(maxlen=5)
    active_event: tuple[int, str, tuple[int, int, int]] | None = None
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
            if observation.team in team_frames and observation.state in {
                PossessionState.CONTROLLED,
                PossessionState.INFERRED,
            }:
                team_frames[observation.team] += 1
            if frame_number in passes_at:
                event = passes_at[frame_number]
                pass_counts[event.team] = pass_counts.get(event.team, 0) + 1
                text = f"PASS: {_short_name(event.from_label)} > {_short_name(event.to_label)}"
                active_event = (frame_number, text, (0, 220, 0))
                recent_events.appendleft(active_event)
            if frame_number in turnovers_at:
                event = turnovers_at[frame_number]
                loss_counts[event.from_team] = loss_counts.get(event.from_team, 0) + 1
                text = f"BALVERLIES: {_short_name(event.from_label)}"
                active_event = (frame_number, text, (0, 90, 255))
                recent_events.appendleft(active_event)
            panel_width = max(390, int(round(frame.shape[1] * 0.30)))
            canvas = cv2.copyMakeBorder(
                frame,
                0,
                0,
                0,
                panel_width,
                cv2.BORDER_CONSTANT,
                value=(20, 35, 20),
            )
            _draw_dashboard(
                canvas,
                frame.shape[1],
                panel_width,
                frame_number,
                fps,
                observation,
                team_names,
                team_frames,
                pass_counts,
                loss_counts,
                active_event,
                recent_events,
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(raw_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (canvas.shape[1], canvas.shape[0]),
                )
            writer.write(canvas)
            frame_number += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def _team_names(observations) -> dict[str, str]:
    names: dict[str, str] = {}
    for item in observations:
        if item.team is None or item.label is None or item.team in names:
            continue
        club = item.label.split(" - ", 1)[0].strip()
        names[item.team] = club or item.team
    return names


def _short_name(label: str) -> str:
    return label.split(" - ", 1)[-1][:24]


def _draw_dashboard(
    canvas,
    x0: int,
    width: int,
    frame_number: int,
    fps: float,
    observation,
    team_names,
    team_frames,
    pass_counts,
    loss_counts,
    active_event,
    recent_events,
) -> None:
    cv2.rectangle(canvas, (x0, 0), (x0 + width, canvas.shape[0]), (19, 43, 24), -1)
    cv2.line(canvas, (x0, 0), (x0, canvas.shape[0]), (70, 100, 70), 2)
    left = x0 + 22
    cv2.putText(canvas, "WEDSTRIJDANALYSE", (left, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
    seconds = frame_number / max(fps, 1e-6)
    cv2.putText(canvas, f"Tijd {int(seconds // 60):02d}:{int(seconds % 60):02d}", (left, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 205, 190), 1, cv2.LINE_AA)

    state_text = {
        PossessionState.CONTROLLED: "BEVESTIGD BEZIT",
        PossessionState.INFERRED: "VERMOEDELIJK BEZIT",
        PossessionState.CONTESTED: "DUEL / ONZEKER",
        PossessionState.LOOSE: "LOSSE BAL",
        PossessionState.UNKNOWN: "BAL ONBEKEND",
    }[observation.state]
    state_color = (0, 220, 0) if observation.state is PossessionState.CONTROLLED else (0, 210, 255)
    cv2.putText(canvas, state_text, (left, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.58, state_color, 2, cv2.LINE_AA)
    owner = _short_name(observation.label) if observation.label else "-"
    cv2.putText(canvas, owner, (left, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (235, 235, 235), 1, cv2.LINE_AA)

    total = sum(team_frames.values())
    y = 188
    team_colors = {
        "team_a": (255, 130, 20),
        "team_b": (30, 70, 255),
    }
    for team, name in team_names.items():
        percentage = 100.0 * team_frames.get(team, 0) / total if total else 0.0
        team_color = team_colors.get(team, (180, 180, 180))
        cv2.putText(canvas, name[:22], (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, team_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Bezit {percentage:5.1f}%", (left, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Passes {pass_counts.get(team, 0)}   Balverlies {loss_counts.get(team, 0)}", (left, y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (205, 215, 205), 1, cv2.LINE_AA)
        y += 92

    cv2.line(canvas, (left, y), (x0 + width - 22, y), (70, 100, 70), 1)
    y += 32
    if active_event is not None and frame_number - active_event[0] <= int(round(fps * 2.5)):
        _, event_text, event_color = active_event
        cv2.rectangle(canvas, (left - 8, y - 24), (x0 + width - 18, y + 20), (30, 55, 30), -1)
        cv2.putText(canvas, event_text[:38], (left, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, event_color, 2, cv2.LINE_AA)
        y += 62

    cv2.putText(canvas, "RECENTE GEBEURTENISSEN", (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 220, 210), 1, cv2.LINE_AA)
    y += 28
    for event_frame, event_text, event_color in recent_events:
        event_seconds = event_frame / max(fps, 1e-6)
        stamp = f"{int(event_seconds // 60):02d}:{int(event_seconds % 60):02d}"
        cv2.putText(canvas, f"{stamp}  {event_text[:31]}", (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, event_color, 1, cv2.LINE_AA)
        y += 24

    cv2.putText(canvas, "Bezit omvat bevestigd + vermoedelijk", (left, canvas.shape[0] - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (155, 175, 155), 1, cv2.LINE_AA)


def _transcode(raw_path: Path, output_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", str(output_path),
    ], check=True)
    raw_path.unlink()


if __name__ == "__main__":
    main()
