from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.analysis.entity_timeline import (
    EntityTimeline,
    TimelineEntity,
    apply_team_roster,
    load_entity_timeline,
)
from football_ai.analysis.match_timeline import MatchTimelineEngine
from football_ai.analysis.possession import (
    PossessionState,
    PossessionTracker,
    save_possession_report,
    should_render_inferred_ball,
)
from football_ai.detection.ball_tracking import BallObservation
from football_ai.visualizer import draw_ball_observation
from football_ai.privacy import anonymize_people_heads
from football_ai.tracking.entity_roster import load_team_roster
from football_ai.tracking.entity_corrections import EntityRole
from football_ai.visualization.tactical_map import (
    GoalkeeperAnchoredProjector,
    TacticalMapRenderer,
    TeamHeatmapAccumulator,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyseer voorzichtig balbezit en mogelijke passes.")
    parser.add_argument("--video", required=True, help="Bestandsnaam in videos/ of volledig pad.")
    parser.add_argument(
        "--camera-height",
        type=float,
        default=3.75,
        help="Geschatte camerahoogte in meters voor de tijdelijke diepteweergave (standaard: 3.75).",
    )
    parser.add_argument(
        "--internal-identities",
        action="store_true",
        help=(
            "Gebruik intern echte spelersnamen en onvervaagd beeld. "
            "Laat dit uit voor deelbare QA-exports."
        ),
    )
    args = parser.parse_args()
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / "videos" / video_path
    prefix = video_path.stem
    entity_dir = PROJECT_ROOT / "output" / "entities"
    ball_dir = PROJECT_ROOT / "output" / "ball"
    timeline_path = entity_dir / f"{prefix}_entity_timeline.json"
    ball_path = ball_dir / f"{prefix}_ball_tracking.json"
    # Render vanaf de originele video en gebruik de entiteitentijdlijn alleen
    # als data-overlay. Afgeleide QA-video's kunnen verouderd of gedeeltelijk
    # beschadigd zijn en mogen de bezitstijdlijn niet afkappen. Bovendien
    # tekenen we zo bewust precies een balbron per frame.
    base_video = video_path
    if not timeline_path.exists():
        raise FileNotFoundError(
            f"Entiteitentijdlijn ontbreekt: {timeline_path}. Draai eerst tools/analyze_entities.py opnieuw."
        )
    if not ball_path.exists():
        raise FileNotFoundError(f"Balrapport ontbreekt: {ball_path}. Draai eerst tools/analyze_ball.py.")

    timeline = load_entity_timeline(timeline_path)
    roster_path = entity_dir / f"{prefix}_team_roster.json"
    if roster_path.exists() and args.internal_identities:
        timeline = apply_team_roster(timeline, load_team_roster(roster_path))
        print(f"INTERNE EXPORT: spelersnamen en onvervaagd beeld: {roster_path}")
    if not args.internal_identities:
        timeline = _pseudonymize_timeline(timeline)
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
    raw_observations = [
        tracker.update(frame, balls.get(frame), entities_by_frame.get(frame, []))
        for frame in range(maximum_frame + 1)
    ]
    match_timeline = MatchTimelineEngine(fps=timeline.fps).resolve(
        raw_observations,
        tracker.passes,
        tracker.turnovers,
    )
    observations = list(match_timeline.observations)
    passes = list(match_timeline.passes)
    turnovers = list(match_timeline.turnovers)
    public_team_names = _team_names(timeline.observations, observations)

    output_dir = PROJECT_ROOT / "output" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{prefix}_possession.json"
    raw_path = output_dir / f"{prefix}_possession_qa_raw.mp4"
    output_path = output_dir / f"{prefix}_possession_qa.mp4"
    ball_only_raw_path = output_dir / f"{prefix}_ball_tracking_only_raw.mp4"
    ball_only_path = output_dir / f"{prefix}_ball_tracking_only.mp4"
    save_possession_report(
        report_path,
        timeline.source_video,
        timeline.fps,
        observations,
        passes,
        turnovers,
        timeline_metadata={
            "name": "match_timeline",
            "version": 2,
            "suppressed_team_switch_frames": (
                match_timeline.suppressed_team_switches
            ),
            "unknown_frames_excluded_from_possession": True,
        },
        timeline_events=[
            item.to_dict(timeline.fps, public_team_names)
            for item in match_timeline.events
        ],
    )
    _render(
        base_video,
        raw_path,
        output_dir,
        prefix,
        observations,
        entities_by_frame,
        balls,
        passes,
        turnovers,
        public_team_names,
        args.camera_height,
        anonymize_people=not args.internal_identities,
    )
    _transcode(raw_path, output_path)
    _render_ball_tracking_only(
        video_path,
        ball_only_raw_path,
        observations,
        entities_by_frame,
        balls,
        anonymize_people=not args.internal_identities,
    )
    _transcode(ball_only_raw_path, ball_only_path)

    controlled = sum(item.state is PossessionState.CONTROLLED for item in observations)
    inferred = sum(item.state is PossessionState.INFERRED for item in observations)
    contested = sum(item.state is PossessionState.CONTESTED for item in observations)
    print(f"Bevestigd balbezit: {controlled}/{len(observations)} frames")
    print(f"Vermoedelijk behouden bezit: {inferred}/{len(observations)} frames")
    print(f"Duel/onzeker bezit: {contested}/{len(observations)} frames")
    print(f"Bevestigde passes in wedstrijdtijdlijn: {len(passes)}")
    intercepted = sum(
        item.event_type == "intercepted_pass" for item in turnovers
    )
    print(f"Onderschepte passes: {intercepted}")
    print(f"Bevestigd balverlies: {len(turnovers)}")
    print(
        "Onderdrukte onbetrouwbare teamwissels: "
        f"{match_timeline.suppressed_team_switches} frames"
    )
    print(f"QA-video: {output_path}")
    print(f"Alleen balltracking: {ball_only_path}")
    print(f"Balbezitrapport: {report_path}")


def _render_ball_tracking_only(
    video_path: Path,
    output_path: Path,
    observations,
    entities_by_frame,
    balls,
    *,
    anonymize_people: bool = True,
) -> None:
    """Render original footage with no overlays except the selected ball."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video kon niet worden geopend: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame_number, possession in enumerate(observations):
            success, frame = capture.read()
            if not success:
                break
            if anonymize_people:
                frame = _anonymize_frame(frame, entities_by_frame.get(frame_number, []))
            raw_ball = balls.get(frame_number)
            # Dit is expliciet de balltracking-only video. Iedere positie die
            # de tracker zelf levert heeft daarom voorrang, ook wanneer het
            # een zwakke voorspelling of interpolatie is. De bezitsmagneet is
            # uitsluitend een laatste visuele fallback bij een echt leeg
            # trackerframe.
            if raw_ball is None and possession.state is PossessionState.INFERRED:
                owner = next(
                    (
                        item
                        for item in entities_by_frame.get(frame_number, [])
                        if item.track_id == possession.track_id
                    ),
                    None,
                )
                if owner is not None:
                    point = tuple(int(round(value)) for value in owner.footpoint)
                    cv2.circle(frame, point, 10, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.circle(frame, point, 10, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                frame = draw_ball_observation(frame, raw_ball)
            writer.write(frame)
    finally:
        capture.release()
        writer.release()


def _render(
    base_video: Path,
    raw_path: Path,
    output_dir: Path,
    prefix: str,
    observations,
    entities_by_frame,
    balls,
    passes,
    turnovers,
    team_names,
    camera_height_m: float,
    *,
    anonymize_people: bool = True,
) -> None:
    capture = cv2.VideoCapture(str(base_video))
    if not capture.isOpened():
        raise RuntimeError(f"Video kon niet worden geopend: {base_video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    frame_number = 0
    team_frames = {team: 0 for team in team_names}
    pass_counts = {team: 0 for team in team_names}
    failed_pass_counts = {team: 0 for team in team_names}
    loss_counts = {team: 0 for team in team_names}
    interception_counts = {team: 0 for team in team_names}
    passes_at = {item.end_frame: item for item in passes}
    turnovers_at = {item.end_frame: item for item in turnovers}
    recent_events: deque[tuple[int, str, tuple[int, int, int]]] = deque(maxlen=5)
    active_event: tuple[int, str, tuple[int, int, int]] | None = None
    projector = GoalkeeperAnchoredProjector(camera_height_m=camera_height_m)
    tactical_map = TacticalMapRenderer(projector)
    heatmaps = TeamHeatmapAccumulator(projector)
    source_frame_size: tuple[int, int] | None = None
    try:
        while frame_number < len(observations):
            success, frame = capture.read()
            if not success:
                break
            observation = observations[frame_number]
            source_frame_size = (frame.shape[1], frame.shape[0])
            frame_entities = entities_by_frame.get(frame_number, [])
            if anonymize_people:
                frame = _anonymize_frame(frame, frame_entities)
            projector.update(frame_entities, source_frame_size)
            raw_ball = balls.get(frame_number)
            draw_inferred_ball = should_render_inferred_ball(observation, raw_ball)
            if not draw_inferred_ball:
                frame = draw_ball_observation(frame, raw_ball)
            heatmaps.add(frame_entities, source_frame_size)
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
                    (item for item in frame_entities if item.track_id == observation.track_id),
                    None,
                )
                if owner is not None:
                    point = tuple(int(round(value)) for value in owner.footpoint)
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
                if event.event_type == "intercepted_pass":
                    failed_pass_counts[event.from_team] = (
                        failed_pass_counts.get(event.from_team, 0) + 1
                    )
                    interception_counts[event.to_team] = (
                        interception_counts.get(event.to_team, 0) + 1
                    )
                    text = (
                        f"PASS ONDERSCHEPT: {_short_name(event.from_label)}"
                        f" > {_short_name(event.to_label)}"
                    )
                else:
                    text = f"BALVERLIES: {_short_name(event.from_label)}"
                active_event = (frame_number, text, (0, 90, 255))
                recent_events.appendleft(active_event)
            panel_width = max(430, int(round(frame.shape[1] * 0.34)))
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
                failed_pass_counts,
                loss_counts,
                interception_counts,
                active_event,
                recent_events,
                tactical_map,
                frame_entities,
                balls.get(frame_number),
                source_frame_size,
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
    if source_frame_size is not None:
        team_a_path, team_b_path = heatmaps.save(output_dir, prefix, team_names)
        print(f"Heatmap {team_names.get('team_a', 'team_a')}: {team_a_path}")
        print(f"Heatmap {team_names.get('team_b', 'team_b')}: {team_b_path}")


def _anonymize_frame(frame, entities):
    boxes = np.asarray(
        [item.box for item in entities],
        dtype=np.float64,
    ).reshape(-1, 4)
    return anonymize_people_heads(frame, boxes)


def _pseudonymize_timeline(timeline: EntityTimeline) -> EntityTimeline:
    observations = tuple(
        TimelineEntity(
            frame_number=item.frame_number,
            track_id=item.track_id,
            identity_id=item.identity_id,
            label=_pseudonymous_label(item),
            role=item.role,
            team=item.team,
            box=item.box,
            footpoint=item.footpoint,
        )
        for item in timeline.observations
    )
    return EntityTimeline(timeline.source_video, timeline.fps, observations)


def _pseudonymous_label(item: TimelineEntity) -> str:
    role = "Keeper" if item.role is EntityRole.GOALKEEPER else "Speler"
    stable_id = item.identity_id if item.identity_id is not None else item.track_id
    return f"{item.team.value} - {role} {stable_id}"


def _team_names(*observation_groups) -> dict[str, str]:
    names: dict[str, str] = {}
    observed_teams: set[str] = set()
    for observations in observation_groups:
        for item in observations:
            raw_team = getattr(item, "team", None)
            if raw_team is None:
                continue
            team = str(getattr(raw_team, "value", raw_team))
            observed_teams.add(team)
            label = getattr(item, "label", None)
            if label is None:
                continue
            club = label.split(" - ", 1)[0].strip()
            if not club or club.lower() == "onbekend" or club.upper().startswith("ID "):
                continue
            names.setdefault(team, club)
    for team in observed_teams:
        names.setdefault(team, team)
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
    failed_pass_counts,
    loss_counts,
    interception_counts,
    active_event,
    recent_events,
    tactical_map,
    entities,
    ball,
    source_frame_size,
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
    y = 176
    team_colors = {
        "team_a": (255, 130, 20),
        "team_b": (30, 70, 255),
    }
    for team, name in team_names.items():
        percentage = 100.0 * team_frames.get(team, 0) / total if total else 0.0
        team_color = team_colors.get(team, (180, 180, 180))
        cv2.putText(canvas, name[:22], (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, team_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Bezit {percentage:5.1f}%", (left, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            (
                f"Passes {pass_counts.get(team, 0)}   Mislukt "
                f"{failed_pass_counts.get(team, 0)}"
            ),
            (left, y + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (205, 215, 205),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"Onderscheppingen {interception_counts.get(team, 0)}   "
                f"Balverlies {loss_counts.get(team, 0)}"
            ),
            (left, y + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (205, 215, 205),
            1,
            cv2.LINE_AA,
        )
        y += 99

    map_top = y + 22
    map_bottom = min(canvas.shape[0] - 142, map_top + max(150, int(round(width * 0.54))))
    tactical_map.draw(
        canvas,
        (left + 8, map_top, x0 + width - 30, map_bottom),
        entities,
        ball,
        observation,
        source_frame_size,
    )
    y = map_bottom + 28
    cv2.line(canvas, (left, y), (x0 + width - 22, y), (70, 100, 70), 1)
    y += 28
    if active_event is not None and frame_number - active_event[0] <= int(round(fps * 2.5)):
        _, event_text, event_color = active_event
        cv2.rectangle(canvas, (left - 8, y - 24), (x0 + width - 18, y + 20), (30, 55, 30), -1)
        cv2.putText(canvas, event_text[:38], (left, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, event_color, 2, cv2.LINE_AA)
        y += 62

    cv2.putText(canvas, "RECENTE GEBEURTENISSEN", (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 220, 210), 1, cv2.LINE_AA)
    y += 28
    for event_frame, event_text, event_color in list(recent_events)[:3]:
        event_seconds = event_frame / max(fps, 1e-6)
        stamp = f"{int(event_seconds // 60):02d}:{int(event_seconds % 60):02d}"
        cv2.putText(canvas, f"{stamp}  {event_text[:31]}", (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, event_color, 1, cv2.LINE_AA)
        y += 24

    cv2.putText(canvas, "Kaart/heatmaps: keeper-verankerde schatting", (left, canvas.shape[0] - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (155, 175, 155), 1, cv2.LINE_AA)


def _transcode(raw_path: Path, output_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", str(output_path),
    ], check=True)
    raw_path.unlink()


if __name__ == "__main__":
    main()
