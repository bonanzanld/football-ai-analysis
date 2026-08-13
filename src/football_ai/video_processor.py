from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np

from football_ai.classification.team_classifier import TeamClassifier
from football_ai.classification.team_consensus import TeamConsensus
from football_ai.debug.homography_debugger import HomographyDebugger
from football_ai.detector import FootballDetector
from football_ai.filtering.player_filter import PlayerFilter
from football_ai.pitch.field_projector import FieldProjector
from football_ai.pitch.calibration_model import PitchCalibration
from football_ai.tracker import FootballTracker
from football_ai.tracking.track_engine import TrackEngine
from football_ai.tracking.entity_resolver import EntityResolver
from football_ai.tracking.entity_corrections import (
    EntityCorrectionSet,
    TeamAssignment,
)
from football_ai.tracking.entity_identity import EntityIdentitySet
from football_ai.tracking.entity_roster import TeamRoster
from football_ai.tracking.track_segmentation import (
    TeamEvidence,
    TrackSegmentation,
    TrackSegmentationSet,
    save_track_segmentations,
    segment_track_by_team_switches,
)
from football_ai.tracking.entity_review_manifest import (
    build_entity_review_manifest,
    save_entity_review_manifest,
)
from football_ai.visualizer import draw_resolved_track_boxes, draw_tracked_players
from football_ai.analysis.entity_timeline import build_entity_timeline, save_entity_timeline
from football_ai.tracking.box_interpolation import observations_with_short_gaps
from football_ai.privacy import anonymize_people_heads


class VideoProcessor:
    """
    Verwerkt een voetbalvideo met detectie, tracking, teamclassificatie
    en optionele veldprojectie-debugging.

    Wanneer homography-debugging actief is, bevat de outputvideo:

    - links: het geannoteerde originele videobeeld;
    - rechts: het 2D-veld met de actuele geprojecteerde tracks.
    """

    def __init__(
        self,
        detector: FootballDetector,
        pitch_calibration: PitchCalibration | None = None,
        debug_homography: bool = True,
        debug_panel_width: int = 640,
        debug_panel_height: int = 720,
        entity_corrections: EntityCorrectionSet | None = None,
        entity_identities: EntityIdentitySet | None = None,
        team_roster: TeamRoster | None = None,
        anonymize_people: bool = True,
    ) -> None:
        self.detector = detector
        self.pitch_calibration = pitch_calibration

        self.debug_homography = bool(debug_homography)
        self.debug_panel_width = int(debug_panel_width)
        self.debug_panel_height = int(debug_panel_height)

        self.player_filter = PlayerFilter(
            minimum_box_height=24,
            minimum_aspect_ratio=1.15,
            maximum_aspect_ratio=6.0,
            minimum_foot_y_ratio=0.15,
            minimum_green_ratio=0.18,
            pitch_calibration=None,
        )

        self.team_classifier = TeamClassifier(
            samples_per_player=30,
            minimum_players=4,
            refit_interval=30,
        )

        self.entity_resolver = EntityResolver(entity_corrections)
        self.entity_identities = entity_identities
        self.team_roster = team_roster
        self.anonymize_people = bool(anonymize_people)

    def process(
        self,
        video_path: Path,
        output_path: Path,
        max_seconds: float | None = None,
        review_manifest_path: Path | None = None,
        segmentation_path: Path | None = None,
        entity_timeline_path: Path | None = None,
        stable_team_render: bool = False,
    ) -> int:
        if not video_path.exists():
            raise FileNotFoundError(
                f"Video niet gevonden: {video_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(
                f"Video kon niet worden geopend: {video_path}"
            )

        fps = capture.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30.0

        frames_to_process = (
            int(fps * max_seconds)
            if max_seconds is not None
            else None
        )

        tracker = FootballTracker(
            frame_rate=fps,
        )

        field_projector = self._create_field_projector()

        track_engine = TrackEngine(
            field_projector=field_projector,
        )

        homography_debugger = self._create_homography_debugger(
            field_projector=field_projector,
        )

        if field_projector is None:
            print(
                "ℹ️ Geen veldkalibratie beschikbaar; "
                "veldprojectie is uitgeschakeld."
            )
        else:
            print(
                "✅ FieldProjector actief; "
                "veldposities worden opgeslagen."
            )

        if homography_debugger is None:
            print(
                "ℹ️ HomographyDebugger is uitgeschakeld."
            )
        else:
            print(
                "✅ HomographyDebugger actief; "
                "2D-veld wordt naast de video gerenderd."
            )

        writer: cv2.VideoWriter | None = None
        frame_number = 0
        team_consensus = TeamConsensus(
            minimum_votes=15,
            minimum_agreement_ratio=0.80,
        )

        try:
            while True:
                if (
                    frames_to_process is not None
                    and frame_number >= frames_to_process
                ):
                    break

                success, frame = capture.read()

                if not success:
                    break

                (
                    _all_detections,
                    player_detections,
                    _ball_detections,
                ) = self.detector.detect(frame)

                filtered_player_detections = (
                    self.player_filter.filter(
                        frame=frame,
                        detections=player_detections,
                        frame_number=frame_number,
                    )
                )

                tracked_players = tracker.update(
                    filtered_player_detections
                )

                track_engine.update(
                    tracked_players=tracked_players,
                    frame_number=frame_number,
                )

                team_by_tracker_id = (
                    self.team_classifier.update(
                        frame=frame,
                        tracked_players=tracked_players,
                    )
                )

                tracker_ids = (
                    [int(value) for value in tracked_players.tracker_id]
                    if tracked_players.tracker_id is not None
                    else []
                )
                resolved_entities = self.entity_resolver.resolve_many(
                    track_ids=tracker_ids,
                    automatic_teams=team_by_tracker_id,
                )

                team_consensus.record(
                    visible_track_ids=tracker_ids,
                    team_by_tracker_id=team_by_tracker_id,
                )

                if not stable_team_render:
                    render_frame = (
                        anonymize_people_heads(frame, player_detections.xyxy)
                        if self.anonymize_people
                        else frame
                    )
                    annotated_frame = draw_tracked_players(
                        frame=render_frame,
                        tracked_players=tracked_players,
                        team_by_tracker_id=team_by_tracker_id,
                        resolved_entities=resolved_entities,
                    )

                    output_frame = self._create_output_frame(
                        annotated_frame=annotated_frame,
                        track_engine=track_engine,
                        homography_debugger=homography_debugger,
                        frame_number=frame_number,
                    )

                    if writer is None:
                        writer = self._create_video_writer(
                            output_path=output_path,
                            fps=fps,
                            frame=output_frame,
                        )

                    writer.write(output_frame)

                frame_number += 1

                if frame_number % 30 == 0:
                    if frames_to_process is None:
                        print(
                            f"Frame {frame_number} verwerkt"
                        )
                    else:
                        print(
                            f"Frame {frame_number}/"
                            f"{frames_to_process} verwerkt"
                        )

        finally:
            capture.release()

            if writer is not None:
                writer.release()

        if frame_number == 0:
            raise RuntimeError(
                "Er zijn geen videoframes verwerkt; "
                "outputvideo is niet aangemaakt."
            )

        track_engine.finalize()
        track_engine.print_summary()

        consensus_results = team_consensus.finalize(
            [track.track_id for track in track_engine.tracks]
        )

        segmentations: dict[int, TrackSegmentation] = {}
        if stable_team_render:
            segmentations = self._render_stable_team_video(
                video_path=video_path,
                output_path=output_path,
                fps=fps,
                frames_to_render=frame_number,
                tracks=track_engine.tracks,
                consensus_results=consensus_results,
            )
            if segmentation_path is not None:
                save_track_segmentations(
                    TrackSegmentationSet(
                        source_video=str(video_path),
                        fps=fps,
                        tracks=tuple(
                            item for _, item in sorted(segmentations.items())
                        ),
                    ),
                    segmentation_path,
                )
                print(f"Tracksegmenten opgeslagen: {segmentation_path}")

        self._transcode_for_playback(output_path)

        if review_manifest_path is not None:
            manifest = build_entity_review_manifest(
                source_video=str(video_path),
                fps=fps,
                tracks=track_engine.tracks,
                team_consensus=consensus_results,
                track_segmentations=segmentations,
            )
            save_entity_review_manifest(manifest, review_manifest_path)
            print(f"Entity-reviewbestand: {review_manifest_path}")

        if entity_timeline_path is not None:
            timeline = build_entity_timeline(
                source_video=str(video_path),
                fps=fps,
                tracks=track_engine.tracks,
                segmentations=segmentations,
                resolver=self.entity_resolver,
                identities=self.entity_identities,
                final_teams={
                    track_id: result.team_id
                    for track_id, result in consensus_results.items()
                    if result.team_id is not None
                },
            )
            if self.team_roster is not None:
                from football_ai.analysis.entity_timeline import apply_team_roster

                timeline = apply_team_roster(timeline, self.team_roster)
            save_entity_timeline(timeline, entity_timeline_path)
            print(f"Entiteitentijdlijn: {entity_timeline_path}")

        return frame_number

    def _render_stable_team_video(
        self,
        video_path: Path,
        output_path: Path,
        fps: float,
        frames_to_render: int,
        tracks: list[Any],
        consensus_results: dict[int, Any],
    ) -> dict[int, TrackSegmentation]:
        observations: dict[int, dict[int, tuple[float, float, float, float]]] = {}
        for track in tracks:
            for observed_frame, box in observations_with_short_gaps(track):
                observations.setdefault(observed_frame, {})[track.track_id] = box

        final_teams = {
            track_id: result.team_id
            for track_id, result in consensus_results.items()
            if result.team_id is not None
        }
        resolved_entities = self.entity_resolver.resolve_many(
            track_ids=[track.track_id for track in tracks],
            automatic_teams=final_teams,
        )
        segmentations, frame_team_evidence = self._detect_team_switches(
            video_path=video_path,
            frames_to_render=frames_to_render,
            observations=observations,
            consensus_results=consensus_results,
            resolved_entities=resolved_entities,
        )
        agreements = {
            track_id: result.agreement_ratio
            for track_id, result in consensus_results.items()
        }
        identity_labels = {}
        if self.entity_identities is not None:
            for identity in self.entity_identities.identities:
                for track_id in identity.track_ids:
                    identity_labels[track_id] = self._identity_display_label(identity)
        segment_identity_labels = self._build_segment_identity_labels(
            tracks=tracks,
            segmentations=segmentations,
        )

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Video kon niet opnieuw worden geopend: {video_path}")
        writer: cv2.VideoWriter | None = None
        temporary_path = output_path.with_name(
            f"{output_path.stem}_stable_raw{output_path.suffix}"
        )
        try:
            for current_frame in range(frames_to_render):
                success, frame = capture.read()
                if not success:
                    break
                current_boxes = observations.get(current_frame, {})
                current_entities = dict(resolved_entities)
                current_identity_labels = dict(identity_labels)
                for track_id, box in current_boxes.items():
                    segmentation = segmentations.get(track_id)
                    segment = (
                        segmentation.segment_at(current_frame)
                        if segmentation is not None
                        else None
                    )
                    if segment is not None and segment.team_id in (0, 1):
                        current_entities[track_id] = self.entity_resolver.resolve(
                            track_id,
                            automatic_team_id=segment.team_id,
                            prefer_current_team=True,
                            segment_index=segment.index,
                        )
                        if len(segmentation.segments) > 1:
                            current_identity_labels[track_id] = segment_identity_labels.get(
                                (track_id, segment.index),
                                f"ID {track_id}.{segment.index}",
                            )
                            identity = (
                                self.entity_identities.identity_for_track(track_id)
                                if self.entity_identities is not None
                                else None
                            )
                            expected_team = self._identity_team_id(identity)
                            if expected_team == segment.team_id and segment.index == 1:
                                current_identity_labels[track_id] = self._identity_display_label(
                                    identity
                                )
                        continue
                    consensus = consensus_results.get(track_id)
                    if consensus is None or consensus.is_reliable:
                        continue
                    current_team, margin = frame_team_evidence.get(
                        (current_frame, track_id),
                        (None, 0.0),
                    )
                    if current_team is None or margin < 0.12:
                        continue
                    current_entities[track_id] = self.entity_resolver.resolve(
                        track_id,
                        automatic_team_id=current_team,
                        prefer_current_team=True,
                    )
                    identity = (
                        self.entity_identities.identity_for_track(track_id)
                        if self.entity_identities is not None
                        else None
                    )
                    expected_team = self._identity_team_id(identity)
                    if expected_team is not None and expected_team != current_team:
                        current_identity_labels.pop(track_id, None)
                current_boxes, current_entities, current_identity_labels = (
                    self._deduplicate_identity_boxes(
                        current_boxes,
                        current_entities,
                        current_identity_labels,
                    )
                )
                render_frame = (
                    anonymize_people_heads(
                        frame,
                        np.asarray(list(current_boxes.values()), dtype=np.float64),
                    )
                    if self.anonymize_people
                    else frame
                )
                annotated = draw_resolved_track_boxes(
                    frame=render_frame,
                    boxes_by_tracker_id=current_boxes,
                    resolved_entities=current_entities,
                    agreement_by_tracker_id=agreements,
                    label_by_tracker_id=current_identity_labels,
                )
                if writer is None:
                    writer = self._create_video_writer(temporary_path, fps, annotated)
                writer.write(annotated)
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if writer is None:
            raise RuntimeError("De stabiele tweede videopass bevat geen frames.")
        os.replace(temporary_path, output_path)
        print("✅ Tweede videopass met definitieve teamlabels gereed.")
        return segmentations

    @staticmethod
    def _deduplicate_identity_boxes(
        boxes: dict[int, tuple[float, float, float, float]],
        entities: dict[int, Any],
        labels: dict[int, str],
    ) -> tuple[
        dict[int, tuple[float, float, float, float]],
        dict[int, Any],
        dict[int, str],
    ]:
        """Keep one visible box when two technical tracks represent one player."""

        winner_by_label: dict[str, tuple[int, float]] = {}
        for track_id, box in boxes.items():
            label = labels.get(track_id)
            if label is None:
                continue
            x1, y1, x2, y2 = box
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            current = winner_by_label.get(label)
            if current is None or area > current[1]:
                winner_by_label[label] = (track_id, area)
        duplicate_ids = {
            track_id
            for track_id, label in labels.items()
            if label in winner_by_label and winner_by_label[label][0] != track_id
        }
        if not duplicate_ids:
            return boxes, entities, labels
        return (
            {key: value for key, value in boxes.items() if key not in duplicate_ids},
            {key: value for key, value in entities.items() if key not in duplicate_ids},
            {key: value for key, value in labels.items() if key not in duplicate_ids},
        )

    def _build_segment_identity_labels(
        self,
        tracks: list[Any],
        segmentations: dict[int, TrackSegmentation],
    ) -> dict[tuple[int, int], str]:
        """Reconnect a post-occlusion segment to a recently disappearing player.

        ByteTrack can briefly keep the old box alive while already assigning a
        second technical ID to the same person. A small overlap is therefore
        allowed, but only for the same team and a nearby foot position.
        """

        if self.entity_identities is None:
            return {}
        tracks_by_id = {track.track_id: track for track in tracks}
        labels: dict[tuple[int, int], str] = {}
        for track_id, segmentation in segmentations.items():
            target_track = tracks_by_id.get(track_id)
            if target_track is None:
                continue
            for segment in segmentation.segments:
                if segment.index == 1 or segment.team_id not in (0, 1):
                    continue
                target_box = self._box_nearest_frame(
                    target_track,
                    segment.first_frame,
                    prefer_after=True,
                )
                if target_box is None:
                    continue
                best: tuple[float, str] | None = None
                for identity in self.entity_identities.identities:
                    if self._identity_team_id(identity) != segment.team_id:
                        continue
                    for candidate_id in identity.track_ids:
                        if candidate_id == track_id:
                            continue
                        candidate = tracks_by_id.get(candidate_id)
                        if candidate is None:
                            continue
                        frame_gap = segment.first_frame - candidate.last_frame
                        if frame_gap < -20 or frame_gap > 45:
                            continue
                        candidate_box = self._box_nearest_frame(
                            candidate,
                            segment.first_frame,
                            prefer_after=False,
                        )
                        if candidate_box is None:
                            continue
                        score = self._handover_score(
                            target_box,
                            candidate_box,
                            frame_gap,
                        )
                        if score <= 1.35 and (best is None or score < best[0]):
                            best = (score, self._identity_display_label(identity))
                if best is not None:
                    labels[(track_id, segment.index)] = best[1]
        return labels

    def _identity_display_label(self, identity: Any) -> str:
        """Use a real name only for identities belonging to the user's team."""

        if (
            self.team_roster is None
            or identity is None
            or identity.team is not self.team_roster.own_team
        ):
            return identity.label
        player = self.team_roster.display_label(identity.identity_id, "")
        if not player:
            return identity.label
        return f"{self.team_roster.own_team_name} - {player}"

    @staticmethod
    def _box_nearest_frame(
        track: Any,
        frame_number: int,
        prefer_after: bool,
    ) -> tuple[float, float, float, float] | None:
        observations = list(zip(track.observation_frames, track.boxes, strict=True))
        if not observations:
            return None
        preferred = [
            item for item in observations
            if (item[0] >= frame_number if prefer_after else item[0] <= frame_number)
        ]
        values = preferred or observations
        return min(values, key=lambda item: abs(item[0] - frame_number))[1]

    @staticmethod
    def _handover_score(
        first_box: tuple[float, float, float, float],
        previous_box: tuple[float, float, float, float],
        frame_gap: int,
    ) -> float:
        first = np.asarray(first_box, dtype=np.float32)
        previous = np.asarray(previous_box, dtype=np.float32)
        first_foot = np.asarray(((first[0] + first[2]) / 2.0, first[3]))
        previous_foot = np.asarray(((previous[0] + previous[2]) / 2.0, previous[3]))
        body_scale = max(12.0, ((first[3] - first[1]) + (previous[3] - previous[1])) / 2.0)
        position_error = float(np.linalg.norm(first_foot - previous_foot)) / body_scale
        time_error = abs(frame_gap) / 45.0
        return position_error + 0.35 * time_error

    def _detect_team_switches(
        self,
        video_path: Path,
        frames_to_render: int,
        observations: dict[int, dict[int, tuple[float, float, float, float]]],
        consensus_results: dict[int, Any],
        resolved_entities: dict[int, Any],
    ) -> tuple[
        dict[int, TrackSegmentation],
        dict[tuple[int, int], tuple[int | None, float]],
    ]:
        """Collect per-frame shirt evidence before rendering final labels."""

        candidate_ids = {
            track_id
            for track_id, result in consensus_results.items()
            if not result.is_reliable
        }
        evidence_by_track: dict[int, list[TeamEvidence]] = {
            track_id: [] for track_id in candidate_ids
        }
        frame_evidence: dict[tuple[int, int], tuple[int | None, float]] = {}
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Video kon niet worden geopend voor ID-wisselcontrole: {video_path}")
        try:
            for frame_number in range(frames_to_render):
                success, frame = capture.read()
                if not success:
                    break
                for track_id, box in observations.get(frame_number, {}).items():
                    if track_id not in candidate_ids:
                        continue
                    team_id, confidence = self.team_classifier.classify_box(
                        frame,
                        np.asarray(box, dtype=np.float32),
                    )
                    frame_evidence[(frame_number, track_id)] = (team_id, confidence)
                    evidence_by_track[track_id].append(
                        TeamEvidence(frame_number, team_id, confidence)
                    )
        finally:
            capture.release()

        segmentations: dict[int, TrackSegmentation] = {}
        for track_id, evidence in evidence_by_track.items():
            entity = resolved_entities[track_id]
            initial_team = (
                0 if entity.team is TeamAssignment.TEAM_A
                else 1 if entity.team is TeamAssignment.TEAM_B
                else None
            )
            segmentation = segment_track_by_team_switches(
                track_id=track_id,
                evidence=evidence,
                initial_team_id=initial_team,
            )
            if len(segmentation.segments) > 1:
                segmentations[track_id] = segmentation

        if segmentations:
            summary = ", ".join(
                f"ID {track_id}: {len(item.segments)} delen"
                for track_id, item in sorted(segmentations.items())
            )
            print(f"ID-wissels gesplitst: {summary}")
        else:
            print("ID-wisselcontrole: geen duurzame teamwissels gevonden.")
        return segmentations, frame_evidence

    @staticmethod
    def _identity_team_id(identity: Any | None) -> int | None:
        if identity is None:
            return None
        if identity.team is TeamAssignment.TEAM_A:
            return 0
        if identity.team is TeamAssignment.TEAM_B:
            return 1
        return None

    @staticmethod
    def _transcode_for_playback(output_path: Path) -> None:
        """Zet OpenCV's tijdelijke MPEG-4-output om naar brede H.264-compatibiliteit."""

        transcoded_path = output_path.with_name(
            f"{output_path.stem}_h264{output_path.suffix}"
        )
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(output_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(transcoded_path),
        ]

        try:
            subprocess.run(command, check=True)
            os.replace(transcoded_path, output_path)
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            if transcoded_path.exists():
                transcoded_path.unlink()
            raise RuntimeError(
                "De QA-video is gemaakt, maar kon niet naar H.264 worden "
                "omgezet. Controleer of ffmpeg beschikbaar is."
            ) from error

        print("✅ Outputvideo omgezet naar H.264/yuv420p voor macOS-weergave.")

    def _create_output_frame(
        self,
        annotated_frame: np.ndarray,
        track_engine: TrackEngine,
        homography_debugger: HomographyDebugger | None,
        frame_number: int,
    ) -> np.ndarray:
        """
        Maak het frame dat daadwerkelijk naar de outputvideo gaat.
        """

        if homography_debugger is None:
            return annotated_frame

        debug_tracks = self._get_current_debug_tracks(
            track_engine=track_engine,
            frame_number=frame_number,
        )

        return homography_debugger.render(
            frame=annotated_frame,
            tracks=debug_tracks,
            frame_index=frame_number,
        )

    def _create_homography_debugger(
        self,
        field_projector: FieldProjector | None,
    ) -> HomographyDebugger | None:
        """
        Maak de debugger alleen wanneer debugging en projectie actief
        zijn.
        """

        if not self.debug_homography:
            return None

        if field_projector is None:
            return None

        field_length_meters = self._extract_numeric_attribute(
            source=field_projector,
            attribute_names=(
                "field_length_meters",
                "pitch_length_meters",
                "field_length",
                "pitch_length",
                "length_meters",
                "length",
            ),
            default=64.0,
        )

        field_width_meters = self._extract_numeric_attribute(
            source=field_projector,
            attribute_names=(
                "field_width_meters",
                "pitch_width_meters",
                "field_width",
                "pitch_width",
                "width_meters",
                "width",
            ),
            default=42.0,
        )

        return HomographyDebugger(
            field_length_meters=field_length_meters,
            field_width_meters=field_width_meters,
            panel_width=self.debug_panel_width,
            panel_height=self.debug_panel_height,
            show_track_ids=True,
            show_statistics=True,
            show_image_markers=True,
            include_inactive_tracks=False,
        )

    @classmethod
    def _get_current_debug_tracks(
        cls,
        track_engine: TrackEngine,
        frame_number: int,
    ) -> list[Any]:
        """
        Haal de TrackState-objecten uit TrackEngine en behoud alleen
        tracks die bij het huidige frame horen.

        De methode ondersteunt zowel dictionaries als lijsten en blijft
        bruikbaar wanneer de interne TrackManager-API later verandert.
        """

        tracks = cls._extract_tracks_from_engine(
            track_engine=track_engine,
        )

        return [
            track
            for track in tracks
            if cls._is_track_current(
                track=track,
                frame_number=frame_number,
            )
        ]

    @classmethod
    def _extract_tracks_from_engine(
        cls,
        track_engine: TrackEngine,
    ) -> list[Any]:
        """
        Zoek defensief naar de actuele TrackState-collectie.
        """

        direct_attribute_names = (
            "current_tracks",
            "active_tracks",
            "tracks",
            "track_states",
        )

        for attribute_name in direct_attribute_names:
            value = getattr(
                track_engine,
                attribute_name,
                None,
            )

            tracks = cls._coerce_track_collection(value)

            if tracks is not None:
                return tracks

        manager = getattr(
            track_engine,
            "track_manager",
            None,
        )

        if manager is not None:
            manager_attribute_names = (
                "current_tracks",
                "active_tracks",
                "tracks",
                "track_states",
            )

            for attribute_name in manager_attribute_names:
                value = getattr(
                    manager,
                    attribute_name,
                    None,
                )

                tracks = cls._coerce_track_collection(value)

                if tracks is not None:
                    return tracks

            manager_method_names = (
                "get_current_tracks",
                "get_active_tracks",
                "get_tracks",
                "get_all_tracks",
            )

            for method_name in manager_method_names:
                method = getattr(
                    manager,
                    method_name,
                    None,
                )

                if not callable(method):
                    continue

                try:
                    value = method()
                except TypeError:
                    continue

                tracks = cls._coerce_track_collection(value)

                if tracks is not None:
                    return tracks

        engine_method_names = (
            "get_current_tracks",
            "get_active_tracks",
            "get_tracks",
            "get_all_tracks",
        )

        for method_name in engine_method_names:
            method = getattr(
                track_engine,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:
                value = method()
            except TypeError:
                continue

            tracks = cls._coerce_track_collection(value)

            if tracks is not None:
                return tracks

        return []

    @staticmethod
    def _coerce_track_collection(
        value: Any,
    ) -> list[Any] | None:
        """
        Zet een bekende trackcollectie om naar een lijst.
        """

        if value is None:
            return None

        if isinstance(value, dict):
            return list(value.values())

        if isinstance(value, (str, bytes)):
            return None

        if isinstance(value, Iterable):
            try:
                return list(value)
            except TypeError:
                return None

        return None

    @staticmethod
    def _is_track_current(
        track: Any,
        frame_number: int,
    ) -> bool:
        """
        Controleer of de laatste waarneming van een track bij het
        huidige frame hoort.

        Wanneer een track geen framenummerattribuut heeft, wordt hij
        behouden. Zo blijft de debugger compatibel met verschillende
        TrackState-versies.
        """

        frame_attribute_names = (
            "last_seen_frame",
            "last_frame_seen",
            "latest_frame",
            "latest_frame_number",
            "last_frame",
            "end_frame",
        )

        for attribute_name in frame_attribute_names:
            value = getattr(
                track,
                attribute_name,
                None,
            )

            if value is None:
                continue

            try:
                return int(value) == frame_number
            except (TypeError, ValueError):
                continue

        active_attribute_names = (
            "is_active",
            "active",
            "currently_active",
        )

        for attribute_name in active_attribute_names:
            value = getattr(
                track,
                attribute_name,
                None,
            )

            if value is not None:
                return bool(value)

        return True

    @staticmethod
    def _create_video_writer(
        output_path: Path,
        fps: float,
        frame: np.ndarray,
    ) -> cv2.VideoWriter:
        """
        Maak de VideoWriter op basis van de werkelijke outputafmetingen.

        Dit is nodig omdat de debugger een extra 2D-paneel naast het
        oorspronkelijke frame plaatst.
        """

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Outputframe moet een BGR-afbeelding zijn."
            )

        height, width = frame.shape[:2]

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Outputvideo kon niet worden gemaakt: "
                f"{output_path}"
            )

        print(
            "✅ Outputvideo gestart: "
            f"{width}x{height} pixels bij {fps:.2f} fps."
        )

        return writer

    @staticmethod
    def _extract_numeric_attribute(
        source: Any,
        attribute_names: tuple[str, ...],
        default: float,
    ) -> float:
        """
        Lees een positieve numerieke waarde uit een object.
        """

        for attribute_name in attribute_names:
            value = getattr(
                source,
                attribute_name,
                None,
            )

            if value is None:
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            if numeric_value > 0:
                return numeric_value

        return float(default)

    def _create_field_projector(
        self,
    ) -> FieldProjector | None:
        """
        Maak een FieldProjector wanneer een geldige kalibratie
        beschikbaar is.
        """

        if self.pitch_calibration is None:
            return None

        return FieldProjector(
            calibration=self.pitch_calibration,
            pitch_margin_m=0.0,
        )
