from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from football_ai.classification.team_classifier import TeamClassifier
from football_ai.debug.homography_debugger import HomographyDebugger
from football_ai.detector import FootballDetector
from football_ai.filtering.player_filter import PlayerFilter
from football_ai.pitch.field_projector import FieldProjector
from football_ai.pitch.calibration_model import PitchCalibration
from football_ai.tracker import FootballTracker
from football_ai.tracking.track_engine import TrackEngine
from football_ai.visualizer import draw_tracked_players


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

    def process(
        self,
        video_path: Path,
        output_path: Path,
        max_seconds: float | None = None,
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

                annotated_frame = draw_tracked_players(
                    frame=frame,
                    tracked_players=tracked_players,
                    team_by_tracker_id=team_by_tracker_id,
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

        if writer is None:
            raise RuntimeError(
                "Er zijn geen videoframes verwerkt; "
                "outputvideo is niet aangemaakt."
            )

        track_engine.finalize()
        track_engine.print_summary()

        return frame_number

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