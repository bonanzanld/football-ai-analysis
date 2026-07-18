from __future__ import annotations

from math import hypot

import supervision as sv

from football_ai.pitch.field_projector import (
    FieldPosition,
    FieldProjector,
)

from .track_state import TrackState


class TrackManager:
    """
    Houdt de historie van alle ByteTrack-tracks bij.

    Iedere track bevat onder meer:

    - eerste en laatste frame;
    - aantal waarnemingen;
    - beeldposities in pixels;
    - veldposities in meters;
    - afgelegde afstand in pixels;
    - afgelegde afstand in meters;
    - gemiddelde boxhoogte;
    - gemiddelde confidence;
    - tijd binnen en buiten het speelveld.
    """

    def __init__(
        self,
        field_projector: FieldProjector | None = None,
    ) -> None:
        self.tracks: dict[int, TrackState] = {}

        self.field_projector = field_projector

    def update(
        self,
        tracked_players: sv.Detections,
        frame_number: int,
    ) -> None:
        """
        Verwerk alle getrackte spelers van één frame.
        """

        if len(tracked_players) == 0:
            return

        tracker_ids = tracked_players.tracker_id

        if tracker_ids is None:
            return

        confidences = tracked_players.confidence

        for detection_index, xyxy in enumerate(
            tracked_players.xyxy
        ):
            tracker_id = tracker_ids[
                detection_index
            ]

            if tracker_id is None:
                continue

            confidence = None

            if (
                confidences is not None
                and detection_index < len(confidences)
            ):
                confidence = confidences[
                    detection_index
                ]

            self._update_track(
                xyxy=xyxy,
                tracker_id=int(tracker_id),
                confidence=confidence,
                frame_number=frame_number,
            )

    def _update_track(
        self,
        xyxy,
        tracker_id: int,
        confidence,
        frame_number: int,
    ) -> None:
        """
        Werk één individuele track bij.
        """

        center_x = float(
            (xyxy[0] + xyxy[2]) / 2.0
        )

        center_y = float(
            (xyxy[1] + xyxy[3]) / 2.0
        )

        pixel_position = (
            center_x,
            center_y,
        )

        box_height = float(
            xyxy[3] - xyxy[1]
        )

        field_position = self._project_field_position(
            xyxy=xyxy,
        )

        if tracker_id not in self.tracks:
            state = TrackState(
                track_id=tracker_id,
                first_frame=frame_number,
                last_frame=frame_number,
                frames_seen=1,
            )

            state.positions.append(
                pixel_position
            )

            state.box_heights.append(
                box_height
            )

            if confidence is not None:
                state.confidences.append(
                    float(confidence)
                )

            self._append_field_position(
                state=state,
                field_position=field_position,
                calculate_distance=False,
            )

            self.tracks[tracker_id] = state
            return

        state = self.tracks[tracker_id]

        self._update_pixel_distance(
            state=state,
            current_position=pixel_position,
        )

        self._append_field_position(
            state=state,
            field_position=field_position,
            calculate_distance=True,
        )

        state.positions.append(
            pixel_position
        )

        state.box_heights.append(
            box_height
        )

        if confidence is not None:
            state.confidences.append(
                float(confidence)
            )

        state.frames_seen += 1
        state.last_frame = frame_number

    def _project_field_position(
        self,
        xyxy,
    ) -> FieldPosition | None:
        """
        Projecteer het voetpunt van de bounding box naar het veld.

        Wanneer geen FieldProjector is ingesteld, wordt None
        geretourneerd.
        """

        if self.field_projector is None:
            return None

        try:
            return (
                self.field_projector.project_bounding_box(
                    bounding_box=xyxy,
                )
            )
        except (
            ValueError,
            FloatingPointError,
        ):
            return None

    def _update_pixel_distance(
        self,
        state: TrackState,
        current_position: tuple[float, float],
    ) -> None:
        """
        Tel de afstand vanaf de vorige beeldpositie op.
        """

        if not state.positions:
            return

        previous_x, previous_y = (
            state.positions[-1]
        )

        current_x, current_y = (
            current_position
        )

        distance = hypot(
            current_x - previous_x,
            current_y - previous_y,
        )

        state.total_distance_pixels += (
            distance
        )

    def _append_field_position(
        self,
        state: TrackState,
        field_position: FieldPosition | None,
        calculate_distance: bool,
    ) -> None:
        """
        Sla een veldpositie en de bijbehorende veldstatus op.

        Afstand in meters wordt alleen toegevoegd wanneer zowel de
        huidige als de vorige geldige positie binnen het speelveld ligt.
        Daardoor veroorzaken posities buiten de homography niet direct
        onrealistisch grote loopafstanden.
        """

        if field_position is None:
            state.field_positions.append(None)
            state.inside_pitch_flags.append(None)
            return

        current_position = (
            field_position.x,
            field_position.y,
        )

        if (
            calculate_distance
            and field_position.is_inside_pitch
        ):
            previous_position = (
                self._last_inside_field_position(
                    state=state,
                )
            )

            if previous_position is not None:
                previous_x, previous_y = (
                    previous_position
                )

                current_x, current_y = (
                    current_position
                )

                distance_meters = hypot(
                    current_x - previous_x,
                    current_y - previous_y,
                )

                state.total_distance_meters += (
                    distance_meters
                )

        state.field_positions.append(
            current_position
        )

        state.inside_pitch_flags.append(
            field_position.is_inside_pitch
        )

    @staticmethod
    def _last_inside_field_position(
        state: TrackState,
    ) -> tuple[float, float] | None:
        """
        Vind de laatste beschikbare veldpositie die binnen het veld lag.
        """

        for position, inside_pitch in zip(
            reversed(state.field_positions),
            reversed(state.inside_pitch_flags),
        ):
            if (
                position is not None
                and inside_pitch is True
            ):
                return position

        return None

    def get_track(
        self,
        track_id: int,
    ) -> TrackState | None:
        """
        Retourneer een track op basis van het track-ID.
        """

        return self.tracks.get(track_id)

    def get_all_tracks(
        self,
    ) -> list[TrackState]:
        """
        Retourneer alle tracks.
        """

        return list(self.tracks.values())

    @property
    def number_of_tracks(self) -> int:
        """
        Aantal unieke tracks.
        """

        return len(self.tracks)