from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


class TrackRole(str, Enum):
    """
    Rollen die door de debugger visueel kunnen worden onderscheiden.

    De debugger bepaalt de rol niet zelf. Een toekomstige RoleResolver,
    RefereeDetector of TeamResolver kan deze waarde aan een track
    toevoegen.
    """

    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    ASSISTANT_REFEREE = "assistant_referee"
    BALL = "ball"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DebugTrackSnapshot:
    """
    Compacte momentopname van één track voor de visualisatie.
    """

    track_id: int
    image_position: tuple[float, float] | None
    field_position: tuple[float, float] | None
    inside_pitch: bool | None
    role: TrackRole
    is_active: bool
    bounding_box: tuple[float, float, float, float] | None = None
    raw_foot_position: tuple[float, float] | None = None
    corrected_foot_position: tuple[float, float] | None = None
    foot_offset_pixels: float = 0.0


@dataclass(frozen=True)
class HomographyDebugStatistics:
    """
    Statistieken van de projecties die in één frame worden getekend.
    """

    total_tracks: int
    tracks_with_projection: int
    tracks_without_projection: int
    inside_pitch: int
    outside_pitch: int
    players: int
    goalkeepers: int
    referees: int
    assistant_referees: int
    balls: int
    unknown_roles: int

    @property
    def inside_ratio(self) -> float:
        if self.tracks_with_projection <= 0:
            return 0.0

        return self.inside_pitch / self.tracks_with_projection


class HomographyDebugger:
    """
    Rendert het originele videoframe naast een 2D-voetbalveld.

    De debugger:

    - tekent geprojecteerde tracks op het 2D-veld;
    - onderscheidt spelers, keepers en scheidsrechters;
    - ondersteunt nul, één of meerdere scheidsrechters;
    - markeert projecties binnen en buiten het veld;
    - toont track-ID's;
    - toont eenvoudige projectiestatistieken;
    - blijft onafhankelijk van detector en tracker.

    De klasse gebruikt defensieve attribuutdetectie. Daardoor kan hij
    samenwerken met TrackState-objecten zonder dat de debugger direct
    afhankelijk is van één specifieke TrackState-implementatie.
    """

    def __init__(
        self,
        field_length_meters: float = 64.0,
        field_width_meters: float = 42.0,
        panel_width: int = 640,
        panel_height: int = 720,
        outer_margin: int = 36,
        field_padding: int = 30,
        header_height: int = 105,
        footer_height: int = 45,
        show_track_ids: bool = True,
        show_statistics: bool = True,
        show_image_markers: bool = True,
        include_inactive_tracks: bool = True,
        show_footpoint_debug: bool = True,
        foot_offset_ratio: float = 0.08,
        minimum_foot_offset_pixels: float = 2.0,
        maximum_foot_offset_pixels: float = 12.0,
    ) -> None:
        if field_length_meters <= 0:
            raise ValueError(
                "field_length_meters moet groter zijn dan 0."
            )

        if field_width_meters <= 0:
            raise ValueError(
                "field_width_meters moet groter zijn dan 0."
            )

        if panel_width < 320:
            raise ValueError(
                "panel_width moet minimaal 320 pixels zijn."
            )

        if panel_height < 320:
            raise ValueError(
                "panel_height moet minimaal 320 pixels zijn."
            )

        if outer_margin < 0:
            raise ValueError(
                "outer_margin mag niet negatief zijn."
            )

        if field_padding < 0:
            raise ValueError(
                "field_padding mag niet negatief zijn."
            )

        if header_height < 0:
            raise ValueError(
                "header_height mag niet negatief zijn."
            )

        if footer_height < 0:
            raise ValueError(
                "footer_height mag niet negatief zijn."
            )

        if foot_offset_ratio < 0.0:
            raise ValueError(
                "foot_offset_ratio mag niet negatief zijn."
            )

        if minimum_foot_offset_pixels < 0.0:
            raise ValueError(
                "minimum_foot_offset_pixels mag niet negatief zijn."
            )

        if maximum_foot_offset_pixels < minimum_foot_offset_pixels:
            raise ValueError(
                "maximum_foot_offset_pixels moet minimaal gelijk zijn "
                "aan minimum_foot_offset_pixels."
            )

        self.field_length_meters = float(field_length_meters)
        self.field_width_meters = float(field_width_meters)

        self.panel_width = int(panel_width)
        self.panel_height = int(panel_height)

        self.outer_margin = int(outer_margin)
        self.field_padding = int(field_padding)
        self.header_height = int(header_height)
        self.footer_height = int(footer_height)

        self.show_track_ids = bool(show_track_ids)
        self.show_statistics = bool(show_statistics)
        self.show_image_markers = bool(show_image_markers)
        self.include_inactive_tracks = bool(include_inactive_tracks)
        self.show_footpoint_debug = bool(show_footpoint_debug)
        self.foot_offset_ratio = float(foot_offset_ratio)
        self.minimum_foot_offset_pixels = float(
            minimum_foot_offset_pixels
        )
        self.maximum_foot_offset_pixels = float(
            maximum_foot_offset_pixels
        )

        self._frame_index = 0

        self._background_color = (28, 28, 28)
        self._panel_color = (40, 40, 40)
        self._pitch_color = (55, 122, 55)
        self._pitch_line_color = (235, 235, 235)
        self._text_color = (245, 245, 245)
        self._muted_text_color = (185, 185, 185)

        self._inside_player_color = (70, 210, 70)
        self._outside_player_color = (50, 50, 235)

        self._inside_referee_color = (0, 225, 255)
        self._outside_referee_color = (0, 140, 255)

        self._inside_goalkeeper_color = (255, 145, 40)
        self._outside_goalkeeper_color = (200, 80, 20)

        self._assistant_referee_color = (220, 80, 220)
        self._ball_color = (245, 245, 245)
        self._unknown_color = (160, 160, 160)

    def reset(self) -> None:
        """
        Reset interne frameteller.
        """

        self._frame_index = 0

    def render(
        self,
        frame: np.ndarray,
        tracks: Iterable[Any],
        frame_index: int | None = None,
    ) -> np.ndarray:
        """
        Maak één gecombineerd debugframe.

        Links staat het originele videoframe.
        Rechts staat het 2D-veld met alle beschikbare projecties.

        Parameters
        ----------
        frame:
            Origineel BGR-videoframe.

        tracks:
            Iterable met TrackState-achtige objecten.

        frame_index:
            Optioneel expliciet framenummer. Bij None gebruikt de
            debugger een interne oplopende teller.
        """

        self._validate_frame(frame)

        current_frame_index = (
            self._frame_index
            if frame_index is None
            else int(frame_index)
        )

        snapshots = self._build_snapshots(tracks)

        statistics = self._calculate_statistics(
            snapshots=snapshots,
        )

        annotated_frame = frame.copy()

        if self.show_image_markers:
            self._draw_image_markers(
                frame=annotated_frame,
                snapshots=snapshots,
            )

        pitch_panel = self._create_pitch_panel(
            snapshots=snapshots,
            statistics=statistics,
            frame_index=current_frame_index,
        )

        output = self._combine_frame_and_panel(
            frame=annotated_frame,
            panel=pitch_panel,
        )

        self._frame_index = current_frame_index + 1

        return output

    def create_pitch_view(
        self,
        tracks: Iterable[Any],
        frame_index: int | None = None,
    ) -> np.ndarray:
        """
        Maak alleen de 2D-veldweergave, zonder origineel videoframe.
        """

        current_frame_index = (
            self._frame_index
            if frame_index is None
            else int(frame_index)
        )

        snapshots = self._build_snapshots(tracks)

        statistics = self._calculate_statistics(
            snapshots=snapshots,
        )

        return self._create_pitch_panel(
            snapshots=snapshots,
            statistics=statistics,
            frame_index=current_frame_index,
        )

    def _create_pitch_panel(
        self,
        snapshots: Sequence[DebugTrackSnapshot],
        statistics: HomographyDebugStatistics,
        frame_index: int,
    ) -> np.ndarray:
        """
        Maak het complete rechterpaneel.
        """

        panel = np.full(
            (
                self.panel_height,
                self.panel_width,
                3,
            ),
            self._panel_color,
            dtype=np.uint8,
        )

        self._draw_header(
            panel=panel,
            frame_index=frame_index,
        )

        field_rectangle = self._get_field_rectangle()

        self._draw_pitch(
            panel=panel,
            field_rectangle=field_rectangle,
        )

        self._draw_projected_tracks(
            panel=panel,
            snapshots=snapshots,
            field_rectangle=field_rectangle,
        )

        if self.show_statistics:
            self._draw_statistics(
                panel=panel,
                statistics=statistics,
            )

        self._draw_legend(panel=panel)

        return panel

    def _get_field_rectangle(
        self,
    ) -> tuple[int, int, int, int]:
        """
        Bepaal het grootste veld dat met behoud van verhouding past.
        """

        available_left = self.outer_margin
        available_top = self.header_height + self.field_padding

        available_right = (
            self.panel_width
            - self.outer_margin
        )

        available_bottom = (
            self.panel_height
            - self.footer_height
            - self.field_padding
        )

        available_width = max(
            1,
            available_right - available_left,
        )

        available_height = max(
            1,
            available_bottom - available_top,
        )

        field_ratio = (
            self.field_length_meters
            / self.field_width_meters
        )

        available_ratio = (
            available_width
            / available_height
        )

        if available_ratio >= field_ratio:
            field_height = available_height
            field_width = int(
                round(field_height * field_ratio)
            )
        else:
            field_width = available_width
            field_height = int(
                round(field_width / field_ratio)
            )

        left = (
            available_left
            + (available_width - field_width) // 2
        )

        top = (
            available_top
            + (available_height - field_height) // 2
        )

        right = left + field_width
        bottom = top + field_height

        return left, top, right, bottom

    def _draw_header(
        self,
        panel: np.ndarray,
        frame_index: int,
    ) -> None:
        """
        Teken titel en framenummer.
        """

        cv2.putText(
            panel,
            "Homography Debugger",
            (24, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            self._text_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            panel,
            f"Frame: {frame_index}",
            (24, 71),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self._muted_text_color,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            panel,
            (
                f"Veld: {self.field_length_meters:.1f} x "
                f"{self.field_width_meters:.1f} m"
            ),
            (190, 71),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self._muted_text_color,
            1,
            cv2.LINE_AA,
        )

    def _draw_pitch(
        self,
        panel: np.ndarray,
        field_rectangle: tuple[int, int, int, int],
    ) -> None:
        """
        Teken een schaalbaar voetbalveld.
        """

        left, top, right, bottom = field_rectangle

        cv2.rectangle(
            panel,
            (left, top),
            (right, bottom),
            self._pitch_color,
            thickness=-1,
        )

        cv2.rectangle(
            panel,
            (left, top),
            (right, bottom),
            self._pitch_line_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        center_x = int(round((left + right) / 2))
        center_y = int(round((top + bottom) / 2))

        cv2.line(
            panel,
            (center_x, top),
            (center_x, bottom),
            self._pitch_line_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        center_circle_radius_meters = min(
            7.5,
            self.field_width_meters * 0.18,
        )

        center_radius_pixels = max(
            5,
            int(
                round(
                    self._meters_to_pixels_y(
                        meters=center_circle_radius_meters,
                        field_rectangle=field_rectangle,
                    )
                )
            ),
        )

        cv2.circle(
            panel,
            (center_x, center_y),
            center_radius_pixels,
            self._pitch_line_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.circle(
            panel,
            (center_x, center_y),
            3,
            self._pitch_line_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        self._draw_penalty_areas(
            panel=panel,
            field_rectangle=field_rectangle,
        )

        self._draw_goals(
            panel=panel,
            field_rectangle=field_rectangle,
        )

    def _draw_penalty_areas(
        self,
        panel: np.ndarray,
        field_rectangle: tuple[int, int, int, int],
    ) -> None:
        """
        Teken proportionele strafschopgebieden.

        De afmetingen worden begrensd zodat deze ook op kleine
        jeugdvelden visueel bruikbaar blijven.
        """

        left, top, right, bottom = field_rectangle

        penalty_depth_meters = min(
            16.5,
            self.field_length_meters * 0.20,
        )

        penalty_width_meters = min(
            40.3,
            self.field_width_meters * 0.70,
        )

        goal_area_depth_meters = min(
            5.5,
            self.field_length_meters * 0.09,
        )

        goal_area_width_meters = min(
            18.3,
            self.field_width_meters * 0.34,
        )

        penalty_depth_pixels = int(
            round(
                self._meters_to_pixels_x(
                    meters=penalty_depth_meters,
                    field_rectangle=field_rectangle,
                )
            )
        )

        penalty_width_pixels = int(
            round(
                self._meters_to_pixels_y(
                    meters=penalty_width_meters,
                    field_rectangle=field_rectangle,
                )
            )
        )

        goal_area_depth_pixels = int(
            round(
                self._meters_to_pixels_x(
                    meters=goal_area_depth_meters,
                    field_rectangle=field_rectangle,
                )
            )
        )

        goal_area_width_pixels = int(
            round(
                self._meters_to_pixels_y(
                    meters=goal_area_width_meters,
                    field_rectangle=field_rectangle,
                )
            )
        )

        center_y = int(round((top + bottom) / 2))

        penalty_top = center_y - penalty_width_pixels // 2
        penalty_bottom = center_y + penalty_width_pixels // 2

        goal_area_top = center_y - goal_area_width_pixels // 2
        goal_area_bottom = center_y + goal_area_width_pixels // 2

        cv2.rectangle(
            panel,
            (left, penalty_top),
            (left + penalty_depth_pixels, penalty_bottom),
            self._pitch_line_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.rectangle(
            panel,
            (right - penalty_depth_pixels, penalty_top),
            (right, penalty_bottom),
            self._pitch_line_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.rectangle(
            panel,
            (left, goal_area_top),
            (left + goal_area_depth_pixels, goal_area_bottom),
            self._pitch_line_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.rectangle(
            panel,
            (right - goal_area_depth_pixels, goal_area_top),
            (right, goal_area_bottom),
            self._pitch_line_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    def _draw_goals(
        self,
        panel: np.ndarray,
        field_rectangle: tuple[int, int, int, int],
    ) -> None:
        """
        Teken eenvoudige doelcontouren buiten het veld.
        """

        left, top, right, bottom = field_rectangle

        goal_width_meters = min(
            7.32,
            self.field_width_meters * 0.18,
        )

        goal_depth_pixels = 12

        goal_width_pixels = int(
            round(
                self._meters_to_pixels_y(
                    meters=goal_width_meters,
                    field_rectangle=field_rectangle,
                )
            )
        )

        center_y = int(round((top + bottom) / 2))

        goal_top = center_y - goal_width_pixels // 2
        goal_bottom = center_y + goal_width_pixels // 2

        cv2.rectangle(
            panel,
            (left - goal_depth_pixels, goal_top),
            (left, goal_bottom),
            self._pitch_line_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.rectangle(
            panel,
            (right, goal_top),
            (right + goal_depth_pixels, goal_bottom),
            self._pitch_line_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    def _draw_projected_tracks(
        self,
        panel: np.ndarray,
        snapshots: Sequence[DebugTrackSnapshot],
        field_rectangle: tuple[int, int, int, int],
    ) -> None:
        """
        Teken alle tracks met een beschikbare veldpositie.
        """

        for snapshot in snapshots:
            if snapshot.field_position is None:
                continue

            pixel_position = self._field_to_panel_position(
                field_position=snapshot.field_position,
                field_rectangle=field_rectangle,
            )

            self._draw_projected_track(
                panel=panel,
                snapshot=snapshot,
                pixel_position=pixel_position,
                field_rectangle=field_rectangle,
            )

    def _draw_projected_track(
        self,
        panel: np.ndarray,
        snapshot: DebugTrackSnapshot,
        pixel_position: tuple[int, int],
        field_rectangle: tuple[int, int, int, int],
    ) -> None:
        """
        Teken één track met rolafhankelijke vorm en kleur.
        """

        original_x, original_y = pixel_position

        visible_x, visible_y = self._clamp_to_field_edge(
            pixel_position=pixel_position,
            field_rectangle=field_rectangle,
            margin=5,
        )

        is_outside_visual_area = (
            original_x != visible_x
            or original_y != visible_y
        )

        color = self._get_track_color(snapshot)

        marker_size = self._get_marker_size(snapshot.role)

        if snapshot.role in {
            TrackRole.REFEREE,
            TrackRole.ASSISTANT_REFEREE,
        }:
            self._draw_diamond(
                image=panel,
                center=(visible_x, visible_y),
                size=marker_size,
                color=color,
            )
        elif snapshot.role == TrackRole.GOALKEEPER:
            cv2.rectangle(
                panel,
                (
                    visible_x - marker_size,
                    visible_y - marker_size,
                ),
                (
                    visible_x + marker_size,
                    visible_y + marker_size,
                ),
                color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        elif snapshot.role == TrackRole.BALL:
            cv2.circle(
                panel,
                (visible_x, visible_y),
                max(3, marker_size - 2),
                color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            cv2.circle(
                panel,
                (visible_x, visible_y),
                max(3, marker_size - 2),
                (20, 20, 20),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        elif snapshot.role == TrackRole.UNKNOWN:
            self._draw_cross(
                image=panel,
                center=(visible_x, visible_y),
                size=marker_size,
                color=color,
            )
        else:
            cv2.circle(
                panel,
                (visible_x, visible_y),
                marker_size,
                color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        if snapshot.inside_pitch is False or is_outside_visual_area:
            self._draw_outside_indicator(
                image=panel,
                center=(visible_x, visible_y),
                color=color,
            )

        if self.show_track_ids:
            self._draw_track_label(
                image=panel,
                track_id=snapshot.track_id,
                position=(visible_x, visible_y),
                color=color,
            )

    def _draw_image_markers(
        self,
        frame: np.ndarray,
        snapshots: Sequence[DebugTrackSnapshot],
    ) -> None:
        """
        Teken trackmarkeringen en optionele voetpuntdiagnostiek.

        Bij ingeschakelde voetpuntdiagnostiek:
        - cyaan kader: bounding box;
        - rood punt: huidige onderkant van de bounding box;
        - groen punt: testpunt met schaalbare positieve y-offset;
        - blauwe punt: midden van de bounding box;
        - witte lijn: afstand tussen huidig en gecorrigeerd voetpunt.

        De groene correctie is alleen een visuele test. De opgeslagen
        veldprojectie wordt in deze debugger nog niet gewijzigd.
        """

        frame_height, frame_width = frame.shape[:2]

        for snapshot in snapshots:
            if self.show_footpoint_debug and snapshot.bounding_box is not None:
                x1, y1, x2, y2 = snapshot.bounding_box

                box_left = int(round(x1))
                box_top = int(round(y1))
                box_right = int(round(x2))
                box_bottom = int(round(y2))

                cv2.rectangle(
                    frame,
                    (box_left, box_top),
                    (box_right, box_bottom),
                    (255, 220, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

                box_center = (
                    int(round((x1 + x2) / 2.0)),
                    int(round((y1 + y2) / 2.0)),
                )

                if (
                    0 <= box_center[0] < frame_width
                    and 0 <= box_center[1] < frame_height
                ):
                    cv2.circle(
                        frame,
                        box_center,
                        4,
                        (255, 120, 0),
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )

                if snapshot.raw_foot_position is not None:
                    raw_point = (
                        int(round(snapshot.raw_foot_position[0])),
                        int(round(snapshot.raw_foot_position[1])),
                    )

                    cv2.circle(
                        frame,
                        raw_point,
                        6,
                        (0, 0, 255),
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )

                    if snapshot.corrected_foot_position is not None:
                        corrected_point = (
                            int(round(snapshot.corrected_foot_position[0])),
                            int(round(snapshot.corrected_foot_position[1])),
                        )

                        cv2.line(
                            frame,
                            raw_point,
                            corrected_point,
                            (245, 245, 245),
                            thickness=2,
                            lineType=cv2.LINE_AA,
                        )

                        cv2.circle(
                            frame,
                            corrected_point,
                            6,
                            (0, 255, 0),
                            thickness=-1,
                            lineType=cv2.LINE_AA,
                        )

                        debug_label = (
                            f"h={max(0.0, y2 - y1):.0f}px "
                            f"off=+{snapshot.foot_offset_pixels:.1f}px"
                        )

                        cv2.putText(
                            frame,
                            debug_label,
                            (
                                max(0, box_left),
                                min(frame_height - 5, box_bottom + 22),
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (245, 245, 245),
                            1,
                            cv2.LINE_AA,
                        )

            if snapshot.image_position is None:
                continue

            x = int(round(snapshot.image_position[0]))
            y = int(round(snapshot.image_position[1]))

            if not (
                0 <= x < frame_width
                and 0 <= y < frame_height
            ):
                continue

            color = self._get_track_color(snapshot)

            if snapshot.role in {
                TrackRole.REFEREE,
                TrackRole.ASSISTANT_REFEREE,
            }:
                self._draw_diamond(
                    image=frame,
                    center=(x, y),
                    size=7,
                    color=color,
                )
            elif snapshot.role == TrackRole.GOALKEEPER:
                cv2.rectangle(
                    frame,
                    (x - 6, y - 6),
                    (x + 6, y + 6),
                    color,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            else:
                cv2.circle(
                    frame,
                    (x, y),
                    6,
                    color,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

            if self.show_track_ids:
                self._draw_track_label(
                    image=frame,
                    track_id=snapshot.track_id,
                    position=(x, y),
                    color=color,
                )

    def _draw_statistics(
        self,
        panel: np.ndarray,
        statistics: HomographyDebugStatistics,
    ) -> None:
        """
        Teken compacte projectiestatistieken boven het veld.
        """

        inside_percentage = (
            statistics.inside_ratio * 100.0
        )

        text = (
            f"Projecties {statistics.tracks_with_projection} | "
            f"binnen {statistics.inside_pitch} | "
            f"buiten {statistics.outside_pitch} | "
            f"{inside_percentage:.1f}%"
        )

        cv2.putText(
            panel,
            text,
            (24, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            self._text_color,
            1,
            cv2.LINE_AA,
        )

    def _draw_legend(
        self,
        panel: np.ndarray,
    ) -> None:
        """
        Teken een compacte legenda onderaan.
        """

        y = self.panel_height - 18

        legend_items = (
            ("speler", self._inside_player_color, "circle"),
            ("scheidsrechter", self._inside_referee_color, "diamond"),
            ("keeper", self._inside_goalkeeper_color, "square"),
            ("buiten", self._outside_player_color, "circle"),
        )

        x = 24

        for label, color, shape in legend_items:
            if shape == "diamond":
                self._draw_diamond(
                    image=panel,
                    center=(x + 6, y - 4),
                    size=5,
                    color=color,
                )
            elif shape == "square":
                cv2.rectangle(
                    panel,
                    (x + 1, y - 9),
                    (x + 11, y + 1),
                    color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
            else:
                cv2.circle(
                    panel,
                    (x + 6, y - 4),
                    5,
                    color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

            cv2.putText(
                panel,
                label,
                (x + 17, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                self._text_color,
                1,
                cv2.LINE_AA,
            )

            x += 122

    def _combine_frame_and_panel(
        self,
        frame: np.ndarray,
        panel: np.ndarray,
    ) -> np.ndarray:
        """
        Schaal frame en paneel naar dezelfde hoogte en voeg ze samen.
        """

        frame_height, frame_width = frame.shape[:2]
        panel_height, panel_width = panel.shape[:2]

        target_height = max(
            frame_height,
            panel_height,
        )

        resized_frame = self._resize_to_height(
            image=frame,
            target_height=target_height,
        )

        resized_panel = self._resize_to_height(
            image=panel,
            target_height=target_height,
        )

        divider = np.full(
            (
                target_height,
                4,
                3,
            ),
            self._background_color,
            dtype=np.uint8,
        )

        return np.concatenate(
            (
                resized_frame,
                divider,
                resized_panel,
            ),
            axis=1,
        )

    @staticmethod
    def _resize_to_height(
        image: np.ndarray,
        target_height: int,
    ) -> np.ndarray:
        """
        Schaal een afbeelding proportioneel naar een bepaalde hoogte.
        """

        current_height, current_width = image.shape[:2]

        if current_height == target_height:
            return image

        scale = target_height / current_height

        target_width = max(
            1,
            int(round(current_width * scale)),
        )

        interpolation = (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        )

        return cv2.resize(
            image,
            (target_width, target_height),
            interpolation=interpolation,
        )

    def _build_snapshots(
        self,
        tracks: Iterable[Any],
    ) -> list[DebugTrackSnapshot]:
        """
        Zet TrackState-achtige objecten om in veilige snapshots.
        """

        snapshots: list[DebugTrackSnapshot] = []

        for track in tracks:
            is_active = self._extract_is_active(track)

            if (
                not self.include_inactive_tracks
                and not is_active
            ):
                continue

            bounding_box = self._extract_latest_bounding_box(
                track=track,
            )

            raw_foot_position: tuple[float, float] | None = None
            corrected_foot_position: tuple[float, float] | None = None
            foot_offset_pixels = 0.0

            if bounding_box is not None:
                x1, y1, x2, y2 = bounding_box
                raw_foot_position = (
                    (x1 + x2) / 2.0,
                    y2,
                )

                box_height = max(0.0, y2 - y1)
                foot_offset_pixels = float(
                    np.clip(
                        box_height * self.foot_offset_ratio,
                        self.minimum_foot_offset_pixels,
                        self.maximum_foot_offset_pixels,
                    )
                )

                corrected_foot_position = (
                    raw_foot_position[0],
                    raw_foot_position[1] + foot_offset_pixels,
                )

            snapshots.append(
                DebugTrackSnapshot(
                    track_id=self._extract_track_id(track),
                    image_position=(
                        self._extract_latest_image_position(track)
                    ),
                    field_position=(
                        self._extract_latest_field_position(track)
                    ),
                    inside_pitch=(
                        self._extract_latest_inside_pitch(track)
                    ),
                    role=self._extract_role(track),
                    is_active=is_active,
                    bounding_box=bounding_box,
                    raw_foot_position=raw_foot_position,
                    corrected_foot_position=corrected_foot_position,
                    foot_offset_pixels=foot_offset_pixels,
                )
            )

        return snapshots

    def _calculate_statistics(
        self,
        snapshots: Sequence[DebugTrackSnapshot],
    ) -> HomographyDebugStatistics:
        """
        Bereken statistieken voor één debugframe.
        """

        with_projection = [
            snapshot
            for snapshot in snapshots
            if snapshot.field_position is not None
        ]

        return HomographyDebugStatistics(
            total_tracks=len(snapshots),
            tracks_with_projection=len(with_projection),
            tracks_without_projection=(
                len(snapshots) - len(with_projection)
            ),
            inside_pitch=sum(
                snapshot.inside_pitch is True
                for snapshot in with_projection
            ),
            outside_pitch=sum(
                snapshot.inside_pitch is False
                for snapshot in with_projection
            ),
            players=sum(
                snapshot.role == TrackRole.PLAYER
                for snapshot in snapshots
            ),
            goalkeepers=sum(
                snapshot.role == TrackRole.GOALKEEPER
                for snapshot in snapshots
            ),
            referees=sum(
                snapshot.role == TrackRole.REFEREE
                for snapshot in snapshots
            ),
            assistant_referees=sum(
                snapshot.role
                == TrackRole.ASSISTANT_REFEREE
                for snapshot in snapshots
            ),
            balls=sum(
                snapshot.role == TrackRole.BALL
                for snapshot in snapshots
            ),
            unknown_roles=sum(
                snapshot.role == TrackRole.UNKNOWN
                for snapshot in snapshots
            ),
        )

    def _field_to_panel_position(
        self,
        field_position: tuple[float, float],
        field_rectangle: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        """
        Zet een veldpositie in meters om naar paneelpixels.

        Coördinaten volgens PitchProfile:
        - field_x loopt over de veldbreedte: 0 tot width_m;
        - field_y loopt over de veldlengte: 0 tot length_m.

        Het getekende veld ligt liggend in het paneel:
        - horizontale paneelas = veldlengte (field_y);
        - verticale paneelas = veldbreedte (field_x).
        """

        field_x, field_y = field_position
        left, top, right, bottom = field_rectangle

        field_width_pixels = right - left
        field_height_pixels = bottom - top

        normalized_horizontal = (
            field_y / self.field_length_meters
        )

        normalized_vertical = (
            field_x / self.field_width_meters
        )

        pixel_x = int(
            round(
                left
                + normalized_horizontal * field_width_pixels
            )
        )

        pixel_y = int(
            round(
                top
                + normalized_vertical * field_height_pixels
            )
        )

        return pixel_x, pixel_y

    @staticmethod
    def _clamp_to_field_edge(
        pixel_position: tuple[int, int],
        field_rectangle: tuple[int, int, int, int],
        margin: int,
    ) -> tuple[int, int]:
        """
        Houd buitenprojecties zichtbaar aan de rand van het veld.
        """

        x, y = pixel_position
        left, top, right, bottom = field_rectangle

        return (
            min(max(x, left + margin), right - margin),
            min(max(y, top + margin), bottom - margin),
        )

    def _meters_to_pixels_x(
        self,
        meters: float,
        field_rectangle: tuple[int, int, int, int],
    ) -> float:
        left, _, right, _ = field_rectangle

        return (
            meters
            / self.field_length_meters
            * (right - left)
        )

    def _meters_to_pixels_y(
        self,
        meters: float,
        field_rectangle: tuple[int, int, int, int],
    ) -> float:
        _, top, _, bottom = field_rectangle

        return (
            meters
            / self.field_width_meters
            * (bottom - top)
        )

    def _get_track_color(
        self,
        snapshot: DebugTrackSnapshot,
    ) -> tuple[int, int, int]:
        """
        Geef een BGR-kleur op basis van rol en binnen/buiten-status.
        """

        inside = snapshot.inside_pitch is True

        if snapshot.role == TrackRole.REFEREE:
            return (
                self._inside_referee_color
                if inside
                else self._outside_referee_color
            )

        if snapshot.role == TrackRole.ASSISTANT_REFEREE:
            return self._assistant_referee_color

        if snapshot.role == TrackRole.GOALKEEPER:
            return (
                self._inside_goalkeeper_color
                if inside
                else self._outside_goalkeeper_color
            )

        if snapshot.role == TrackRole.BALL:
            return self._ball_color

        if snapshot.role == TrackRole.UNKNOWN:
            return self._unknown_color

        return (
            self._inside_player_color
            if inside
            else self._outside_player_color
        )

    @staticmethod
    def _get_marker_size(
        role: TrackRole,
    ) -> int:
        if role == TrackRole.BALL:
            return 5

        if role in {
            TrackRole.REFEREE,
            TrackRole.ASSISTANT_REFEREE,
        }:
            return 8

        if role == TrackRole.GOALKEEPER:
            return 7

        return 7

    @staticmethod
    def _draw_diamond(
        image: np.ndarray,
        center: tuple[int, int],
        size: int,
        color: tuple[int, int, int],
    ) -> None:
        x, y = center

        points = np.array(
            [
                [x, y - size],
                [x + size, y],
                [x, y + size],
                [x - size, y],
            ],
            dtype=np.int32,
        )

        cv2.fillConvexPoly(
            image,
            points,
            color,
            lineType=cv2.LINE_AA,
        )

    @staticmethod
    def _draw_cross(
        image: np.ndarray,
        center: tuple[int, int],
        size: int,
        color: tuple[int, int, int],
    ) -> None:
        x, y = center

        cv2.line(
            image,
            (x - size, y - size),
            (x + size, y + size),
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.line(
            image,
            (x - size, y + size),
            (x + size, y - size),
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    @staticmethod
    def _draw_outside_indicator(
        image: np.ndarray,
        center: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        """
        Markeer een buitenprojectie met een extra ring.
        """

        cv2.circle(
            image,
            center,
            11,
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    @staticmethod
    def _draw_track_label(
        image: np.ndarray,
        track_id: int,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        """
        Teken een leesbaar tracklabel.
        """

        x, y = position
        label = str(track_id)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.43
        thickness = 1

        text_size, baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        text_width, text_height = text_size

        label_x = x + 9
        label_y = y - 8

        cv2.rectangle(
            image,
            (
                label_x - 3,
                label_y - text_height - 3,
            ),
            (
                label_x + text_width + 3,
                label_y + baseline + 2,
            ),
            (20, 20, 20),
            thickness=-1,
        )

        cv2.putText(
            image,
            label,
            (label_x, label_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _extract_track_id(
        track: Any,
    ) -> int:
        value = getattr(track, "track_id", None)

        if value is None:
            value = getattr(track, "id", -1)

        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _extract_is_active(
        track: Any,
    ) -> bool:
        for attribute_name in (
            "is_active",
            "active",
            "currently_active",
        ):
            value = getattr(
                track,
                attribute_name,
                None,
            )

            if value is not None:
                return bool(value)

        return True

    @classmethod
    def _extract_latest_image_position(
        cls,
        track: Any,
    ) -> tuple[float, float] | None:
        direct_candidates = (
            "latest_image_position",
            "current_image_position",
            "image_position",
            "latest_position",
            "current_position",
            "bottom_center",
        )

        for attribute_name in direct_candidates:
            value = getattr(
                track,
                attribute_name,
                None,
            )

            position = cls._coerce_position(value)

            if position is not None:
                return position

        sequence_candidates = (
            "image_positions",
            "positions",
            "centers",
            "bottom_centers",
        )

        for attribute_name in sequence_candidates:
            value = getattr(
                track,
                attribute_name,
                None,
            )

            position = cls._extract_latest_position_from_sequence(
                value
            )

            if position is not None:
                return position

        bounding_box = cls._extract_latest_bounding_box(
            track=track,
        )

        if bounding_box is None:
            return None

        x1, y1, x2, y2 = bounding_box

        return (
            (x1 + x2) / 2.0,
            y2,
        )

    @classmethod
    def _extract_latest_field_position(
        cls,
        track: Any,
    ) -> tuple[float, float] | None:
        direct_candidates = (
            "latest_field_position",
            "current_field_position",
            "field_position",
            "projected_position",
        )

        for attribute_name in direct_candidates:
            value = getattr(
                track,
                attribute_name,
                None,
            )

            position = cls._coerce_position(value)

            if position is not None:
                return position

        sequence_candidates = (
            "field_positions",
            "projected_positions",
            "pitch_positions",
        )

        for attribute_name in sequence_candidates:
            value = getattr(
                track,
                attribute_name,
                None,
            )

            position = cls._extract_latest_position_from_sequence(
                value
            )

            if position is not None:
                return position

        return None

    @staticmethod
    def _extract_latest_inside_pitch(
        track: Any,
    ) -> bool | None:
        direct_candidates = (
            "latest_inside_pitch",
            "current_inside_pitch",
            "inside_pitch",
            "is_inside_pitch",
        )

        for attribute_name in direct_candidates:
            value = getattr(
                track,
                attribute_name,
                None,
            )

            if isinstance(value, (bool, np.bool_)):
                return bool(value)

        sequence_candidates = (
            "inside_pitch_flags",
            "inside_field_flags",
        )

        for attribute_name in sequence_candidates:
            value = getattr(
                track,
                attribute_name,
                None,
            )

            if not isinstance(value, Sequence):
                continue

            for item in reversed(value):
                if isinstance(item, (bool, np.bool_)):
                    return bool(item)

        return None

    @classmethod
    def _extract_role(
        cls,
        track: Any,
    ) -> TrackRole:
        candidates = (
            "role",
            "track_role",
            "object_role",
            "object_type",
            "class_name",
            "label",
        )

        for attribute_name in candidates:
            value = getattr(
                track,
                attribute_name,
                None,
            )

            role = cls._normalize_role(value)

            if role is not None:
                return role

        class_id = getattr(
            track,
            "class_id",
            None,
        )

        role_from_class_id = cls._role_from_class_id(
            class_id
        )

        if role_from_class_id is not None:
            return role_from_class_id

        return TrackRole.PLAYER

    @staticmethod
    def _normalize_role(
        value: Any,
    ) -> TrackRole | None:
        if isinstance(value, TrackRole):
            return value

        if value is None:
            return None

        if isinstance(value, Enum):
            value = value.value

        normalized = str(value).strip().lower()

        role_aliases = {
            "player": TrackRole.PLAYER,
            "speler": TrackRole.PLAYER,
            "person": TrackRole.PLAYER,
            "footballer": TrackRole.PLAYER,
            "goalkeeper": TrackRole.GOALKEEPER,
            "keeper": TrackRole.GOALKEEPER,
            "doelman": TrackRole.GOALKEEPER,
            "referee": TrackRole.REFEREE,
            "scheidsrechter": TrackRole.REFEREE,
            "official": TrackRole.REFEREE,
            "assistant_referee": TrackRole.ASSISTANT_REFEREE,
            "assistant-referee": TrackRole.ASSISTANT_REFEREE,
            "linesman": TrackRole.ASSISTANT_REFEREE,
            "grensrechter": TrackRole.ASSISTANT_REFEREE,
            "ball": TrackRole.BALL,
            "bal": TrackRole.BALL,
            "unknown": TrackRole.UNKNOWN,
            "onbekend": TrackRole.UNKNOWN,
        }

        return role_aliases.get(normalized)

    @staticmethod
    def _role_from_class_id(
        class_id: Any,
    ) -> TrackRole | None:
        """
        Veilige fallback voor gangbare voetbal-datasetindelingen.

        Deze mapping is bewust beperkt. Een expliciete rol of label
        heeft altijd voorrang.
        """

        try:
            normalized_class_id = int(class_id)
        except (TypeError, ValueError):
            return None

        default_mapping = {
            0: TrackRole.PLAYER,
            1: TrackRole.GOALKEEPER,
            2: TrackRole.REFEREE,
            3: TrackRole.BALL,
        }

        return default_mapping.get(normalized_class_id)

    @classmethod
    def _extract_latest_position_from_sequence(
        cls,
        value: Any,
    ) -> tuple[float, float] | None:
        if not isinstance(value, Sequence):
            return None

        for item in reversed(value):
            position = cls._coerce_position(item)

            if position is not None:
                return position

        return None

    @staticmethod
    def _coerce_position(
        value: Any,
    ) -> tuple[float, float] | None:
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            flattened = value.reshape(-1)

            if flattened.size < 2:
                return None

            x = flattened[0]
            y = flattened[1]
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes),
        ):
            if len(value) < 2:
                return None

            x = value[0]
            y = value[1]
        else:
            x = getattr(value, "x", None)
            y = getattr(value, "y", None)

            if x is None or y is None:
                return None

        try:
            float_x = float(x)
            float_y = float(y)
        except (TypeError, ValueError):
            return None

        if not (
            np.isfinite(float_x)
            and np.isfinite(float_y)
        ):
            return None

        return float_x, float_y

    @classmethod
    def _extract_latest_bounding_box(
        cls,
        track: Any,
    ) -> tuple[float, float, float, float] | None:
        direct_candidates = (
            "latest_bbox",
            "current_bbox",
            "bbox",
            "bounding_box",
        )

        for attribute_name in direct_candidates:
            value = getattr(
                track,
                attribute_name,
                None,
            )

            bounding_box = cls._coerce_bounding_box(
                value
            )

            if bounding_box is not None:
                return bounding_box

        sequence_candidates = (
            "bboxes",
            "bounding_boxes",
        )

        for attribute_name in sequence_candidates:
            value = getattr(
                track,
                attribute_name,
                None,
            )

            if not isinstance(value, Sequence):
                continue

            for item in reversed(value):
                bounding_box = cls._coerce_bounding_box(
                    item
                )

                if bounding_box is not None:
                    return bounding_box

        return None

    @staticmethod
    def _coerce_bounding_box(
        value: Any,
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            flattened = value.reshape(-1)

            if flattened.size < 4:
                return None

            coordinates = flattened[:4]
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes),
        ):
            if len(value) < 4:
                return None

            coordinates = value[:4]
        else:
            coordinate_names = (
                ("x1", "y1", "x2", "y2"),
                ("left", "top", "right", "bottom"),
            )

            coordinates = None

            for names in coordinate_names:
                candidate = [
                    getattr(value, name, None)
                    for name in names
                ]

                if all(
                    item is not None
                    for item in candidate
                ):
                    coordinates = candidate
                    break

            if coordinates is None:
                return None

        try:
            x1, y1, x2, y2 = (
                float(coordinate)
                for coordinate in coordinates
            )
        except (TypeError, ValueError):
            return None

        if not all(
            np.isfinite(coordinate)
            for coordinate in (x1, y1, x2, y2)
        ):
            return None

        return x1, y1, x2, y2

    @staticmethod
    def _validate_frame(
        frame: np.ndarray,
    ) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "frame moet een numpy-array zijn."
            )

        if frame.ndim != 3:
            raise ValueError(
                "frame moet drie dimensies hebben."
            )

        if frame.shape[2] != 3:
            raise ValueError(
                "frame moet drie BGR-kleurkanalen hebben."
            )

        if frame.size == 0:
            raise ValueError(
                "frame mag niet leeg zijn."
            )