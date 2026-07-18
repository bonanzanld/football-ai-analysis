from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from football_ai.calibration.quality_report import (
    CalibrationQualityReport,
    ControlPointContext,
    ErrorStatistics,
    PointReprojectionError,
    assess_calibration_quality,
    calculate_quality_from_predictions,
)
from football_ai.calibration.geometry_validation import (
    validate_projected_pitch_geometry,
)
from football_ai.calibration.line_calibration import (
    FieldLineDefinition,
    LinePointObservation,
    create_boundary_line_definitions,
    estimate_homography_with_line_constraints,
    filter_line_observations,
    fit_image_line_robustly,
)
from football_ai.pitch.panorama_builder import (
    FrameRegistrationDiagnostics,
    PanoramaBuilder,
)

from football_ai.pitch.calibration_model import (
    CalibrationKeyframe,
    PitchCalibration,
    PitchProfile,
)


@dataclass(frozen=True)
class SelectedFrame:
    frame: np.ndarray
    frame_number: int
    time_seconds: float


@dataclass(frozen=True)
class LandmarkDefinition:
    key: int
    name: str
    pitch_point: tuple[float, float]


@dataclass(frozen=True)
class LandmarkObservation:
    frame_index: int
    landmark_key: int
    image_point: tuple[float, float]


@dataclass(frozen=True)
class FrameLineObservation:
    frame_index: int
    line_key: int
    image_point: tuple[float, float]


def _constraint_errors_for_keyframe(
    points: list[LandmarkObservation],
    line_points: list[FrameLineObservation],
    field_lines: dict[int, FieldLineDefinition],
) -> list[str]:
    errors: list[str] = []
    complete_line_keys = {
        line_key
        for line_key in field_lines
        if sum(item.line_key == line_key for item in line_points) >= 3
    }
    x_constant_lines = {
        line_key
        for line_key in complete_line_keys
        if abs(field_lines[line_key].equation[0])
        > abs(field_lines[line_key].equation[1])
    }
    y_constant_lines = complete_line_keys - x_constant_lines
    has_full_line_grid = (
        len(x_constant_lines) >= 2 and len(y_constant_lines) >= 2
    )
    if len(points) < 2 and not has_full_line_grid:
        errors.append(
            "Gebruik minimaal 2 exacte hoeken/doelpalen in dit frame. "
            "Zonder exacte punten zijn 4 complete veldlijnen nodig: "
            "2 lange lijnen en 2 dwarslijnen."
        )
    for line_key, field_line in field_lines.items():
        count = sum(item.line_key == line_key for item in line_points)
        if 0 < count < 3:
            errors.append(
                f"{field_line.name} heeft {count} lijnpunt(en); "
                "minimaal 3 vereist."
            )
    scalar_constraints = 2 * len(points) + len(line_points)
    if scalar_constraints < 10:
        remaining = 10 - scalar_constraints
        errors.append(
            f"heeft {len(points)} exact(e) punt(en) en "
            f"{len(line_points)} lijnpunt(en): {scalar_constraints}/10 "
            f"meetwaarden. Voeg nog minimaal {remaining} lijnpunt(en) toe, "
            "of gebruik exacte punten (ieder exact punt telt dubbel)."
        )
    return errors


class OpenCvCalibrationApp:
    WINDOW_NAME = "Football AI - veldkalibratie"
    SIDEBAR_WIDTH = 440
    WINDOW_WIDTH = 1640
    WINDOW_HEIGHT = 900

    def __init__(
        self,
        video_path: Path,
        landmarks: dict[int, LandmarkDefinition],
        field_lines: dict[int, FieldLineDefinition],
    ) -> None:
        self.video_path = video_path
        self.landmarks = landmarks
        self.field_lines = field_lines

        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Video kon niet worden geopend: {video_path}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:
            self.fps = 30.0

        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            raise RuntimeError("De video bevat geen leesbare frames.")

        self.mode = "select"
        self.current_frame_number = 0
        self.current_frame: np.ndarray | None = None

        self.selected_frames: list[SelectedFrame] = []
        self.observations: list[LandmarkObservation] = []
        self.line_observations: list[FrameLineObservation] = []

        self.annotation_frame_index = 0
        self.current_landmark_key: int | None = None
        self.annotation_kind = "point"
        self.current_line_key = 3
        self.guided_steps: list[tuple[str, int]] = []
        self.guided_step_index = 0

        self.display_image_rect = (0, 0, 0, 0)
        self.display_scale = 1.0
        self.display_view_origin = (0.0, 0.0)
        self.zoom_factor = 1.0
        self.zoom_center: tuple[float, float] | None = None
        self.cancelled = False
        self.finished = False
        self.status_message = (
            "Kies minimaal 3 overlappende frames: linker doel, tussenbeeld, "
            "rechter doel."
        )

    def run(
        self,
    ) -> tuple[
        list[SelectedFrame],
        list[LandmarkObservation],
        list[FrameLineObservation],
    ]:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)

        self._load_frame(0)

        while True:
            canvas = self._render()
            cv2.imshow(self.WINDOW_NAME, canvas)
            key = cv2.waitKeyEx(30)

            if key == -1:
                continue

            if key == 27:
                self.cancelled = True
                break

            if self.mode == "select":
                self._handle_select_key(key)
            else:
                self._handle_annotate_key(key)

            if self.finished:
                break

        cv2.destroyWindow(self.WINDOW_NAME)
        self.capture.release()

        if self.cancelled:
            raise RuntimeError("Kalibratie afgebroken.")

        return (
            self.selected_frames,
            self.observations,
            self.line_observations,
        )

    def _handle_select_key(self, key: int) -> None:
        if key in (ord("a"), ord("A")):
            self._load_frame(self.current_frame_number - int(self.fps))
        elif key in (ord("d"), ord("D")):
            self._load_frame(self.current_frame_number + int(self.fps))
        elif key in (ord("s"), ord("S")):
            self._load_frame(self.current_frame_number - int(self.fps * 5))
        elif key in (ord("w"), ord("W")):
            self._load_frame(self.current_frame_number + int(self.fps * 5))
        elif key in (2424832, 65361):
            self._load_frame(self.current_frame_number - 1)
        elif key in (2555904, 65363):
            self._load_frame(self.current_frame_number + 1)
        elif key == 32:
            self._add_current_frame()
        elif key in (8, 127):
            self._remove_last_frame()
        elif key in (10, 13):
            if len(self.selected_frames) < 3:
                self.status_message = (
                    "Selecteer minimaal 3 frames: linker doel, een overlappend "
                    "tussenbeeld en rechter doel."
                )
                return
            self.mode = "annotate"
            self.annotation_frame_index = 0
            self.current_landmark_key = None
            self.current_frame = self.selected_frames[0].frame.copy()
            self.status_message = (
                "Kies A voor het LINKER doel of B voor het RECHTER doel."
            )

    def _handle_annotate_key(self, key: int) -> None:
        if key in (ord("+"), ord("=")):
            self._change_zoom(1.25)
            return
        if key in (ord("-"), ord("_")):
            self._change_zoom(0.8)
            return
        if key == ord("0"):
            self._reset_zoom()
            return
        if key in (ord("a"), ord("A")):
            self._start_goal_first_workflow("A")
            return
        if key in (ord("b"), ord("B")):
            self._start_goal_first_workflow("B")
            return
        if key in (ord("m"), ord("M")):
            self.status_message = (
                "Doelpaalmodus gebruikt alleen vier grondpunten. "
                "Lijninvoer is niet nodig."
            )
            return

        if self.annotation_kind == "point" and ord("1") <= key <= ord("8"):
            self.status_message = (
                "Doelpaalmodus: gebruik A voor het LINKER doel of B voor "
                "het RECHTER doel; vrije hoekpunten zijn uitgeschakeld."
            )
            return

        if self.annotation_kind == "line" and ord("1") <= key <= ord("5"):
            self.current_line_key = int(chr(key))
            return

        if key in (ord("n"), ord("N")):
            self._next_annotation_frame()
        elif key in (ord("p"), ord("P")):
            self._previous_annotation_frame()
        elif key in (ord("u"), ord("U"), 8, 127):
            self._undo_last_observation()
        elif key in (ord("r"), ord("R")):
            self._clear_current_frame()
        elif key in (ord("k"), ord("K")):
            self._select_next_landmark()
        elif key in (10, 13):
            valid_frames, errors = self._validate_keyframe_observations()
            if len(valid_frames) < 2:
                detail = errors[0] if errors else "onvoldoende constraints"
                self.status_message = (
                    "Kalibratie nog niet compleet. " + detail
                )
                return
            self.finished = True

    def _mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        _userdata: object,
    ) -> None:
        if event == cv2.EVENT_MOUSEWHEEL and self.mode == "annotate":
            anchor = self._canvas_to_original(x, y)
            if anchor is None:
                return
            raw_delta = (flags >> 16) & 0xFFFF
            delta = raw_delta - 0x10000 if raw_delta >= 0x8000 else raw_delta
            self._change_zoom(1.25 if delta > 0 else 0.8, anchor)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.mode == "select":
            return

        if self.current_frame is None:
            return

        original = self._canvas_to_original(x, y)
        if original is None:
            return
        original_x, original_y = original

        frame_height, frame_width = self.current_frame.shape[:2]
        if not (
            0 <= original_x < frame_width
            and 0 <= original_y < frame_height
        ):
            return

        if self.annotation_kind == "line":
            self.line_observations.append(
                FrameLineObservation(
                    frame_index=self.annotation_frame_index,
                    line_key=self.current_line_key,
                    image_point=(float(original_x), float(original_y)),
                )
            )
            count = sum(
                observation.frame_index == self.annotation_frame_index
                and observation.line_key == self.current_line_key
                for observation in self.line_observations
            )
            self.status_message = (
                f"Lijnpunt {count}: "
                f"{self.field_lines[self.current_line_key].name}"
            )
            if count >= 3:
                self._advance_guided_step("line", self.current_line_key)
            return

        if self.current_landmark_key not in (5, 6, 7, 8):
            self.status_message = (
                "Kies eerst A voor het LINKER doel of B voor het RECHTER doel."
            )
            return

        self.observations = [
            observation
            for observation in self.observations
            if not (
                observation.frame_index == self.annotation_frame_index
                and observation.landmark_key == self.current_landmark_key
            )
        ]

        self.observations.append(
            LandmarkObservation(
                frame_index=self.annotation_frame_index,
                landmark_key=self.current_landmark_key,
                image_point=(float(original_x), float(original_y)),
            )
        )

        self.status_message = (
            f"Geplaatst: {self.landmarks[self.current_landmark_key].name}"
        )
        if not self._advance_guided_step("point", self.current_landmark_key):
            self._select_next_landmark()

    def _canvas_to_original(
        self,
        x: int,
        y: int,
    ) -> tuple[float, float] | None:
        image_x, image_y, image_w, image_h = self.display_image_rect
        if not (
            image_x <= x < image_x + image_w
            and image_y <= y < image_y + image_h
        ):
            return None
        origin_x, origin_y = self.display_view_origin
        return (
            origin_x + (x - image_x) / self.display_scale,
            origin_y + (y - image_y) / self.display_scale,
        )

    def _change_zoom(
        self,
        multiplier: float,
        anchor: tuple[float, float] | None = None,
    ) -> None:
        if self.current_frame is None:
            return
        self.zoom_factor = min(8.0, max(1.0, self.zoom_factor * multiplier))
        if anchor is not None:
            self.zoom_center = anchor
        elif self.zoom_center is None:
            height, width = self.current_frame.shape[:2]
            self.zoom_center = (width / 2.0, height / 2.0)
        self.status_message = f"Zoom: {self.zoom_factor:.1f}x | 0 = volledig beeld"

    def _reset_zoom(self) -> None:
        self.zoom_factor = 1.0
        self.zoom_center = None
        self.status_message = "Zoom hersteld naar volledig beeld."

    def _start_goal_first_workflow(self, goal: str) -> None:
        if goal == "A":
            self.guided_steps = [
                ("point", 5),
                ("point", 6),
            ]
            goal_description = "LINKER doel in de veldkaart/panorama"
        else:
            self.guided_steps = [
                ("point", 8),
                ("point", 7),
            ]
            goal_description = "RECHTER doel in de veldkaart/panorama"
        self.guided_step_index = 0
        self._activate_guided_step()
        self.status_message = (
            f"DOEL {goal}: {goal_description}. Begin met de paal aan de "
            "VERRE zijlijn. Klik exact waar de paal de grond en witte "
            "doellijn raakt."
        )

    def _advance_guided_step(self, kind: str, key: int) -> bool:
        if not self.guided_steps:
            return False
        expected = self.guided_steps[self.guided_step_index]
        if expected != (kind, key):
            return False
        self.guided_step_index += 1
        if self.guided_step_index >= len(self.guided_steps):
            self.guided_steps = []
            self.status_message = (
                "Doel-eerst workflow voor dit frame voltooid. Druk N voor "
                "het volgende frame of Enter om te controleren."
            )
            return True
        self._activate_guided_step()
        return True

    def _activate_guided_step(self) -> None:
        kind, key = self.guided_steps[self.guided_step_index]
        self.annotation_kind = kind
        if kind == "point":
            self.current_landmark_key = key
            self.status_message = self._guided_point_instruction(key)
        else:
            self.current_line_key = key
            self.status_message = (
                f"WIJS LIJN AAN: {self.field_lines[key].name}. Klik minstens "
                "3 goed verspreide plekken op dezelfde witte lijn."
            )

    @staticmethod
    def _guided_point_instruction(key: int) -> str:
        goal = "A (LINKER doel)" if key in (5, 6) else "B (RECHTER doel)"
        side = (
            "VERRE paal, richting de bovenste/verste zijlijn"
            if key in (5, 8)
            else "DICHTSTBIJZIJNDE paal, richting de onderste/nabije zijlijn"
        )
        return (
            f"DOEL {goal}: klik bij de {side} exact op het CONTACTPUNT "
            "van paal, grond en witte doellijn."
        )

    def _load_frame(self, frame_number: int) -> None:
        frame_number = max(0, min(frame_number, self.total_frames - 1))

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = self.capture.read()

        if not success:
            self.status_message = "Videoframe kon niet worden gelezen."
            return

        self.current_frame_number = frame_number
        self.current_frame = frame

    def _add_current_frame(self) -> None:
        if self.current_frame is None:
            return

        if any(
            item.frame_number == self.current_frame_number
            for item in self.selected_frames
        ):
            self.status_message = "Dit frame is al geselecteerd."
            return

        selected = SelectedFrame(
            frame=self.current_frame.copy(),
            frame_number=self.current_frame_number,
            time_seconds=self.current_frame_number / self.fps,
        )

        self.selected_frames.append(selected)
        self.selected_frames.sort(key=lambda item: item.frame_number)
        self.status_message = (
            f"Frame op {selected.time_seconds:.2f}s toegevoegd."
        )

    def _remove_last_frame(self) -> None:
        if not self.selected_frames:
            return

        removed = self.selected_frames.pop()
        self.status_message = (
            f"Frame op {removed.time_seconds:.2f}s verwijderd."
        )

    def _next_annotation_frame(self) -> None:
        if self.annotation_frame_index >= len(self.selected_frames) - 1:
            return

        self.annotation_frame_index += 1
        self.current_frame = self.selected_frames[
            self.annotation_frame_index
        ].frame.copy()
        self.current_landmark_key = None
        self.guided_steps = []
        self.annotation_kind = "point"
        self._reset_zoom()
        self.status_message = (
            "Kies A voor het LINKER doel of B voor het RECHTER doel."
        )

    def _previous_annotation_frame(self) -> None:
        if self.annotation_frame_index <= 0:
            return

        self.annotation_frame_index -= 1
        self.current_frame = self.selected_frames[
            self.annotation_frame_index
        ].frame.copy()
        self.current_landmark_key = None
        self.guided_steps = []
        self.annotation_kind = "point"
        self._reset_zoom()
        self.status_message = (
            "Kies A voor het LINKER doel of B voor het RECHTER doel."
        )

    def _undo_last_observation(self) -> None:
        if self.annotation_kind == "line":
            indices = [
                index
                for index, observation in enumerate(self.line_observations)
                if observation.frame_index == self.annotation_frame_index
            ]
            if indices:
                removed = self.line_observations.pop(indices[-1])
                self.current_line_key = removed.line_key
                self.status_message = "Laatste lijnpunt verwijderd."
            return

        indices = [
            index
            for index, observation in enumerate(self.observations)
            if observation.frame_index == self.annotation_frame_index
        ]

        if not indices:
            return

        removed = self.observations.pop(indices[-1])
        self.current_landmark_key = removed.landmark_key
        self.status_message = (
            f"Verwijderd: {self.landmarks[removed.landmark_key].name}"
        )

    def _clear_current_frame(self) -> None:
        self.observations = [
            observation
            for observation in self.observations
            if observation.frame_index != self.annotation_frame_index
        ]
        self.line_observations = [
            observation
            for observation in self.line_observations
            if observation.frame_index != self.annotation_frame_index
        ]
        self.current_landmark_key = None
        self.guided_steps = []
        self.annotation_kind = "point"
        self.status_message = "Alle punten uit dit frame verwijderd."

    def _validate_keyframe_observations(
        self,
    ) -> tuple[list[int], list[str]]:
        errors: list[str] = []
        goalpost_observations = [
            item for item in self.observations
            if item.landmark_key in (5, 6, 7, 8)
        ]
        contributing_frames = sorted(
            {
                item.frame_index
                for item in goalpost_observations
            }
        )
        for frame_index in range(len(self.selected_frames)):
            for line_key in self.field_lines:
                count = sum(
                    item.frame_index == frame_index
                    and item.line_key == line_key
                    for item in self.line_observations
                )
                if 0 < count < 3:
                    errors.append(
                        f"Frame {frame_index + 1}: "
                        f"{self.field_lines[line_key].name} heeft {count} "
                        "klik(ken); maak er minimaal 3 van."
                    )

        marked_goalposts = {
            item.landmark_key for item in goalpost_observations
        }
        missing_goalposts = [
            self.landmarks[key].name
            for key in (5, 6, 8, 7)
            if key not in marked_goalposts
        ]
        if missing_goalposts:
            errors.append("Nog nodig: " + ", ".join(missing_goalposts) + ".")

        if len(contributing_frames) < 2:
            errors.append(
                "Gebruik minimaal twee verschillende videoframes: één rond "
                "Doel A en één rond Doel B."
            )
        return contributing_frames if not errors else [], errors

    def _keyframe_constraint_errors(
        self,
        points: list[LandmarkObservation],
        line_points: list[FrameLineObservation],
    ) -> list[str]:
        return _constraint_errors_for_keyframe(
            points,
            line_points,
            self.field_lines,
        )

    def _select_next_landmark(self) -> None:
        marked = {
            observation.landmark_key
            for observation in self.observations
            if observation.frame_index == self.annotation_frame_index
        }

        keys = list(self.landmarks)
        start_index = (
            keys.index(self.current_landmark_key)
            if self.current_landmark_key in keys
            else -1
        )

        for offset in range(1, len(keys) + 1):
            candidate = keys[(start_index + offset) % len(keys)]
            if candidate not in marked:
                self.current_landmark_key = candidate
                return

        self.current_landmark_key = None

    def _render(self) -> np.ndarray:
        canvas = np.full(
            (self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3),
            28,
            dtype=np.uint8,
        )

        self._draw_sidebar(canvas)
        self._draw_video(canvas)
        return canvas

    def _draw_sidebar(self, canvas: np.ndarray) -> None:
        cv2.rectangle(
            canvas,
            (0, 0),
            (self.SIDEBAR_WIDTH, self.WINDOW_HEIGHT),
            (42, 42, 42),
            thickness=-1,
        )

        y = 38
        self._text(canvas, "FOOTBALL AI", (24, y), 0.85, (255, 255, 255), 2)
        y += 35
        title = (
            "1. FRAMES KIEZEN"
            if self.mode == "select"
            else "2. KEYFRAME ANNOTATIE"
        )
        self._text(canvas, title, (24, y), 0.65, (0, 220, 255), 2)
        y += 35

        if self.mode == "select":
            self._draw_select_sidebar(canvas, y)
        else:
            self._draw_annotate_sidebar(canvas, y)

        if self.mode == "annotate":
            cv2.rectangle(
                canvas,
                (12, self.WINDOW_HEIGHT - 258),
                (self.SIDEBAR_WIDTH - 12, self.WINDOW_HEIGHT - 122),
                (28, 28, 28),
                thickness=-1,
            )
        self._draw_wrapped_text(
            canvas,
            self.status_message,
            x=24,
            y=(
                self.WINDOW_HEIGHT - 238
                if self.mode == "annotate"
                else self.WINDOW_HEIGHT - 75
            ),
            max_width=self.SIDEBAR_WIDTH - 48,
            line_height=22,
            color=(230, 230, 230),
        )

    def _draw_select_sidebar(self, canvas: np.ndarray, y: int) -> None:
        self._draw_wrapped_text(
            canvas,
            "Kies 3-8 frames met overlap: linker doel, minimaal één "
            "tussenbeeld en rechter doel. Alleen bij de doelen klik je palen.",
            24,
            y,
            310,
            22,
            (230, 230, 230),
        )
        y += 77

        self._text(
            canvas,
            f"Frame: {self.current_frame_number}",
            (24, y),
            0.55,
            (255, 255, 255),
            1,
        )
        y += 25
        self._text(
            canvas,
            f"Tijd: {self.current_frame_number / self.fps:.2f}s",
            (24, y),
            0.55,
            (255, 255, 255),
            1,
        )
        y += 35

        controls = [
            "A / D  = 1 seconde terug / vooruit",
            "S / W  = 5 seconden terug / vooruit",
            "Pijltjes = 1 frame terug / vooruit",
            "Spatie = huidig frame toevoegen",
            "Backspace = laatste frame verwijderen",
            "Enter = doorgaan naar annotatie",
            "Esc = afbreken",
        ]

        for line in controls:
            self._text(canvas, line, (24, y), 0.48, (215, 215, 215), 1)
            y += 24

        y += 12
        self._text(
            canvas,
            f"Geselecteerd: {len(self.selected_frames)}",
            (24, y),
            0.58,
            (0, 220, 255),
            2,
        )
        y += 28

        for index, frame in enumerate(self.selected_frames[-12:], start=1):
            self._text(
                canvas,
                f"{index}. {frame.time_seconds:.2f}s",
                (35, y),
                0.48,
                (230, 230, 230),
                1,
            )
            y += 22


    def _draw_annotate_sidebar(self, canvas: np.ndarray, y: int) -> None:
        if self.annotation_kind == "line":
            self._draw_line_sidebar(canvas, y)
            return

        selected = self.selected_frames[self.annotation_frame_index]
        goalpost_keys = (5, 6, 8, 7)

        current_marked = {
            item.landmark_key
            for item in self.observations
            if item.frame_index == self.annotation_frame_index
        }

        total_counts = {
            key: sum(
                1
                for item in self.observations
                if item.landmark_key == key
            )
            for key in goalpost_keys
        }

        completed_landmarks = sum(
            1 for count in total_counts.values() if count > 0
        )
        total_observations = sum(total_counts.values())
        completion_ratio = completed_landmarks / len(goalpost_keys)

        self._text(
            canvas,
            (
                f"Frame {self.annotation_frame_index + 1}/"
                f"{len(self.selected_frames)} - {selected.time_seconds:.2f}s"
            ),
            (24, y),
            0.52,
            (255, 255, 255),
            1,
        )
        y += 28

        frame_points, frame_line_points, frame_constraints = (
            self._current_frame_constraint_counts()
        )
        constraint_color = (
            (70, 200, 70) if frame_constraints > 0 else (0, 165, 255)
        )
        self._text(
            canvas,
            (
                f"BIJDRAGE DIT FRAME: {frame_points} exacte punten + "
                f"{frame_line_points} bruikbare lijnklikken"
            ),
            (24, y),
            0.40,
            constraint_color,
            1,
        )
        y += 24

        self._text(
            canvas,
            (
                f"DOELPALEN: {completed_landmarks}/{len(goalpost_keys)}  |  "
                f"{total_observations} waarnemingen"
            ),
            (24, y),
            0.46,
            (0, 220, 255),
            1,
        )
        y += 14

        bar_x = 24
        bar_y = y
        bar_w = self.SIDEBAR_WIDTH - 48
        bar_h = 12

        cv2.rectangle(
            canvas,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            (80, 80, 80),
            thickness=-1,
        )
        cv2.rectangle(
            canvas,
            (bar_x, bar_y),
            (
                bar_x + int(round(bar_w * completion_ratio)),
                bar_y + bar_h,
            ),
            (70, 200, 70),
            thickness=-1,
        )
        y += 22

        indicator_spacing = 82
        indicator_start_x = 96
        indicator_y = y + 10

        for index, key in enumerate(goalpost_keys):
            count = total_counts[key]

            if count == 0:
                indicator_color = (115, 115, 115)
            elif count == 1:
                indicator_color = (0, 220, 255)
            else:
                indicator_color = (70, 200, 70)

            indicator_x = indicator_start_x + index * indicator_spacing
            cv2.circle(
                canvas,
                (indicator_x, indicator_y),
                13,
                indicator_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            self._text(
                canvas,
                str(key),
                (indicator_x - 5, indicator_y + 5),
                0.42,
                (20, 20, 20),
                1,
            )

        y += 31

        self._draw_pitch_diagram(canvas, top=y)
        y += 185

        if self.current_landmark_key is None:
            current_text = "NU: alle zichtbare punten in dit frame afgewerkt"
        else:
            current_text = (
                f"NU: {self.landmarks[self.current_landmark_key].name}"
            )

        self._draw_wrapped_text(
            canvas,
            current_text,
            24,
            y,
            self.SIDEBAR_WIDTH - 48,
            20,
            (0, 220, 255),
        )
        y += 42

        self._text(
            canvas,
            "HUIDIG FRAME",
            (24, y),
            0.48,
            (255, 255, 255),
            2,
        )
        y += 22

        for key in goalpost_keys:
            landmark = self.landmarks[key]
            prefix = "[x]" if key in current_marked else "[ ]"
            color = (
                (0, 220, 255)
                if key == self.current_landmark_key
                else (
                    (70, 200, 70)
                    if key in current_marked
                    else (210, 210, 210)
                )
            )
            self._text(
                canvas,
                f"{prefix} {key}. {landmark.name}",
                (24, y),
                0.40,
                color,
                1,
            )
            y += 19

        controls_y = self.WINDOW_HEIGHT - 104
        controls = [
            "A = begeleid vanaf LINKER doel",
            "B = begeleid vanaf RECHTER doel",
            "Muiswiel of +/- = zoom | 0 = herstel",
            "U undo | R frame leeg | P/N frame",
            "Enter controleren | Esc stoppen",
        ]

        for line in controls:
            self._text(
                canvas,
                line,
                (24, controls_y),
                0.38,
                (185, 185, 185),
                1,
            )
            controls_y += 19

    def _draw_line_sidebar(self, canvas: np.ndarray, y: int) -> None:
        selected = self.selected_frames[self.annotation_frame_index]
        self._text(
            canvas,
            (
                f"Frame {self.annotation_frame_index + 1}/"
                f"{len(self.selected_frames)} - {selected.time_seconds:.2f}s"
            ),
            (24, y),
            0.52,
            (255, 255, 255),
            1,
        )
        y += 32
        frame_points, frame_line_points, frame_constraints = (
            self._current_frame_constraint_counts()
        )
        constraint_color = (
            (70, 200, 70) if frame_constraints > 0 else (0, 165, 255)
        )
        self._text(
            canvas,
            (
                f"BIJDRAGE DIT FRAME: {frame_points} exacte punten + "
                f"{frame_line_points} bruikbare lijnklikken"
            ),
            (24, y),
            0.40,
            constraint_color,
            1,
        )
        y += 28
        self._text(canvas, "LIJNMODUS", (24, y), 0.62, (255, 170, 0), 2)
        y += 30
        self._draw_wrapped_text(
            canvas,
            "Wijs een lijn aan met minimaal 3 globale klikken. De groene "
            "rechte lijn wordt automatisch berekend; rode klikken tellen niet.",
            24,
            y,
            self.SIDEBAR_WIDTH - 48,
            21,
            (235, 235, 235),
        )
        y += 82

        counts = {
            key: sum(
                item.frame_index == self.annotation_frame_index
                and item.line_key == key
                for item in self.line_observations
            )
            for key in self.field_lines
        }
        for key, definition in self.field_lines.items():
            selected_line = key == self.current_line_key
            color = (
                (255, 170, 0)
                if selected_line
                else (70, 200, 70)
                if counts[key] >= 3
                else (210, 210, 210)
            )
            marker = ">" if selected_line else " "
            self._text(
                canvas,
                f"{marker} {key}. {definition.name} [{counts[key]}]",
                (24, y),
                0.42,
                color,
                1,
            )
            y += 24

        y += 18
        current = self.field_lines[self.current_line_key]
        self._draw_wrapped_text(
            canvas,
            current.instruction,
            24,
            y,
            self.SIDEBAR_WIDTH - 48,
            22,
            (255, 170, 0),
        )
        y += 70
        self._draw_wrapped_text(
            canvas,
            "Gebruik bij voorkeur beide lange zijlijnen plus minimaal één "
            "dwarslijn. Wissel met M naar exacte hoeken en doelpalen.",
            24,
            y,
            self.SIDEBAR_WIDTH - 48,
            21,
            (220, 220, 220),
        )

        controls_y = self.WINDOW_HEIGHT - 104
        controls = [
            "A = opnieuw vanaf LINKER doel",
            "B = opnieuw vanaf RECHTER doel",
            "Muiswiel of +/- = zoom | 0 = herstel",
            "U undo | R frame leeg | P/N frame",
            "Enter controleren | Esc stoppen",
        ]
        for line in controls:
            self._text(
                canvas,
                line,
                (24, controls_y),
                0.38,
                (185, 185, 185),
                1,
            )
            controls_y += 19

    def _current_frame_constraint_counts(self) -> tuple[int, int, int]:
        point_count = sum(
            item.frame_index == self.annotation_frame_index
            for item in self.observations
        )
        line_point_count = 0
        for line_key in self.field_lines:
            points = [
                item.image_point
                for item in self.line_observations
                if item.frame_index == self.annotation_frame_index
                and item.line_key == line_key
            ]
            if len(points) < 3:
                line_point_count += len(points)
                continue
            try:
                fitted = fit_image_line_robustly(
                    np.asarray(points, dtype=np.float64)
                )
            except ValueError:
                continue
            line_point_count += sum(fitted.inlier_mask)
        return point_count, line_point_count, 2 * point_count + line_point_count



    def _draw_pitch_diagram(self, canvas: np.ndarray, top: int) -> None:
        left = 78
        right = self.SIDEBAR_WIDTH - 78
        pitch_top = top + 24
        pitch_bottom = top + 130

        self._text(
            canvas,
            "DOEL A",
            (18, (pitch_top + pitch_bottom) // 2 + 5),
            0.38,
            (230, 230, 230),
            1,
        )

        cv2.rectangle(
            canvas,
            (left, pitch_top),
            (right, pitch_bottom),
            (180, 180, 180),
            2,
        )
        goal_top = (pitch_top + pitch_bottom) // 2 - 22
        goal_bottom = (pitch_top + pitch_bottom) // 2 + 22
        cv2.polylines(
            canvas,
            [np.asarray(
                [[left, goal_top], [left - 12, goal_top],
                 [left - 12, goal_bottom], [left, goal_bottom]],
                dtype=np.int32,
            )],
            False,
            (180, 180, 180),
            2,
        )
        cv2.polylines(
            canvas,
            [np.asarray(
                [[right, goal_top], [right + 12, goal_top],
                 [right + 12, goal_bottom], [right, goal_bottom]],
                dtype=np.int32,
            )],
            False,
            (180, 180, 180),
            2,
        )
        cv2.line(
            canvas,
            ((left + right) // 2, pitch_top),
            ((left + right) // 2, pitch_bottom),
            (100, 100, 100),
            1,
        )

        self._text(
            canvas,
            "DOEL B",
            (right + 8, (pitch_top + pitch_bottom) // 2 + 5),
            0.38,
            (230, 230, 230),
            1,
        )

        positions = {
            5: (left, goal_top),
            6: (left, goal_bottom),
            7: (right, goal_bottom),
            8: (right, goal_top),
        }

        current_marked = {
            item.landmark_key
            for item in self.observations
            if item.frame_index == self.annotation_frame_index
        }

        total_marked = {
            item.landmark_key
            for item in self.observations
        }

        for key, point in positions.items():
            if key == self.current_landmark_key:
                color = (0, 220, 255)
            elif key in current_marked:
                color = (70, 200, 70)
            elif key in total_marked:
                color = (0, 165, 255)
            else:
                color = (115, 115, 115)

            cv2.circle(canvas, point, 10, color, -1, cv2.LINE_AA)
            self._text(
                canvas,
                str(key),
                (point[0] - 4, point[1] + 4),
                0.38,
                (20, 20, 20),
                1,
            )




    def _draw_video(self, canvas: np.ndarray) -> None:
        if self.current_frame is None:
            return

        available_width = self.WINDOW_WIDTH - self.SIDEBAR_WIDTH
        available_height = self.WINDOW_HEIGHT

        frame = self.current_frame.copy()

        if self.mode == "annotate":
            for observation in self.observations:
                if observation.frame_index != self.annotation_frame_index:
                    continue

                point = (
                    int(round(observation.image_point[0])),
                    int(round(observation.image_point[1])),
                )
                cv2.circle(frame, point, 9, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    str(observation.landmark_key),
                    (point[0] + 12, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            for line_key in self.field_lines:
                points = [
                    observation.image_point
                    for observation in self.line_observations
                    if observation.frame_index == self.annotation_frame_index
                    and observation.line_key == line_key
                ]
                integer_points = [
                    tuple(np.round(point).astype(int)) for point in points
                ]
                fitted = None
                if len(points) >= 3:
                    try:
                        fitted = fit_image_line_robustly(
                            np.asarray(points, dtype=np.float64)
                        )
                    except ValueError:
                        fitted = None
                inlier_mask = (
                    fitted.inlier_mask
                    if fitted is not None
                    else tuple(False for _ in points)
                )
                for point, is_inlier in zip(integer_points, inlier_mask):
                    color = (70, 210, 70) if is_inlier else (0, 0, 255)
                    cv2.circle(frame, point, 7, color, -1, cv2.LINE_AA)
                if fitted is not None:
                    segment = self._line_segment_in_frame(
                        fitted.equation,
                        frame.shape[1],
                        frame.shape[0],
                    )
                    if segment is not None:
                        cv2.line(
                            frame,
                            segment[0],
                            segment[1],
                            (70, 210, 70),
                            3,
                            cv2.LINE_AA,
                        )

        frame_height, frame_width = frame.shape[:2]
        if self.mode == "annotate" and self.zoom_factor > 1.0:
            crop_width = max(2, int(round(frame_width / self.zoom_factor)))
            crop_height = max(2, int(round(frame_height / self.zoom_factor)))
            if self.zoom_center is None:
                self.zoom_center = (frame_width / 2.0, frame_height / 2.0)
            center_x, center_y = self.zoom_center
            x0 = int(round(center_x - crop_width / 2.0))
            y0 = int(round(center_y - crop_height / 2.0))
            x0 = max(0, min(x0, frame_width - crop_width))
            y0 = max(0, min(y0, frame_height - crop_height))
            self.zoom_center = (
                x0 + crop_width / 2.0,
                y0 + crop_height / 2.0,
            )
            frame = frame[y0:y0 + crop_height, x0:x0 + crop_width]
            self.display_view_origin = (float(x0), float(y0))
        else:
            self.display_view_origin = (0.0, 0.0)

        view_height, view_width = frame.shape[:2]
        scale = min(
            available_width / view_width,
            available_height / view_height,
        )

        render_width = int(view_width * scale)
        render_height = int(view_height * scale)

        resized = cv2.resize(
            frame,
            (render_width, render_height),
            interpolation=cv2.INTER_AREA,
        )

        x = self.SIDEBAR_WIDTH + (available_width - render_width) // 2
        y = (available_height - render_height) // 2

        canvas[y:y + render_height, x:x + render_width] = resized
        self.display_image_rect = (x, y, render_width, render_height)
        self.display_scale = scale

    @staticmethod
    def _line_segment_in_frame(
        equation: tuple[float, float, float],
        width: int,
        height: int,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        a, b, c = equation
        candidates: list[tuple[float, float]] = []
        if abs(b) > 1e-9:
            candidates.extend([(0.0, -c / b), (width - 1.0, -(a * (width - 1.0) + c) / b)])
        if abs(a) > 1e-9:
            candidates.extend([(-c / a, 0.0), (-(b * (height - 1.0) + c) / a, height - 1.0)])
        inside = [
            (int(round(x)), int(round(y)))
            for x, y in candidates
            if -0.5 <= x <= width - 0.5 and -0.5 <= y <= height - 0.5
        ]
        unique = list(dict.fromkeys(inside))
        return (unique[0], unique[1]) if len(unique) >= 2 else None

    @staticmethod
    def _text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_wrapped_text(
        self,
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        max_width: int,
        line_height: int,
        color: tuple[int, int, int],
    ) -> None:
        words = text.split()
        lines: list[str] = []
        current = ""

        for word in words:
            candidate = word if not current else f"{current} {word}"
            width = cv2.getTextSize(
                candidate,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                1,
            )[0][0]

            if width <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        for index, line in enumerate(lines):
            self._text(
                image,
                line,
                (x, y + index * line_height),
                0.48,
                color,
                1,
            )


class MultiFramePitchCalibrator:
    def __init__(self, profile: PitchProfile) -> None:
        self.profile = profile
        self.selected_frames: list[SelectedFrame] = []
        self.observations: list[LandmarkObservation] = []
        self.line_observations: list[FrameLineObservation] = []
        self.frame_transforms: list[np.ndarray] = []
        self.frame_registration_diagnostics: list[
            FrameRegistrationDiagnostics
        ] = []
        self.quality_report: CalibrationQualityReport | None = None
        self.source_video_path: Path | None = None
        self.landmarks = self._create_landmark_definitions()
        self.field_lines = create_boundary_line_definitions(
            pitch_width=self.profile.width_m,
            pitch_length=self.profile.length_m,
        )

    def calibrate_video(self, video_path: Path) -> PitchCalibration:
        if not video_path.exists():
            raise FileNotFoundError(f"Video niet gevonden: {video_path}")

        self.source_video_path = video_path
        app = OpenCvCalibrationApp(
            video_path=video_path,
            landmarks=self.landmarks,
            field_lines=self.field_lines,
        )
        (
            self.selected_frames,
            self.observations,
            self.line_observations,
        ) = app.run()
        transforms = self._calculate_frame_transforms()
        self.frame_transforms = transforms

        # Analyseer eerst de geometrie van alle frame-transformaties.
        panorama_report = PanoramaBuilder(
                selected_frames=self.selected_frames,
                frame_transforms=transforms,
                registration_diagnostics=self.frame_registration_diagnostics,
        ).analyze()

        print()
        panorama_report.print_summary()
        print()

        keyframes, keyframe_failures = self._calculate_keyframes()
        if not keyframes:
            raise RuntimeError("Geen enkel keyframe kon worden gekalibreerd.")

        observed_points: list[tuple[float, float]] = []
        expected_points: list[tuple[float, float]] = []
        predicted_points: list[tuple[float, float]] = []
        point_contexts: list[ControlPointContext] = []
        inlier_mask: list[bool] = []
        keyframe_by_selected_index = {
            selected_index: keyframe
            for selected_index, keyframe in keyframes
        }
        for observation in self._goalpost_observations():
            keyframe = keyframe_by_selected_index.get(observation.frame_index)
            if keyframe is None:
                continue
            landmark = self.landmarks[observation.landmark_key]
            predicted = cv2.perspectiveTransform(
                np.asarray([[landmark.pitch_point]], dtype=np.float64),
                keyframe.pitch_to_image_matrix,
            )[0, 0]
            error = float(
                np.linalg.norm(
                    np.asarray(observation.image_point) - predicted
                )
            )
            selected_frame = self.selected_frames[observation.frame_index]
            observed_points.append(observation.image_point)
            expected_points.append(landmark.pitch_point)
            predicted_points.append((float(predicted[0]), float(predicted[1])))
            inlier_mask.append(error <= 12.0 and keyframe.is_valid)
            point_contexts.append(
                ControlPointContext(
                    landmark_key=landmark.key,
                    landmark_name=landmark.name,
                    frame_index=observation.frame_index,
                    frame_number=selected_frame.frame_number,
                )
            )

        self.quality_report = calculate_quality_from_predictions(
            observed_image_points=np.asarray(
                observed_points, dtype=np.float64
            ).reshape(-1, 2),
            expected_pitch_points=np.asarray(
                expected_points, dtype=np.float64
            ).reshape(-1, 2),
            reprojected_image_points=np.asarray(
                predicted_points, dtype=np.float64
            ).reshape(-1, 2),
            inlier_mask=np.asarray(inlier_mask, dtype=bool),
            point_contexts=point_contexts,
        )
        self.quality_report = assess_calibration_quality(
            self.quality_report,
            pitch_width=self.profile.width_m,
            pitch_length=self.profile.length_m,
            frame_new_coverage=[
                frame_report.new_coverage_ratio
                for frame_report in panorama_report.frame_reports
            ],
            additional_failures=keyframe_failures,
            supporting_line_point_count=0,
            geometry_coverage=self._guided_geometry_coverage(),
            model_geometry_support=True,
        )

        reference_index, reference_keyframe = keyframes[0]
        reference = self.selected_frames[reference_index]
        frame_height, frame_width = reference.frame.shape[:2]

        print(self.quality_report.format_terminal_report())

        return PitchCalibration(
            profile=self.profile,
            image_corners=reference_keyframe.image_corners,
            image_to_pitch_matrix=reference_keyframe.image_to_pitch_matrix,
            pitch_to_image_matrix=reference_keyframe.pitch_to_image_matrix,
            source_video=str(video_path),
            source_frame_number=reference.frame_number,
            source_time_seconds=reference.time_seconds,
            frame_width=frame_width,
            frame_height=frame_height,
            quality=self.quality_report,
            keyframes=tuple(keyframe for _, keyframe in keyframes),
        )

    def _guided_geometry_coverage(self) -> tuple[float, float, float]:
        landmark_keys = {
            item.landmark_key for item in self._goalpost_observations()
        }
        complete_goal_geometry = {5, 6, 7, 8} <= landmark_keys
        length_coverage = 1.0 if complete_goal_geometry else 0.0
        width_coverage = 1.0 if complete_goal_geometry else 0.0
        hull_coverage = 1.0 if complete_goal_geometry else 0.0
        return width_coverage, length_coverage, hull_coverage

    def _goalpost_observations(self) -> list[LandmarkObservation]:
        return [
            item for item in self.observations
            if item.landmark_key in (5, 6, 7, 8)
        ]

    def _calculate_keyframes(
        self,
    ) -> tuple[list[tuple[int, CalibrationKeyframe]], list[str]]:
        failures: list[str] = []
        panorama_points: list[tuple[float, float]] = []
        pitch_points: list[tuple[float, float]] = []
        for observation in self._goalpost_observations():
            transformed = cv2.perspectiveTransform(
                np.asarray([[observation.image_point]], dtype=np.float64),
                self.frame_transforms[observation.frame_index],
            )[0, 0]
            panorama_points.append((float(transformed[0]), float(transformed[1])))
            pitch_points.append(
                self.landmarks[observation.landmark_key].pitch_point
            )
        try:
            panorama_to_pitch = estimate_homography_with_line_constraints(
                np.asarray(panorama_points, dtype=np.float64).reshape(-1, 2),
                np.asarray(pitch_points, dtype=np.float64).reshape(-1, 2),
                [],
                self.field_lines,
            )
        except ValueError as error:
            return [], [f"Gezamenlijke veldkalibratie: {error}"]

        keyframes: list[tuple[int, CalibrationKeyframe]] = []
        for frame_index, selected in enumerate(self.selected_frames):
            image_to_pitch = (
                panorama_to_pitch @ self.frame_transforms[frame_index]
            )
            image_to_pitch /= image_to_pitch[2, 2]
            pitch_to_image = np.linalg.inv(image_to_pitch)
            image_corners = cv2.perspectiveTransform(
                self.profile.world_corners.reshape(-1, 1, 2).astype(np.float64),
                pitch_to_image,
            ).reshape(-1, 2)
            height, width = selected.frame.shape[:2]
            geometry = validate_projected_pitch_geometry(
                image_corners,
                frame_width=width,
                frame_height=height,
            )
            line_rms = self._line_rms_error(
                image_to_pitch,
                [
                    item for item in self.line_observations
                    if item.frame_index == frame_index
                ],
            )
            geometry_errors = list(geometry.errors)
            if line_rms is not None and line_rms > 5.0:
                geometry_errors.append(
                    f"Lijn-RMS is {line_rms:.1f} px en groter dan 5 px."
                )
            keyframe = CalibrationKeyframe(
                frame_number=selected.frame_number,
                time_seconds=selected.time_seconds,
                image_to_pitch_matrix=image_to_pitch,
                pitch_to_image_matrix=pitch_to_image,
                image_corners=image_corners,
                point_count=sum(
                    item.frame_index == frame_index
                    for item in self._goalpost_observations()
                ),
                line_point_count=sum(
                    item.frame_index == frame_index
                    for item in self.line_observations
                ),
                line_rms_error_pixels=line_rms,
                geometry_errors=tuple(geometry_errors),
            )
            keyframes.append((frame_index, keyframe))
            failures.extend(
                f"Keyframe {frame_index + 1}: {error}"
                for error in geometry_errors
            )

        return keyframes, failures

    def _line_rms_error(
        self,
        image_to_pitch: np.ndarray,
        observations: list[FrameLineObservation],
    ) -> float | None:
        if not observations:
            return None
        filtered = filter_line_observations(
            [
                LinePointObservation(item.line_key, item.image_point)
                for item in observations
            ]
        )
        squared_errors: list[float] = []
        for observation in filtered:
            world_line = np.asarray(
                self.field_lines[observation.line_key].equation,
                dtype=np.float64,
            )
            image_line = image_to_pitch.T @ world_line
            normal = float(np.linalg.norm(image_line[:2]))
            if normal < 1e-12:
                return float("inf")
            point = np.array([*observation.image_point, 1.0])
            distance = float(abs(image_line @ point) / normal)
            squared_errors.append(distance * distance)
        return float(np.sqrt(np.mean(squared_errors)))

    def create_preview(
        self,
        calibration: PitchCalibration,
        grid_interval_m: float = 5.0,
    ) -> np.ndarray:
        """
        Maak een panorama van alle geselecteerde kalibratieframes.

        Alle frames worden naar het coördinatenstelsel van het eerste frame
        gewarpt, samengevoegd op één groter canvas en daarna voorzien van:

        - de volledige veldomtrek;
        - een raster per `grid_interval_m`;
        - een extra duidelijke middenlijn;
        - beide doelen;
        - alle handmatig aangeklikte landmarks;
        - de door de uiteindelijke homografie voorspelde landmarkposities;
        - pixelafwijkingen tussen klik en voorspelling.

        Daardoor is direct zichtbaar of:
        - de vier frames correct aan elkaar aansluiten;
        - het hele veld logisch wordt gereconstrueerd;
        - zijlijnen, achterlijnen, doelen en middenlijn overeenkomen;
        - de opgegeven veldafmetingen geometrisch plausibel zijn.
        """
        if not self.selected_frames:
            raise RuntimeError("Geen geselecteerde frames beschikbaar.")

        if len(self.frame_transforms) != len(self.selected_frames):
            self.frame_transforms = self._calculate_frame_transforms()

        panorama, reference_to_canvas = self._build_panorama(
            self.selected_frames,
            self.frame_transforms,
        )

        keyframe_by_number = {
            keyframe.frame_number: keyframe
            for keyframe in calibration.keyframes
        }
        pitch_to_canvas_by_frame: dict[int, np.ndarray] = {}
        for frame_index, selected in enumerate(self.selected_frames):
            keyframe = keyframe_by_number.get(selected.frame_number)
            if keyframe is None:
                continue
            pitch_to_canvas = (
                reference_to_canvas
                @ self.frame_transforms[frame_index]
                @ keyframe.pitch_to_image_matrix
            )
            pitch_to_canvas /= pitch_to_canvas[2, 2]
            pitch_to_canvas_by_frame[frame_index] = pitch_to_canvas
            self._draw_keyframe_pitch_grid(
                panorama,
                pitch_to_canvas,
                grid_interval_m,
            )

        for observation_index, observation in enumerate(self.observations):
            if observation.frame_index >= len(self.frame_transforms):
                continue

            clicked_source = np.asarray(
                [[observation.image_point]],
                dtype=np.float32,
            )

            clicked_reference = cv2.perspectiveTransform(
                clicked_source,
                self.frame_transforms[observation.frame_index],
            )[0, 0]

            clicked_canvas = cv2.perspectiveTransform(
                np.asarray([[clicked_reference]], dtype=np.float32),
                reference_to_canvas,
            )[0, 0]

            landmark = self.landmarks[observation.landmark_key]
            pitch_to_canvas = pitch_to_canvas_by_frame.get(
                observation.frame_index
            )
            if pitch_to_canvas is None:
                continue
            predicted_canvas = cv2.perspectiveTransform(
                np.asarray(
                    [[landmark.pitch_point]],
                    dtype=np.float32,
                ),
                pitch_to_canvas,
            )[0, 0]

            if not (
                np.all(np.isfinite(clicked_canvas))
                and np.all(np.isfinite(predicted_canvas))
            ):
                continue

            point_quality = self._get_point_quality(
                calibration.quality,
                observation_index,
                frame_index=observation.frame_index,
                landmark_key=observation.landmark_key,
            )
            error_pixels = (
                point_quality.error_pixels
                if point_quality is not None
                else float(np.linalg.norm(clicked_canvas - predicted_canvas))
            )

            clicked_point = tuple(np.round(clicked_canvas).astype(int))
            predicted_point = tuple(np.round(predicted_canvas).astype(int))

            if point_quality is None:
                error_color = (0, 165, 255)
            elif point_quality.is_inlier:
                error_color = (60, 210, 60)
            else:
                error_color = (0, 0, 255)

            cv2.line(
                panorama,
                clicked_point,
                predicted_point,
                error_color,
                2,
                cv2.LINE_AA,
            )

            cv2.circle(
                panorama,
                clicked_point,
                8,
                (0, 255, 255),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                panorama,
                clicked_point,
                11,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

            self._draw_cross(
                panorama,
                predicted_point,
                color=(255, 0, 255),
                size=11,
                thickness=3,
            )

            label = (
                f"{observation.landmark_key} "
                f"F{observation.frame_index + 1} "
                f"{error_pixels:.1f}px"
            )
            cv2.putText(
                panorama,
                label,
                (clicked_point[0] + 12, clicked_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                error_color,
                2,
                cv2.LINE_AA,
            )

        self._draw_quality_dashboard(panorama, calibration.quality)
        return panorama

    def _draw_keyframe_pitch_grid(
        self,
        image: np.ndarray,
        pitch_to_image: np.ndarray,
        grid_interval_m: float,
    ) -> None:
        pitch_corners = cv2.perspectiveTransform(
            self.profile.world_corners.reshape(-1, 1, 2).astype(np.float32),
            pitch_to_image,
        ).reshape(-1, 2)
        cv2.polylines(
            image,
            [np.round(pitch_corners).astype(np.int32)],
            isClosed=True,
            color=(0, 255, 255),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        for x_value in np.arange(
            0.0,
            self.profile.width_m + 0.001,
            grid_interval_m,
            dtype=np.float32,
        ):
            self._draw_pitch_line(
                image,
                pitch_to_image,
                (float(x_value), 0.0),
                (float(x_value), self.profile.length_m),
                color=(255, 180, 0),
                thickness=1,
            )
        for y_value in np.arange(
            0.0,
            self.profile.length_m + 0.001,
            grid_interval_m,
            dtype=np.float32,
        ):
            self._draw_pitch_line(
                image,
                pitch_to_image,
                (0.0, float(y_value)),
                (self.profile.width_m, float(y_value)),
                color=(255, 180, 0),
                thickness=1,
            )
        middle_y = self.profile.length_m / 2.0
        self._draw_pitch_line(
            image,
            pitch_to_image,
            (0.0, middle_y),
            (self.profile.width_m, middle_y),
            color=(0, 255, 255),
            thickness=4,
        )
        self._draw_goal_reference(
            image,
            pitch_to_image,
            self.profile.goal_a_posts,
        )
        self._draw_goal_reference(
            image,
            pitch_to_image,
            self.profile.goal_b_posts,
        )

    @staticmethod
    def _build_panorama(
        selected_frames: list[SelectedFrame],
        frame_transforms: list[np.ndarray],
        padding: int = 80,
        maximum_dimension: int = 10000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Warp alle geselecteerde frames naar het referentiecoördinatenstelsel.

        `frame_transforms[i]` zet frame i om naar frame 0. Op basis van de
        getransformeerde beeldhoeken wordt een canvas gemaakt dat groot genoeg
        is voor alle frames. Het resultaat bevat tevens de matrix die punten
        uit het referentieframe naar het panoramocanvas verplaatst.
        """
        if len(selected_frames) != len(frame_transforms):
            raise RuntimeError(
                "Aantal frames en aantal frame-transformaties komen niet overeen."
            )

        all_corners: list[np.ndarray] = []

        for selected, transform in zip(selected_frames, frame_transforms):
            height, width = selected.frame.shape[:2]
            corners = np.asarray(
                [
                    [0.0, 0.0],
                    [float(width - 1), 0.0],
                    [float(width - 1), float(height - 1)],
                    [0.0, float(height - 1)],
                ],
                dtype=np.float32,
            ).reshape(-1, 1, 2)

            warped_corners = cv2.perspectiveTransform(
                corners,
                transform.astype(np.float64),
            ).reshape(-1, 2)

            if not np.all(np.isfinite(warped_corners)):
                raise RuntimeError(
                    "Een frame-transformatie leverde ongeldige panoramapunten op."
                )

            all_corners.append(warped_corners)

        combined = np.vstack(all_corners)
        minimum = np.floor(combined.min(axis=0)).astype(int)
        maximum = np.ceil(combined.max(axis=0)).astype(int)

        canvas_width = int(maximum[0] - minimum[0] + 1 + 2 * padding)
        canvas_height = int(maximum[1] - minimum[1] + 1 + 2 * padding)

        if canvas_width <= 0 or canvas_height <= 0:
            raise RuntimeError("Ongeldige afmetingen voor het panoramocanvas.")

        if (
            canvas_width > maximum_dimension
            or canvas_height > maximum_dimension
        ):
            raise RuntimeError(
                "Het berekende panorama is onrealistisch groot. "
                "Waarschijnlijk is een frame verkeerd geregistreerd."
            )

        translation = np.asarray(
            [
                [1.0, 0.0, float(-minimum[0] + padding)],
                [0.0, 1.0, float(-minimum[1] + padding)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        accumulator = np.zeros(
            (canvas_height, canvas_width, 3),
            dtype=np.float32,
        )
        total_weight = np.zeros(
            (canvas_height, canvas_width),
            dtype=np.float32,
        )

        for selected, transform in zip(selected_frames, frame_transforms):
            frame_to_canvas = translation @ transform
            frame_to_canvas /= frame_to_canvas[2, 2]

            warped_frame = cv2.warpPerspective(
                selected.frame,
                frame_to_canvas,
                (canvas_width, canvas_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            source_mask = np.full(
                selected.frame.shape[:2],
                255,
                dtype=np.uint8,
            )
            warped_mask = cv2.warpPerspective(
                source_mask,
                frame_to_canvas,
                (canvas_width, canvas_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            valid = warped_mask > 0
            if not np.any(valid):
                continue

            # Zachte overgangen in overlappende gebieden.
            binary_mask = valid.astype(np.uint8)
            distance = cv2.distanceTransform(
                binary_mask,
                cv2.DIST_L2,
                3,
            )
            weight = np.where(
                valid,
                np.maximum(distance, 1.0),
                0.0,
            ).astype(np.float32)

            accumulator += warped_frame.astype(np.float32) * weight[..., None]
            total_weight += weight

        if not np.any(total_weight > 0):
            raise RuntimeError("De geselecteerde frames konden niet worden samengevoegd.")

        panorama = np.zeros_like(accumulator, dtype=np.uint8)
        valid_pixels = total_weight > 0
        panorama[valid_pixels] = np.clip(
            accumulator[valid_pixels]
            / total_weight[valid_pixels, None],
            0,
            255,
        ).astype(np.uint8)

        return panorama, translation

    @staticmethod
    def _draw_cross(
        image: np.ndarray,
        point: tuple[int, int],
        color: tuple[int, int, int],
        size: int,
        thickness: int,
    ) -> None:
        x, y = point
        cv2.line(
            image,
            (x - size, y - size),
            (x + size, y + size),
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.line(
            image,
            (x - size, y + size),
            (x + size, y - size),
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_quality_dashboard(
        image: np.ndarray,
        quality: CalibrationQualityReport | None,
    ) -> None:
        panel_x = 18
        panel_y = 18
        panel_w = min(700, max(300, image.shape[1] - 36))
        panel_h = min(360, max(180, image.shape[0] - 36))

        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)

        x = panel_x + 16
        y = panel_y + 30
        MultiFramePitchCalibrator._dashboard_text(
            image, "KALIBRATIE QA", (x, y), 0.72, (255, 255, 255), 2
        )
        MultiFramePitchCalibrator._dashboard_text(
            image,
            "Geel = klik | magenta = voorspeld | groen = inlier | rood = outlier",
            (x, y + 30),
            0.45,
            (225, 225, 225),
            1,
        )

        if quality is None:
            MultiFramePitchCalibrator._dashboard_text(
                image,
                "Geen kwaliteitsrapport beschikbaar.",
                (x, y + 72),
                0.55,
                (0, 165, 255),
                2,
            )
            return

        MultiFramePitchCalibrator._dashboard_text(
            image,
            (
                f"Punten {quality.point_count}   |   "
                f"Inliers {quality.inlier_count}   |   "
                f"Outliers {quality.outlier_count}"
            ),
            (x, y + 72),
            0.58,
            (255, 255, 255),
            2,
        )

        columns = ("Gem.", "Mediaan", "RMS", "Max.")
        column_x = (x + 250, x + 360, x + 475, x + 580)
        for label, label_x in zip(columns, column_x):
            MultiFramePitchCalibrator._dashboard_text(
                image,
                label,
                (label_x, y + 116),
                0.43,
                (190, 190, 190),
                1,
            )

        MultiFramePitchCalibrator._draw_dashboard_statistics_row(
            image,
            "Alle punten",
            quality.all_points,
            x,
            y + 153,
            column_x,
        )
        MultiFramePitchCalibrator._draw_dashboard_statistics_row(
            image,
            "Alleen inliers",
            quality.inlier_points,
            x,
            y + 190,
            column_x,
        )

        outliers = [
            point_error
            for point_error in quality.point_errors
            if not point_error.is_inlier
        ]
        MultiFramePitchCalibrator._dashboard_text(
            image,
            "Outlierdetails",
            (x, y + 230),
            0.44,
            (190, 190, 190),
            1,
        )
        if not outliers:
            MultiFramePitchCalibrator._dashboard_text(
                image,
                "Geen outliers gedetecteerd.",
                (x + 130, y + 230),
                0.44,
                (60, 210, 60),
                1,
            )
        else:
            for row_index, point_error in enumerate(outliers[:2]):
                MultiFramePitchCalibrator._dashboard_text(
                    image,
                    MultiFramePitchCalibrator._format_dashboard_outlier(
                        point_error
                    ),
                    (x + 130, y + 230 + row_index * 28),
                    0.42,
                    (80, 80, 255),
                    1,
                )
            if len(outliers) > 2:
                MultiFramePitchCalibrator._dashboard_text(
                    image,
                    f"+ {len(outliers) - 2} overige outlier(s)",
                    (x + 130, y + 286),
                    0.40,
                    (80, 80, 255),
                    1,
                )

        MultiFramePitchCalibrator._dashboard_text(
            image,
            MultiFramePitchCalibrator._format_dashboard_assessment(quality),
            (x, y + 320),
            0.48,
            MultiFramePitchCalibrator._assessment_color(quality),
            2,
        )

    @staticmethod
    def _format_dashboard_assessment(
        quality: CalibrationQualityReport,
    ) -> str:
        if quality.assessment is None:
            return "Status niet beschikbaar; kalibreer opnieuw."
        return (
            f"STATUS {quality.assessment.status.value} | "
            f"CONFIDENCE {quality.assessment.confidence_score:.1f}/100"
        )

    @staticmethod
    def _assessment_color(
        quality: CalibrationQualityReport,
    ) -> tuple[int, int, int]:
        if quality.assessment is None:
            return (0, 165, 255)
        status = quality.assessment.status.value
        if status == "PASS":
            return (60, 210, 60)
        if status == "WARNING":
            return (0, 165, 255)
        return (80, 80, 255)

    @staticmethod
    def _format_dashboard_outlier(
        point_error: PointReprojectionError,
    ) -> str:
        landmark = point_error.landmark_name or (
            f"Punt {point_error.point_index + 1}"
        )
        frame = (
            f"F{point_error.frame_index + 1}"
            if point_error.frame_index is not None
            else "frame onbekend"
        )
        return f"{landmark} | {frame} | {point_error.error_pixels:.1f} px"

    @staticmethod
    def _draw_dashboard_statistics_row(
        image: np.ndarray,
        label: str,
        statistics: ErrorStatistics,
        x: int,
        y: int,
        column_x: tuple[int, int, int, int],
    ) -> None:
        MultiFramePitchCalibrator._dashboard_text(
            image, label, (x, y), 0.48, (235, 235, 235), 1
        )
        values = (
            statistics.mean_error,
            statistics.median_error,
            statistics.rms_error,
            statistics.max_error,
        )
        for value, value_x in zip(values, column_x):
            text = "n.v.t." if value is None else f"{value:.1f} px"
            MultiFramePitchCalibrator._dashboard_text(
                image, text, (value_x, y), 0.45, (255, 255, 255), 1
            )

    @staticmethod
    def _dashboard_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _get_point_quality(
        quality: CalibrationQualityReport | None,
        point_index: int,
        frame_index: int | None = None,
        landmark_key: int | None = None,
    ) -> PointReprojectionError | None:
        if quality is None:
            return None
        if frame_index is not None and landmark_key is not None:
            match = next(
                (
                    item for item in quality.point_errors
                    if item.frame_index == frame_index
                    and item.landmark_key == landmark_key
                ),
                None,
            )
            if match is not None:
                return match
        if point_index >= len(quality.point_errors):
            return None
        return quality.point_errors[point_index]

    def _create_landmark_definitions(self) -> dict[int, LandmarkDefinition]:
        width = float(self.profile.width_m)
        length = float(self.profile.length_m)
        goal_width = float(self.profile.goal_width_m)

        left_goal_post_left = (width - goal_width) / 2.0
        left_goal_post_right = (width + goal_width) / 2.0

        return {
            1: LandmarkDefinition(1, "Hoek A - kaart-boven", (0.0, 0.0)),
            2: LandmarkDefinition(2, "Hoek B - kaart-boven", (0.0, length)),
            3: LandmarkDefinition(3, "Hoek A - kaart-onder", (width, 0.0)),
            4: LandmarkDefinition(4, "Hoek B - kaart-onder", (width, length)),
            5: LandmarkDefinition(
                5,
                "Doel A (links) - grondpunt verre paal",
                (left_goal_post_left, 0.0),
            ),
            6: LandmarkDefinition(
                6,
                "Doel A (links) - grondpunt dichtstbijzijnde paal",
                (left_goal_post_right, 0.0),
            ),
            7: LandmarkDefinition(
                7,
                "Doel B (rechts) - grondpunt dichtstbijzijnde paal",
                (left_goal_post_right, length),
            ),
            8: LandmarkDefinition(
                8,
                "Doel B (rechts) - grondpunt verre paal",
                (left_goal_post_left, length),
            ),
        }

    def _calculate_frame_transforms(self) -> list[np.ndarray]:
        transforms: list[np.ndarray] = [
            np.eye(3, dtype=np.float64)
        ]
        self.frame_registration_diagnostics = []

        bridge_capture = (
            cv2.VideoCapture(str(self.source_video_path))
            if self.source_video_path is not None
            else None
        )
        for frame_index in range(1, len(self.selected_frames)):
            previous_frame = self.selected_frames[
                frame_index - 1
            ].frame
            current_frame = self.selected_frames[
                frame_index
            ].frame

            current_to_previous, diagnostics = self._estimate_transform_chain(
                previous_frame=previous_frame,
                current_frame=current_frame,
                previous_frame_number=self.selected_frames[
                    frame_index - 1
                ].frame_number,
                current_frame_number=self.selected_frames[frame_index].frame_number,
                source_frame_index=frame_index,
                target_frame_index=frame_index - 1,
                capture=bridge_capture,
            )
            self.frame_registration_diagnostics.append(diagnostics)

            current_to_reference = (
                transforms[frame_index - 1]
                @ current_to_previous
            )
            current_to_reference /= current_to_reference[2, 2]
            transforms.append(current_to_reference)

            print(
                f"Frame {frame_index + 1} gekoppeld aan referentieframe."
            )

        if bridge_capture is not None:
            bridge_capture.release()
        return transforms

    def _estimate_transform_chain(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        previous_frame_number: int,
        current_frame_number: int,
        source_frame_index: int,
        target_frame_index: int,
        capture: cv2.VideoCapture | None,
    ) -> tuple[np.ndarray, FrameRegistrationDiagnostics]:
        gap = current_frame_number - previous_frame_number
        if capture is None or gap <= 45:
            return self._estimate_image_transform(
                current_frame,
                previous_frame,
                source_frame_index,
                target_frame_index,
            )

        step_numbers = list(
            range(previous_frame_number + 30, current_frame_number, 30)
        ) + [current_frame_number]
        accumulated = np.eye(3, dtype=np.float64)
        target = previous_frame
        diagnostics_items: list[FrameRegistrationDiagnostics] = []
        for step_number in step_numbers:
            if step_number == current_frame_number:
                source = current_frame
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES, step_number)
                success, source = capture.read()
                if not success:
                    raise RuntimeError(
                        f"Automatisch brugframe {step_number} kon niet worden gelezen."
                    )
            try:
                step_transform, step_diagnostics = self._estimate_image_transform(
                    source,
                    target,
                    source_frame_index,
                    target_frame_index,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    "Automatische framekoppeling mislukte rond videoframe "
                    f"{step_number}. Kies een extra handmatig tussenframe."
                ) from error
            accumulated = accumulated @ step_transform
            accumulated /= accumulated[2, 2]
            diagnostics_items.append(step_diagnostics)
            target = source

        candidate_matches = sum(
            item.candidate_matches or 0 for item in diagnostics_items
        )
        inlier_count = sum(item.inlier_count or 0 for item in diagnostics_items)
        errors = [
            item.median_error_pixels for item in diagnostics_items
            if item.median_error_pixels is not None
        ]
        return accumulated, FrameRegistrationDiagnostics(
            source_frame_index=source_frame_index,
            target_frame_index=target_frame_index,
            method=f"Automatische ORB/ECC-keten ({len(step_numbers)} stappen)",
            candidate_matches=candidate_matches or None,
            inlier_count=inlier_count or None,
            median_error_pixels=float(np.median(errors)) if errors else None,
        )

    def _estimate_image_transform(
        self,
        source: np.ndarray,
        target: np.ndarray,
        source_frame_index: int,
        target_frame_index: int,
    ) -> tuple[np.ndarray, FrameRegistrationDiagnostics]:
        try:
            return self._estimate_orb_transform(
                source,
                target,
                source_frame_index,
                target_frame_index,
            )
        except RuntimeError:
            return self._estimate_ecc_transform(
                source,
                target,
                source_frame_index,
                target_frame_index,
            )

    @staticmethod
    def _estimate_orb_transform(
        source: np.ndarray,
        target: np.ndarray,
        source_frame_index: int,
        target_frame_index: int,
    ) -> tuple[np.ndarray, FrameRegistrationDiagnostics]:
        """
        Registreer het bronframe op het doelframe met ORB-kenmerken.

        Voor de panorama-opbouw gebruiken we bewust een affine transformatie
        in plaats van een volledige projectieve homografie. Een homografie kan
        buiten het overlappende beeldgebied extreem vervormen en daardoor een
        praktisch oneindig panoramocanvas veroorzaken.

        De affine transformatie ondersteunt:

        - translatie;
        - rotatie;
        - uniforme schaal.

        Dit is voor een camerapan doorgaans stabieler dan een vrije homografie.
        """
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(
            nfeatures=9000,
            scaleFactor=1.2,
            nlevels=8,
        )

        source_keypoints, source_descriptors = orb.detectAndCompute(
            source_gray,
            None,
        )
        target_keypoints, target_descriptors = orb.detectAndCompute(
            target_gray,
            None,
        )

        if source_descriptors is None or target_descriptors is None:
            raise RuntimeError("ORB vond onvoldoende kenmerken.")

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False,
        )

        pairs = matcher.knnMatch(
            source_descriptors,
            target_descriptors,
            k=2,
        )

        good_matches = []

        for pair in pairs:
            if len(pair) != 2:
                continue

            best, second = pair

            if best.distance < 0.78 * second.distance:
                good_matches.append(best)

        if len(good_matches) < 12:
            raise RuntimeError("ORB vond te weinig overeenkomsten.")

        source_points = np.float32(
            [
                source_keypoints[match.queryIdx].pt
                for match in good_matches
            ]
        )

        target_points = np.float32(
            [
                target_keypoints[match.trainIdx].pt
                for match in good_matches
            ]
        )

        affine, mask = cv2.estimateAffinePartial2D(
            source_points,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=5000,
            confidence=0.995,
            refineIters=25,
        )

        if affine is None or mask is None:
            raise RuntimeError("ORB-affine registratie mislukt.")

        inlier_count = int(mask.sum())

        if inlier_count < 10:
            raise RuntimeError(
                f"ORB-affine registratie had te weinig inliers: "
                f"{inlier_count}."
            )

        transform = np.vstack(
            [
                affine.astype(np.float64),
                np.array(
                    [0.0, 0.0, 1.0],
                    dtype=np.float64,
                ),
            ]
        )

        if not np.all(np.isfinite(transform)):
            raise RuntimeError(
                "ORB-affine registratie leverde ongeldige waarden op."
            )

        determinant = float(np.linalg.det(transform[:2, :2]))

        if abs(determinant) < 1e-8:
            raise RuntimeError(
                "ORB-affine registratie is singulier of bijna singulier."
            )

        inlier_mask = mask.reshape(-1).astype(bool)
        source_inliers = source_points[inlier_mask]
        target_inliers = target_points[inlier_mask]
        source_hull_area = float(cv2.contourArea(cv2.convexHull(source_inliers)))
        target_hull_area = float(cv2.contourArea(cv2.convexHull(target_inliers)))
        image_area = float(source.shape[0] * source.shape[1])
        if (
            source_hull_area / image_area < 0.02
            or target_hull_area / image_area < 0.02
        ):
            raise RuntimeError(
                "ORB-overeenkomsten liggen te geconcentreerd in beeld. "
                "Kies een extra tussenframe met meer visuele overlap."
            )
        predicted = cv2.transform(
            source_points.reshape(-1, 1, 2),
            affine,
        ).reshape(-1, 2)
        errors = np.linalg.norm(predicted - target_points, axis=1)
        median_error = float(np.median(errors[inlier_mask]))

        return transform, FrameRegistrationDiagnostics(
            source_frame_index=source_frame_index,
            target_frame_index=target_frame_index,
            method="ORB affine",
            candidate_matches=len(good_matches),
            inlier_count=inlier_count,
            median_error_pixels=median_error,
        )

    @staticmethod
    def _estimate_ecc_transform(
        source: np.ndarray,
        target: np.ndarray,
        source_frame_index: int,
        target_frame_index: int,
    ) -> tuple[np.ndarray, FrameRegistrationDiagnostics]:
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        max_width = 960
        scale = min(1.0, max_width / source_gray.shape[1])

        if scale < 1.0:
            new_size = (
                int(source_gray.shape[1] * scale),
                int(source_gray.shape[0] * scale),
            )
            source_small = cv2.resize(source_gray, new_size)
            target_small = cv2.resize(target_gray, new_size)
        else:
            source_small = source_gray
            target_small = target_gray

        source_float = source_small.astype(np.float32) / 255.0
        target_float = target_small.astype(np.float32) / 255.0

        warp = np.eye(2, 3, dtype=np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            150,
            1e-6,
        )

        try:
            correlation, warp = cv2.findTransformECC(
                templateImage=target_float,
                inputImage=source_float,
                warpMatrix=warp,
                motionType=cv2.MOTION_AFFINE,
                criteria=criteria,
                inputMask=None,
                gaussFiltSize=5,
            )
        except cv2.error as exc:
            raise RuntimeError(
                "Frames overlappen onvoldoende. Kies tussenframes dichter bij elkaar."
            ) from exc

        if correlation < 0.70:
            raise RuntimeError(
                "Frame-registratie was onvoldoende betrouwbaar."
            )

        affine = np.vstack(
            [
                warp,
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
            ]
        )

        if scale < 1.0:
            scale_matrix = np.array(
                [
                    [scale, 0.0, 0.0],
                    [0.0, scale, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            inverse_scale = np.linalg.inv(scale_matrix)
            affine = (
                inverse_scale
                @ affine.astype(np.float64)
                @ scale_matrix
            )

        return affine.astype(np.float64), FrameRegistrationDiagnostics(
            source_frame_index=source_frame_index,
            target_frame_index=target_frame_index,
            method="ECC affine fallback",
            correlation=float(correlation),
        )

    @staticmethod
    def _draw_pitch_line(
        image: np.ndarray,
        matrix: np.ndarray,
        start: tuple[float, float],
        end: tuple[float, float],
        color: tuple[int, int, int] = (255, 180, 0),
        thickness: int = 1,
    ) -> None:
        pitch_points = np.asarray(
            [start, end],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        image_points = cv2.perspectiveTransform(
            pitch_points,
            matrix,
        ).reshape(-1, 2)

        point_a = tuple(np.round(image_points[0]).astype(int))
        point_b = tuple(np.round(image_points[1]).astype(int))

        cv2.line(
            image,
            point_a,
            point_b,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_goal_reference(
        image: np.ndarray,
        matrix: np.ndarray,
        posts: np.ndarray,
    ) -> None:
        image_points = cv2.perspectiveTransform(
            posts.reshape(-1, 1, 2).astype(np.float32),
            matrix,
        ).reshape(-1, 2)

        point_a = tuple(np.round(image_points[0]).astype(int))
        point_b = tuple(np.round(image_points[1]).astype(int))

        cv2.line(
            image,
            point_a,
            point_b,
            (0, 0, 255),
            5,
            cv2.LINE_AA,
        )
