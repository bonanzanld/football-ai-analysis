from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from football_ai.calibration.quality_report import (
    CalibrationQualityReport,
    calculate_quality_report,
)
from football_ai.pitch.panorama_builder import PanoramaBuilder

from football_ai.pitch.calibration_model import PitchCalibration, PitchProfile


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


class OpenCvCalibrationApp:
    WINDOW_NAME = "Football AI - veldkalibratie"
    SIDEBAR_WIDTH = 440
    WINDOW_WIDTH = 1640
    WINDOW_HEIGHT = 900

    def __init__(
        self,
        video_path: Path,
        landmarks: dict[int, LandmarkDefinition],
    ) -> None:
        self.video_path = video_path
        self.landmarks = landmarks

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

        self.annotation_frame_index = 0
        self.current_landmark_key: int | None = 1

        self.display_image_rect = (0, 0, 0, 0)
        self.display_scale = 1.0
        self.cancelled = False
        self.finished = False
        self.status_message = "Kies 4-8 overlappende frames."

    def run(self) -> tuple[list[SelectedFrame], list[LandmarkObservation]]:
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

        return self.selected_frames, self.observations

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
            if len(self.selected_frames) < 2:
                self.status_message = "Selecteer minimaal twee overlappende frames."
                return
            self.mode = "annotate"
            self.annotation_frame_index = 0
            self.current_landmark_key = 1
            self.current_frame = self.selected_frames[0].frame.copy()
            self.status_message = "Markeer alleen punten die echt zichtbaar zijn."

    def _handle_annotate_key(self, key: int) -> None:
        if ord("1") <= key <= ord("8"):
            self.current_landmark_key = int(chr(key))
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
            unique_landmarks = {item.landmark_key for item in self.observations}

            if len(unique_landmarks) < 4:
                self.status_message = (
                    "Markeer minimaal vier verschillende veldpunten."
                )
                return

            missing_keys = [
                key
                for key in self.landmarks
                if key not in unique_landmarks
            ]

            if missing_keys:
                missing_names = ", ".join(
                    self.landmarks[key].name
                    for key in missing_keys
                )

                if not self.status_message.startswith("WAARSCHUWING:"):
                    self.status_message = (
                        "WAARSCHUWING: ontbrekend: "
                        f"{missing_names}. Druk nogmaals Enter om toch door te gaan."
                    )
                    return

            self.finished = True

    def _mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        _flags: int,
        _userdata: object,
    ) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.mode == "select":
            return

        if self.current_landmark_key is None or self.current_frame is None:
            return

        image_x, image_y, image_w, image_h = self.display_image_rect

        if not (
            image_x <= x < image_x + image_w
            and image_y <= y < image_y + image_h
        ):
            return

        original_x = (x - image_x) / self.display_scale
        original_y = (y - image_y) / self.display_scale

        frame_height, frame_width = self.current_frame.shape[:2]
        if not (
            0 <= original_x < frame_width
            and 0 <= original_y < frame_height
        ):
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
        self._select_next_landmark()

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
        self.current_landmark_key = 1

    def _previous_annotation_frame(self) -> None:
        if self.annotation_frame_index <= 0:
            return

        self.annotation_frame_index -= 1
        self.current_frame = self.selected_frames[
            self.annotation_frame_index
        ].frame.copy()
        self.current_landmark_key = 1

    def _undo_last_observation(self) -> None:
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
        self.current_landmark_key = 1
        self.status_message = "Alle punten uit dit frame verwijderd."

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
            else "2. VELDPUNTEN"
        )
        self._text(canvas, title, (24, y), 0.65, (0, 220, 255), 2)
        y += 35

        if self.mode == "select":
            self._draw_select_sidebar(canvas, y)
        else:
            self._draw_annotate_sidebar(canvas, y)

        self._draw_wrapped_text(
            canvas,
            self.status_message,
            x=24,
            y=self.WINDOW_HEIGHT - 75,
            max_width=self.SIDEBAR_WIDTH - 48,
            line_height=22,
            color=(230, 230, 230),
        )

    def _draw_select_sidebar(self, canvas: np.ndarray, y: int) -> None:
        self._draw_wrapped_text(
            canvas,
            "Kies 4-8 frames met duidelijke overlap.",
            24,
            y,
            310,
            22,
            (230, 230, 230),
        )
        y += 55

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
        selected = self.selected_frames[self.annotation_frame_index]

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
            for key in self.landmarks
        }

        completed_landmarks = sum(
            1 for count in total_counts.values() if count > 0
        )
        total_observations = len(self.observations)
        completion_ratio = completed_landmarks / max(1, len(self.landmarks))

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

        self._text(
            canvas,
            (
                f"TOTAAL: {completed_landmarks}/{len(self.landmarks)} punten  |  "
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

        indicator_spacing = 42
        indicator_start_x = 42
        indicator_y = y + 10

        for index, key in enumerate(self.landmarks):
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

        for key, landmark in self.landmarks.items():
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

        y += 4
        self._text(
            canvas,
            "TOTAALOVERZICHT",
            (24, y),
            0.48,
            (255, 255, 255),
            2,
        )
        y += 22

        for key, landmark in self.landmarks.items():
            count = total_counts[key]

            if count == 0:
                marker = "[ ]"
                color = (120, 120, 120)
            elif count == 1:
                marker = "[1]"
                color = (0, 220, 255)
            else:
                marker = "[x]"
                color = (70, 200, 70)

            self._text(
                canvas,
                f"{marker} {key}. {landmark.name:<30} {count}x",
                (24, y),
                0.38,
                color,
                1,
            )
            y += 18

        y += 10

        controls_y = self.WINDOW_HEIGHT - 112
        controls = [
            "1-8 punt | klik plaatsen | K overslaan",
            "U undo | R frame leeg | P/N frame",
            "Enter berekenen | Esc stoppen",
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



    def _draw_pitch_diagram(self, canvas: np.ndarray, top: int) -> None:
        left = 78
        right = self.SIDEBAR_WIDTH - 78
        pitch_top = top + 24
        pitch_bottom = top + 130

        self._text(
            canvas,
            "ACHTER",
            (self.SIDEBAR_WIDTH // 2 - 36, top + 14),
            0.42,
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
        cv2.line(
            canvas,
            ((left + right) // 2, pitch_top),
            ((left + right) // 2, pitch_bottom),
            (100, 100, 100),
            1,
        )

        self._text(
            canvas,
            "CAMERA / VOOR",
            (self.SIDEBAR_WIDTH // 2 - 66, pitch_bottom + 24),
            0.40,
            (230, 230, 230),
            1,
        )

        positions = {
            1: (left, pitch_top),
            2: (right, pitch_top),
            3: (left, pitch_bottom),
            4: (right, pitch_bottom),
            5: (left, pitch_top + 35),
            6: (left, pitch_top + 72),
            7: (right, pitch_top + 72),
            8: (right, pitch_top + 35),
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

        frame_height, frame_width = frame.shape[:2]
        scale = min(
            available_width / frame_width,
            available_height / frame_height,
        )

        render_width = int(frame_width * scale)
        render_height = int(frame_height * scale)

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
        self.frame_transforms: list[np.ndarray] = []
        self.quality_report: CalibrationQualityReport | None = None
        self.landmarks = self._create_landmark_definitions()

    def calibrate_video(self, video_path: Path) -> PitchCalibration:
        if not video_path.exists():
            raise FileNotFoundError(f"Video niet gevonden: {video_path}")

        app = OpenCvCalibrationApp(
            video_path=video_path,
            landmarks=self.landmarks,
        )
        self.selected_frames, self.observations = app.run()
        transforms = self._calculate_frame_transforms()
        self.frame_transforms = transforms

        # Analyseer eerst de geometrie van alle frame-transformaties.
        panorama_report = PanoramaBuilder(
                selected_frames=self.selected_frames,
                frame_transforms=transforms,
        ).analyze()

        print()
        panorama_report.print_summary()
        print()

        image_points: list[tuple[float, float]] = []
        pitch_points: list[tuple[float, float]] = []

        for observation in self.observations:
            source_point = np.array(
                [[observation.image_point]],
                dtype=np.float32,
            )
            transformed_point = cv2.perspectiveTransform(
                source_point,
                transforms[observation.frame_index],
            )[0, 0]

            image_points.append(
                (
                    float(transformed_point[0]),
                    float(transformed_point[1]),
                )
            )
            pitch_points.append(
                self.landmarks[observation.landmark_key].pitch_point
            )

        image_array = np.asarray(image_points, dtype=np.float32)
        pitch_array = np.asarray(pitch_points, dtype=np.float32)

        if len(np.unique(pitch_array, axis=0)) < 4:
            raise RuntimeError(
                "Er zijn minimaal vier verschillende veldpunten nodig."
            )

        image_to_pitch, mask = cv2.findHomography(
            image_array,
            pitch_array,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
        )

        if image_to_pitch is None:
            raise RuntimeError(
                "De gezamenlijke homography kon niet worden berekend."
            )

        pitch_to_image = np.linalg.inv(image_to_pitch)
        self.quality_report = calculate_quality_report(
            image_points=image_array,
            pitch_points=pitch_array,
            image_to_pitch_matrix=image_to_pitch,
            inlier_mask=mask,
        )
        image_corners = cv2.perspectiveTransform(
            self.profile.world_corners.reshape(-1, 1, 2).astype(np.float32),
            pitch_to_image,
        ).reshape(-1, 2)

        reference = self.selected_frames[0]
        frame_height, frame_width = reference.frame.shape[:2]

        print(self.quality_report.format_terminal_report())

        return PitchCalibration(
            profile=self.profile,
            image_corners=image_corners,
            image_to_pitch_matrix=image_to_pitch,
            pitch_to_image_matrix=pitch_to_image,
            source_video=str(video_path),
            source_frame_number=reference.frame_number,
            source_time_seconds=reference.time_seconds,
            frame_width=frame_width,
            frame_height=frame_height,
            quality=self.quality_report,
        )

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

        pitch_to_canvas = (
            reference_to_canvas
            @ calibration.pitch_to_image_matrix
        )
        pitch_to_canvas /= pitch_to_canvas[2, 2]

        # Volledige veldomtrek.
        pitch_corners = cv2.perspectiveTransform(
            self.profile.world_corners.reshape(-1, 1, 2).astype(np.float32),
            pitch_to_canvas,
        ).reshape(-1, 2)

        polygon = np.round(pitch_corners).astype(np.int32)
        cv2.polylines(
            panorama,
            [polygon],
            isClosed=True,
            color=(0, 255, 255),
            thickness=4,
            lineType=cv2.LINE_AA,
        )

        # Raster over de veldbreedte.
        for x_value in np.arange(
            0.0,
            self.profile.width_m + 0.001,
            grid_interval_m,
            dtype=np.float32,
        ):
            self._draw_pitch_line(
                panorama,
                pitch_to_canvas,
                (float(x_value), 0.0),
                (float(x_value), self.profile.length_m),
                color=(255, 180, 0),
                thickness=1,
            )

        # Raster over de veldlengte.
        for y_value in np.arange(
            0.0,
            self.profile.length_m + 0.001,
            grid_interval_m,
            dtype=np.float32,
        ):
            self._draw_pitch_line(
                panorama,
                pitch_to_canvas,
                (0.0, float(y_value)),
                (self.profile.width_m, float(y_value)),
                color=(255, 180, 0),
                thickness=1,
            )

        # Middenlijn altijd exact op halve veldlengte tekenen.
        middle_y = self.profile.length_m / 2.0
        self._draw_pitch_line(
            panorama,
            pitch_to_canvas,
            (0.0, middle_y),
            (self.profile.width_m, middle_y),
            color=(0, 255, 255),
            thickness=5,
        )

        self._draw_goal_reference(
            panorama,
            pitch_to_canvas,
            self.profile.goal_a_posts,
        )
        self._draw_goal_reference(
            panorama,
            pitch_to_canvas,
            self.profile.goal_b_posts,
        )

        errors: list[float] = []

        for observation in self.observations:
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

            error_pixels = float(
                np.linalg.norm(clicked_canvas - predicted_canvas)
            )
            errors.append(error_pixels)

            clicked_point = tuple(np.round(clicked_canvas).astype(int))
            predicted_point = tuple(np.round(predicted_canvas).astype(int))

            error_color = (
                (60, 210, 60)
                if error_pixels <= 12.0
                else (0, 0, 255)
            )

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

        self._draw_validation_legend(panorama, errors)
        return panorama

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
    def _draw_validation_legend(
        image: np.ndarray,
        errors: list[float],
    ) -> None:
        panel_x = 18
        panel_y = 18
        panel_w = 500
        panel_h = 142

        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)

        cv2.putText(
            image,
            "KALIBRATIEPANORAMA",
            (panel_x + 14, panel_y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "Geel = klik | magenta = voorspeld | cyaan = veldrand/middenlijn",
            (panel_x + 14, panel_y + 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )

        if errors:
            mean_error = float(np.mean(errors))
            max_error = float(np.max(errors))
            status = "GOED" if max_error <= 12.0 else "CONTROLEREN"
            status_color = (
                (60, 210, 60)
                if status == "GOED"
                else (0, 0, 255)
            )
            summary = (
                f"Gemiddeld {mean_error:.1f}px | "
                f"maximaal {max_error:.1f}px | {status}"
            )
        else:
            status_color = (0, 165, 255)
            summary = "Geen waarnemingen beschikbaar voor validatie."

        cv2.putText(
            image,
            summary,
            (panel_x + 14, panel_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            status_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "Controleer aansluiting frames, beide doelen, zijlijnen en middenlijn.",
            (panel_x + 14, panel_y + 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )

    def _create_landmark_definitions(self) -> dict[int, LandmarkDefinition]:
        width = float(self.profile.width_m)
        length = float(self.profile.length_m)
        goal_width = float(self.profile.goal_width_m)

        left_goal_post_left = (width - goal_width) / 2.0
        left_goal_post_right = (width + goal_width) / 2.0

        return {
            1: LandmarkDefinition(1, "Hoek linksachter", (0.0, 0.0)),
            2: LandmarkDefinition(2, "Hoek rechtsachter", (0.0, length)),
            3: LandmarkDefinition(3, "Hoek linksvoor", (width, 0.0)),
            4: LandmarkDefinition(4, "Hoek rechtsvoor", (width, length)),
            5: LandmarkDefinition(
                5,
                "Linkerdoel - doelpaal links",
                (left_goal_post_left, 0.0),
            ),
            6: LandmarkDefinition(
                6,
                "Linkerdoel - doelpaal rechts",
                (left_goal_post_right, 0.0),
            ),
            7: LandmarkDefinition(
                7,
                "Rechterdoel - doelpaal links",
                (left_goal_post_right, length),
            ),
            8: LandmarkDefinition(
                8,
                "Rechterdoel - doelpaal rechts",
                (left_goal_post_left, length),
            ),
        }

    def _calculate_frame_transforms(self) -> list[np.ndarray]:
        transforms: list[np.ndarray] = [
            np.eye(3, dtype=np.float64)
        ]

        for frame_index in range(1, len(self.selected_frames)):
            previous_frame = self.selected_frames[
                frame_index - 1
            ].frame
            current_frame = self.selected_frames[
                frame_index
            ].frame

            current_to_previous = self._estimate_image_transform(
                source=current_frame,
                target=previous_frame,
            )

            current_to_reference = (
                transforms[frame_index - 1]
                @ current_to_previous
            )
            current_to_reference /= current_to_reference[2, 2]
            transforms.append(current_to_reference)

            print(
                f"Frame {frame_index + 1} gekoppeld aan referentieframe."
            )

        return transforms

    def _estimate_image_transform(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        try:
            return self._estimate_orb_transform(source, target)
        except RuntimeError:
            return self._estimate_ecc_transform(source, target)

    @staticmethod
    def _estimate_orb_transform(
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
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

        return transform

    @staticmethod
    def _estimate_ecc_transform(
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
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

        return affine.astype(np.float64)

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
