from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from football_ai.calibration.manual_midfield_line import (
    ManualMidfieldLine,
    load_manual_midfield_line,
    save_manual_midfield_line,
)
from football_ai.calibration.manual_perspective_reference import load_manual_perspective_reference


class MidfieldLineCollector:
    WINDOW = "Football AI - 11v11 middenlijn"
    CANVAS = (1700, 920)
    PANEL_WIDTH = 500
    POINT_COUNT = 7

    def __init__(
        self,
        video: Path,
        initial_time_seconds: float,
        existing: ManualMidfieldLine | None = None,
    ) -> None:
        self.video = video
        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Video kon niet worden geopend: {video}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.time_seconds = float(initial_time_seconds)
        self.frame = self._read()
        self.points: list[tuple[float, float]] = []
        self.reference_frame_number: int | None = None
        self.reference_time_seconds: float | None = None
        self.front_frame_number: int | None = None
        self.front_not_visible = False
        self.zoom = 1.0
        self.center: tuple[float, float] | None = None
        self.mapping = (0.0, 0.0, 1.0, 0.0, 0.0)
        self.status = "Zoek een beeld waarop de witte 11v11-middenlijn goed zichtbaar is."
        self.saved: ManualMidfieldLine | None = None
        if existing is not None:
            if existing.front_sideline_point is not None:
                raise ValueError("De bestaande middenlijn heeft al een voorste zijlijnpositie.")
            if existing.rear_sideline_point is None:
                raise ValueError("De bestaande middenlijn mist de achterste zijlijnpositie.")
            self.points = [*existing.points, existing.rear_sideline_point]
            self.reference_frame_number = existing.frame_number
            self.reference_time_seconds = existing.time_seconds
            self.front_frame_number = None
            self.status = (
                "Bestaande richting en ACHTER geladen. Zoek met , en . een beeld met een "
                "betrouwbaar hoedje of punt OP de VOORSTE 8v8-zijlijn. Deze lijn loopt parallel "
                "aan de witte 11v11-middenlijn; zoek dus GEEN kruispunt."
            )

    def run(self) -> ManualMidfieldLine:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, *self.CANVAS)
        cv2.setMouseCallback(self.WINDOW, self._mouse)
        try:
            while self.saved is None:
                cv2.imshow(self.WINDOW, self._render())
                key = cv2.waitKeyEx(30)
                if key in (27, ord("q"), ord("Q")):
                    raise KeyboardInterrupt("Middenlijninvoer afgebroken.")
                if key in (ord("u"), ord("U"), 8, 127):
                    self._undo()
                elif key in (ord("r"), ord("R")):
                    self.points.clear()
                    self.reference_frame_number = None
                    self.reference_time_seconds = None
                    self.front_frame_number = None
                    self.front_not_visible = False
                    self.status = "Alles gewist. Begin opnieuw met vijf punten op de witte referentielijn."
                elif key in (ord("+"), ord("=")):
                    self._zoom(1.25)
                elif key in (ord("-"), ord("_")):
                    self._zoom(0.8)
                elif key == ord("0"):
                    self.zoom, self.center = 1.0, None
                elif key in (2424832, 65361, 63234, 81, ord("a"), ord("A")):
                    self._pan(-0.16, 0.0)
                elif key in (2555904, 65363, 63235, 83, ord("d"), ord("D")):
                    self._pan(0.16, 0.0)
                elif key in (2490368, 65362, 63232, 82, ord("w"), ord("W")):
                    self._pan(0.0, -0.16)
                elif key in (2621440, 65364, 63233, 84, ord("s"), ord("S")):
                    self._pan(0.0, 0.16)
                elif key in (ord(","), ord("<")):
                    self._shift_time(-1.0)
                elif key in (ord("."), ord(">")):
                    self._shift_time(1.0)
                elif key in (ord("n"), ord("N")) and len(self.points) == 6:
                    self.front_not_visible = True
                    self.front_frame_number = None
                    self.status = "VOOR niet zichtbaar. Druk Enter om zonder voorste zijlijn op te slaan."
                elif key in (10, 13):
                    self._finish()
        finally:
            self.capture.release()
            cv2.destroyWindow(self.WINDOW)
        return self.saved

    def _read(self) -> np.ndarray:
        frame_number = int(round(self.time_seconds * self.fps))
        frame_number = int(np.clip(frame_number, 0, max(self.frame_count - 1, 0)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"Frame rond {self.time_seconds:.1f}s kon niet worden gelezen.")
        self.time_seconds = frame_number / self.fps
        return frame

    def _shift_time(self, seconds: float) -> None:
        if self.points and len(self.points) < 6:
            self.status = "Leg eerst richting en ACHTER vast voordat je van videomoment wisselt."
            return
        if len(self.points) >= 7 or self.front_not_visible:
            self.status = "VOOR is al opgeslagen. Gebruik U of R om dit te wijzigen."
            return
        duration = max((self.frame_count - 1) / self.fps, 0.0)
        self.time_seconds = float(np.clip(self.time_seconds + seconds, 0.0, duration))
        self.frame = self._read()
        self.zoom, self.center = 1.0, None

    def _zoom(self, factor: float) -> None:
        height, width = self.frame.shape[:2]
        self.center = self.center or (width / 2.0, height / 2.0)
        self.zoom = float(np.clip(self.zoom * factor, 1.0, 10.0))
        self._clamp_center()

    def _pan(self, dx: float, dy: float) -> None:
        if self.zoom <= 1.0:
            self.status = "Zoom eerst in; bij volledig beeld is bewegen niet nodig."
            return
        height, width = self.frame.shape[:2]
        crop_w, crop_h = width / self.zoom, height / self.zoom
        cx, cy = self.center or (width / 2.0, height / 2.0)
        self.center = (cx + dx * crop_w, cy + dy * crop_h)
        self._clamp_center()

    def _clamp_center(self) -> None:
        if self.center is None:
            return
        height, width = self.frame.shape[:2]
        crop_w, crop_h = width / self.zoom, height / self.zoom
        self.center = (
            float(np.clip(self.center[0], crop_w / 2.0, width - crop_w / 2.0)),
            float(np.clip(self.center[1], crop_h / 2.0, height - crop_h / 2.0)),
        )

    def _crop(self):
        height, width = self.frame.shape[:2]
        crop_w, crop_h = width / self.zoom, height / self.zoom
        cx, cy = self.center or (width / 2.0, height / 2.0)
        x0 = float(np.clip(cx - crop_w / 2.0, 0.0, width - crop_w))
        y0 = float(np.clip(cy - crop_h / 2.0, 0.0, height - crop_h))
        return self.frame[int(y0):int(y0 + crop_h), int(x0):int(x0 + crop_w)], (x0, y0, crop_w, crop_h)

    def _mouse(self, event: int, x: int, y: int, flags: int, _data) -> None:
        if event == cv2.EVENT_MOUSEWHEEL:
            x0, y0, scale, ox, oy = self.mapping
            if x >= ox and y >= oy:
                self.center = (x0 + (x - ox) / scale, y0 + (y - oy) / scale)
            self._zoom(1.25 if flags > 0 else 0.8)
            return
        if (
            event != cv2.EVENT_LBUTTONDOWN
            or len(self.points) >= self.POINT_COUNT
            or self.front_not_visible
        ):
            return
        x0, y0, scale, ox, oy = self.mapping
        point = (x0 + (x - ox) / scale, y0 + (y - oy) / scale)
        height, width = self.frame.shape[:2]
        if not (0.0 <= point[0] < width and 0.0 <= point[1] < height):
            self.status = "Klik binnen het videobeeld."
            return
        self.points.append(point)
        current_frame = int(round(self.time_seconds * self.fps))
        if len(self.points) == 1:
            self.reference_frame_number = current_frame
            self.reference_time_seconds = self.time_seconds
        if len(self.points) == 7:
            self.front_frame_number = current_frame
        if len(self.points) == self.POINT_COUNT:
            self.status = "Beide 8v8-zijlijnposities compleet. Druk Enter om op te slaan."
        elif len(self.points) == 5:
            self.status = "Richting compleet. Klik nu 1 punt op de ACHTERSTE 8v8-zijlijn."
        elif len(self.points) == 6:
            self.status = (
                "ACHTER opgeslagen. Klik VOOR als zichtbaar, of blader met , en . "
                "naar een ander moment."
            )
        else:
            self.status = f"Richtingspunt {len(self.points)}/5 opgeslagen. Blijf op dezelfde witte lijn."

    def _undo(self) -> None:
        if self.points:
            if len(self.points) == 7:
                self.front_frame_number = None
            self.front_not_visible = False
            self.points.pop()
            self.status = "Laatste punt verwijderd."
        else:
            self.status = "Er is nog geen punt om te verwijderen."

    def _observation(self) -> ManualMidfieldLine | None:
        if len(self.points) < 5:
            return None
        return ManualMidfieldLine.fit(
            self.video.name,
            self.reference_frame_number or int(round(self.time_seconds * self.fps)),
            self.reference_time_seconds if self.reference_time_seconds is not None else self.time_seconds,
            tuple(self.points[:5]),
            None if len(self.points) < 6 else self.points[5],
            None if len(self.points) < 7 else self.points[6],
            self.reference_frame_number,
            self.front_frame_number,
        )

    def _finish(self) -> None:
        if len(self.points) != self.POINT_COUNT and not (
            len(self.points) == 6 and self.front_not_visible
        ):
            self.status = f"Nog {self.POINT_COUNT - len(self.points)} klik(ken) nodig voordat Enter werkt."
            return
        observation = self._observation()
        if observation is None:
            self.status = "De ingevoerde referentielijn kon niet worden berekend."
            return
        self.saved = observation

    def _render(self) -> np.ndarray:
        canvas = np.full((self.CANVAS[1], self.CANVAS[0], 3), 24, np.uint8)
        crop, (x0, y0, crop_w, crop_h) = self._crop()
        area_w = self.CANVAS[0] - self.PANEL_WIDTH
        scale = min(area_w / crop_w, self.CANVAS[1] / crop_h)
        shown = cv2.resize(crop, (round(crop_w * scale), round(crop_h * scale)))
        ox = self.PANEL_WIDTH + (area_w - shown.shape[1]) // 2
        oy = (self.CANVAS[1] - shown.shape[0]) // 2
        canvas[oy:oy + shown.shape[0], ox:ox + shown.shape[1]] = shown
        self.mapping = (x0, y0, scale, float(ox), float(oy))

        observation = self._observation()
        current_frame = int(round(self.time_seconds * self.fps))
        showing_reference = self.reference_frame_number in (None, current_frame)
        if observation is not None and showing_reference:
            first, second = observation.endpoints(self.frame.shape[1], self.frame.shape[0])
            cv2.line(
                canvas,
                self._display(first, x0, y0, scale, ox, oy),
                self._display(second, x0, y0, scale, ox, oy),
                (255, 255, 0),
                4,
                cv2.LINE_AA,
            )
        visible_points = self.points if showing_reference else self.points[6:]
        start_index = 1 if showing_reference else 7
        for index, point in enumerate(visible_points, start=start_index):
            display = self._display(point, x0, y0, scale, ox, oy)
            color = (255, 0, 255) if index <= 5 else ((0, 165, 255) if index == 6 else (0, 255, 255))
            label = str(index) if index <= 5 else ("ACHTER" if index == 6 else "VOOR")
            cv2.circle(canvas, display, 7, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, label, (display[0] + 8, display[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        self._draw_panel(canvas)
        return canvas

    def _draw_panel(self, canvas: np.ndarray) -> None:
        lines = (
            "11v11-MIDDENLIJN VASTLEGGEN",
            "",
            "WAT MOET JE AANKLIKKEN?",
            "De WITTE MIDDENLIJN van het grote",
            "11-tegen-11-veld.",
            "",
            "Dit is de rechte lijn door de",
            "middencirkel, dwars over het grote veld.",
            "Het is NIET een gele 8v8-zijlijn.",
            "",
            "STAP 1 - KLIK 5 PUNTEN",
            "- allemaal op DEZELFDE kalklijn",
            "- verspreid over de zichtbare lengte",
            "- ongeveer midden in de witte kalk",
            "- volgorde maakt niet uit",
            "",
            "STAP 2 - KLIK 1 PUNT ACHTER",
            "Op de 8v8-zijlijn het verst van de camera.",
            "Een hoedje of zichtbaar lijnpunt is voldoende.",
            "",
            "STAP 3 - KLIK 1 PUNT VOOR",
            "Op de 8v8-zijlijn het dichtst bij de camera.",
            "Een hoedje of zichtbaar lijnpunt is voldoende.",
            "De 8v8-zijlijn loopt PARALLEL aan de",
            "witte 11v11-middenlijn: GEEN kruispunt.",
            "NIET ZICHTBAAR? Gebruik , en . om",
            "naar een ander videomoment te bladeren.",
            "Ook nergens zichtbaar? Druk N om deze",
            "observatie eerlijk leeg te laten.",
            "De eerdere klikken blijven bewaard.",
            "",
            f"Videomoment: {self.time_seconds:.1f}s",
            f"Zoom: {self.zoom:.1f}x | klikken: {len(self.points)}/7",
            "",
            "Muiswiel of +/-: zoomen",
            "Pijltjes of W/A/S/D: beeld bewegen",
            ", en . : 1 seconde eerder/later",
            "0: volledig beeld | U: laatste punt | N: niet zichtbaar",
            "R: alle punten wissen",
            "Enter: opslaan | Esc: afbreken",
            "",
            self.status,
        )
        for index, line in enumerate(lines):
            color = (0, 230, 255) if index in (0, 2, 10, 17, 21, 34) else (235, 235, 235)
            cv2.putText(canvas, line, (18, 34 + index * 31), cv2.FONT_HERSHEY_SIMPLEX, 0.49, color, 2 if index in (0, 2, 10) else 1, cv2.LINE_AA)

    @staticmethod
    def _display(point, x0, y0, scale, ox, oy):
        return int(round(ox + (point[0] - x0) * scale)), int(round(oy + (point[1] - y0) * scale))


def _initial_time(output: Path, fallback: float | None) -> float:
    if fallback is not None:
        return fallback
    if output.exists():
        reference = load_manual_perspective_reference(output)
        return next(view.time_seconds for view in reference.views if view.label == "center")
    return 0.0


def _draw_preview(frame: np.ndarray, observation: ManualMidfieldLine) -> np.ndarray:
    preview = frame.copy()
    first, second = observation.endpoints(frame.shape[1], frame.shape[0])
    cv2.line(preview, first, second, (255, 255, 0), 6, cv2.LINE_AA)
    preview_points = list(observation.points)
    if observation.rear_sideline_point is not None:
        preview_points.append(observation.rear_sideline_point)
    if (
        observation.front_sideline_point is not None
        and observation.front_sideline_frame_number in (None, observation.frame_number)
    ):
        preview_points.append(observation.front_sideline_point)
    for index, point in enumerate(preview_points, start=1):
        display = tuple(np.rint(point).astype(int))
        color = (255, 0, 255) if index <= 5 else ((0, 165, 255) if index == 6 else (0, 255, 255))
        label = str(index) if index <= 5 else ("ACHTER" if index == 6 else "VOOR")
        cv2.circle(preview, display, 9, color, -1, cv2.LINE_AA)
        cv2.putText(preview, label, (display[0] + 10, display[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.rectangle(preview, (0, 0), (frame.shape[1], 62), (12, 12, 12), -1)
    cv2.putText(preview, "11v11-MIDDENLIJN | REFERENTIE VOOR 8v8-ZIJLIJNEN", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 230, 255), 2, cv2.LINE_AA)
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Leg de 11v11-middenlijn met vijf punten vast.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--time", type=float, help="Optioneel startmoment in seconden.")
    parser.add_argument(
        "--resume-front",
        action="store_true",
        help="Behoud de vijf lijnpunten en ACHTER; vraag alleen het ontbrekende VOOR-punt.",
    )
    args = parser.parse_args()

    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    perspective_path = output_dir / f"{prefix}_manual_perspective_reference.json"
    output = output_dir / f"{prefix}_manual_midfield_line.json"
    existing = load_manual_midfield_line(output) if args.resume_front else None
    initial_time = existing.time_seconds if existing is not None else _initial_time(perspective_path, args.time)
    observation = MidfieldLineCollector(video, initial_time, existing=existing).run()

    save_manual_midfield_line(observation, output)
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, observation.frame_number)
    ok, frame = capture.read()
    capture.release()
    preview_path = output_dir / f"{prefix}_manual_midfield_line.jpg"
    if ok:
        cv2.imwrite(str(preview_path), _draw_preview(frame, observation))

    print(f"11v11-middenlijn opgeslagen: {output}")
    print(f"Frame {observation.frame_number} ({observation.time_seconds:.1f}s)")
    print(f"Lijnfit: RMS {observation.rms_error_px:.2f}px | max {observation.maximum_error_px:.2f}px")
    if ok:
        print(f"QA-preview: {preview_path}")


if __name__ == "__main__":
    main()
