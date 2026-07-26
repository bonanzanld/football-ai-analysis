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

from football_ai.calibration.manual_midfield_line import load_manual_midfield_line
from football_ai.calibration.manual_parallel_lines import (
    ManualParallelLine,
    ManualParallelLineReference,
    save_manual_parallel_lines,
)
from football_ai.calibration.manual_perspective_reference import load_manual_perspective_reference


class ExtraParallelLineCollector:
    WINDOW = "Football AI - 11v11 parallelle lijnen"
    CANVAS = (1700, 920)
    PANEL_WIDTH = 520
    POINT_COUNT = 5
    TARGETS = (
        (
            "goal_area_5m",
            "WITTE 5-METERLIJN",
            "De lange voorste lijn van het kleine doelgebied van het grote 11v11-veld.",
        ),
        (
            "penalty_area_16m",
            "WITTE 16-METERLIJN",
            "De lange voorste lijn van het strafschopgebied van het grote 11v11-veld.",
        ),
    )

    def __init__(self, video: Path, initial_time_seconds: float) -> None:
        self.video = video
        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened():
            raise FileNotFoundError(video)
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.time_seconds = initial_time_seconds
        self.frame = self._read()
        self.target_index = 0
        self.points: list[tuple[float, float]] = []
        self.results: list[ManualParallelLine] = []
        self.zoom = 1.0
        self.center: tuple[float, float] | None = None
        self.mapping = (0.0, 0.0, 1.0, 0.0, 0.0)
        self.status = "Kies zo nodig een ander videomoment en klik daarna vijf punten."

    def run(self) -> tuple[ManualParallelLine, ...]:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, *self.CANVAS)
        cv2.setMouseCallback(self.WINDOW, self._mouse)
        try:
            while self.target_index < len(self.TARGETS):
                cv2.imshow(self.WINDOW, self._render())
                key = cv2.waitKeyEx(30)
                if key in (27, ord("q"), ord("Q")):
                    raise KeyboardInterrupt("Invoer van 11v11-lijnen afgebroken.")
                if key in (ord("u"), ord("U"), 8, 127):
                    self._undo()
                elif key in (ord("r"), ord("R")):
                    self.points.clear()
                    self.status = "Punten gewist. Klik opnieuw vijf punten op dezelfde lijn."
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
                elif key in (10, 13):
                    self._finish_target()
        finally:
            self.capture.release()
            cv2.destroyWindow(self.WINDOW)
        return tuple(self.results)

    def _read(self):
        number = int(np.clip(round(self.time_seconds * self.fps), 0, self.frame_count - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, number)
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"Frame rond {self.time_seconds:.1f}s kon niet worden gelezen.")
        self.time_seconds = number / self.fps
        return frame

    def _shift_time(self, seconds):
        if self.points:
            self.status = "Druk eerst R om de klikken te wissen voordat je van frame wisselt."
            return
        duration = max((self.frame_count - 1) / self.fps, 0.0)
        self.time_seconds = float(np.clip(self.time_seconds + seconds, 0.0, duration))
        self.frame = self._read()
        self.zoom, self.center = 1.0, None

    def _zoom(self, factor):
        height, width = self.frame.shape[:2]
        self.center = self.center or (width / 2.0, height / 2.0)
        self.zoom = float(np.clip(self.zoom * factor, 1.0, 10.0))
        self._clamp_center()

    def _pan(self, dx, dy):
        if self.zoom <= 1.0:
            self.status = "Zoom eerst in; pannen is bij volledig beeld niet nodig."
            return
        height, width = self.frame.shape[:2]
        crop_w, crop_h = width / self.zoom, height / self.zoom
        cx, cy = self.center or (width / 2.0, height / 2.0)
        self.center = (cx + dx * crop_w, cy + dy * crop_h)
        self._clamp_center()

    def _clamp_center(self):
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

    def _mouse(self, event, x, y, flags, _data):
        if event == cv2.EVENT_MOUSEWHEEL:
            x0, y0, scale, ox, oy = self.mapping
            if x >= ox and y >= oy:
                self.center = (x0 + (x - ox) / scale, y0 + (y - oy) / scale)
            self._zoom(1.25 if flags > 0 else 0.8)
            return
        if event != cv2.EVENT_LBUTTONDOWN or len(self.points) >= self.POINT_COUNT:
            return
        x0, y0, scale, ox, oy = self.mapping
        point = (x0 + (x - ox) / scale, y0 + (y - oy) / scale)
        height, width = self.frame.shape[:2]
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            self.status = "Klik binnen het videobeeld."
            return
        self.points.append(point)
        self.status = (
            "Vijf punten compleet. Controleer de cyaan lijn en druk Enter."
            if len(self.points) == 5
            else f"Punt {len(self.points)}/5 opgeslagen; blijf op dezelfde witte lijn."
        )

    def _current(self):
        if len(self.points) != 5:
            return None
        return ManualParallelLine.fit(
            self.TARGETS[self.target_index][0],
            int(round(self.time_seconds * self.fps)),
            self.time_seconds,
            tuple(self.points),
        )

    def _finish_target(self):
        current = self._current()
        if current is None:
            self.status = f"Nog {5 - len(self.points)} punt(en) nodig voordat Enter werkt."
            return
        self.results.append(current)
        self.target_index += 1
        self.points.clear()
        self.zoom, self.center = 1.0, None
        if self.target_index < len(self.TARGETS):
            self.status = "5m-lijn opgeslagen. Zoek nu een goed beeld van de 16m-lijn."

    def _undo(self):
        if self.points:
            self.points.pop()
            self.status = "Laatste punt verwijderd."
        else:
            self.status = "Er is nog geen punt om te verwijderen."

    def _render(self):
        canvas = np.full((self.CANVAS[1], self.CANVAS[0], 3), 24, np.uint8)
        crop, (x0, y0, crop_w, crop_h) = self._crop()
        area_w = self.CANVAS[0] - self.PANEL_WIDTH
        scale = min(area_w / crop_w, self.CANVAS[1] / crop_h)
        shown = cv2.resize(crop, (round(crop_w * scale), round(crop_h * scale)))
        ox = self.PANEL_WIDTH + (area_w - shown.shape[1]) // 2
        oy = (self.CANVAS[1] - shown.shape[0]) // 2
        canvas[oy:oy + shown.shape[0], ox:ox + shown.shape[1]] = shown
        self.mapping = (x0, y0, scale, float(ox), float(oy))
        current = self._current()
        if current is not None:
            line = np.asarray(current.equation)
            endpoints = _line_endpoints(line, self.frame.shape[1], self.frame.shape[0])
            cv2.line(
                canvas,
                self._display(endpoints[0], x0, y0, scale, ox, oy),
                self._display(endpoints[1], x0, y0, scale, ox, oy),
                (255, 255, 0), 4, cv2.LINE_AA,
            )
        for index, point in enumerate(self.points, 1):
            display = self._display(point, x0, y0, scale, ox, oy)
            cv2.circle(canvas, display, 7, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, str(index), (display[0] + 8, display[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
        self._draw_panel(canvas)
        return canvas

    def _draw_panel(self, canvas):
        _kind, title, explanation = self.TARGETS[self.target_index]
        lines = (
            "11v11-DWARSLIJNEN",
            f"STAP {self.target_index + 1} VAN 2",
            "",
            f"KLIK NU: {title}",
            explanation,
            "",
            "Niet aanklikken:",
            "- een zijrand van het strafschopgebied",
            "- een gele 8v8-lijn of rij hoedjes",
            "- de doellijn door de doelpalen",
            "",
            "Klik 5 verspreide punten op precies",
            "dezelfde lange rechte witte kalklijn.",
            "Volgorde en kleine afwijkingen zijn niet erg.",
            "",
            f"Videomoment {self.time_seconds:.1f}s | zoom {self.zoom:.1f}x",
            f"Punten {len(self.points)}/5",
            "",
            "Muiswiel of +/-: zoomen",
            "Pijltjes of W/A/S/D: beeld bewegen",
            ", en . : 1 seconde eerder/later",
            "0: volledig beeld | U: laatste punt",
            "R: punten wissen | Enter: bevestigen",
            "Esc: afbreken",
            "",
            self.status,
        )
        for index, text in enumerate(lines):
            color = (0, 230, 255) if index in (0, 1, 3, 25) else (235, 235, 235)
            cv2.putText(canvas, text, (18, 34 + index * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 2 if index in (0, 3) else 1, cv2.LINE_AA)

    @staticmethod
    def _display(point, x0, y0, scale, ox, oy):
        return int(round(ox + (point[0] - x0) * scale)), int(round(oy + (point[1] - y0) * scale))


def _line_endpoints(line, width, height):
    a, b, c = map(float, line)
    candidates = []
    if abs(b) > 1e-9:
        for x in (0.0, width - 1.0):
            y = -(a * x + c) / b
            if 0 <= y < height:
                candidates.append((x, y))
    if abs(a) > 1e-9:
        for y in (0.0, height - 1.0):
            x = -(b * y + c) / a
            if 0 <= x < width:
                candidates.append((x, y))
    unique = []
    for point in candidates:
        if not any(np.linalg.norm(np.asarray(point) - item) < 1 for item in unique):
            unique.append(np.asarray(point))
    if len(unique) < 2:
        raise ValueError("De lijn kan niet binnen het frame worden getekend.")
    return tuple(unique[0]), tuple(unique[1])


def _write_preview(video, reference, path):
    capture = cv2.VideoCapture(str(video))
    panels = []
    labels = {"midfield": "MIDDENLIJN", "goal_area_5m": "5-METERLIJN", "penalty_area_16m": "16-METERLIJN"}
    try:
        for line in reference.lines:
            capture.set(cv2.CAP_PROP_POS_FRAMES, line.frame_number)
            ok, frame = capture.read()
            if not ok:
                continue
            first, second = _line_endpoints(np.asarray(line.equation), frame.shape[1], frame.shape[0])
            cv2.line(frame, tuple(np.rint(first).astype(int)), tuple(np.rint(second).astype(int)), (255, 255, 0), 6, cv2.LINE_AA)
            for point in line.points:
                cv2.circle(frame, tuple(np.rint(point).astype(int)), 8, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 65), (12, 12, 12), -1)
            cv2.putText(frame, f"11v11 {labels[line.line_type]} | PARALLEL AAN 8v8-ZIJLIJNEN", (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 230, 255), 2, cv2.LINE_AA)
            panels.append(cv2.resize(frame, (640, 360)))
    finally:
        capture.release()
    if len(panels) == 3:
        cv2.imwrite(str(path), np.hstack(panels))


def main():
    parser = argparse.ArgumentParser(description="Verzamel 5m- en 16m-lijn als parallelcontrole.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--time", type=float, help="Optioneel startmoment voor de 5m-lijn.")
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    midfield = load_manual_midfield_line(output / f"{prefix}_manual_midfield_line.json")
    perspective = load_manual_perspective_reference(output / f"{prefix}_manual_perspective_reference.json")
    initial = args.time
    if initial is None:
        initial = next(view.time_seconds for view in perspective.views if view.label == "right_goal")
    extras = ExtraParallelLineCollector(video, initial).run()
    reference = ManualParallelLineReference(
        video.name, (ManualParallelLine.from_midfield(midfield), *extras)
    )
    path = output / f"{prefix}_manual_parallel_lines.json"
    save_manual_parallel_lines(reference, path)
    preview = output / f"{prefix}_manual_parallel_lines.jpg"
    _write_preview(video, reference, preview)
    print(f"Parallelle 11v11-lijnen opgeslagen: {path}")
    for line in reference.lines:
        print(
            f"{line.line_type}: frame {line.frame_number} ({line.time_seconds:.1f}s) | "
            f"RMS {line.rms_error_px:.2f}px | max {line.maximum_error_px:.2f}px"
        )
    print(f"QA-preview: {preview}")


if __name__ == "__main__":
    main()
