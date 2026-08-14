from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EndLineCollector:
    WINDOW = "Football AI - 8v8 achterlijn"
    PANEL = 480
    WIDTH, HEIGHT = 1700, 900

    def __init__(self, video: Path, start: float, minimum: float, maximum: float, side: str, corners: bool = False, sideline_supports: bool = False, reference_corners: tuple[tuple[float, float], ...] = ()) -> None:
        self.video = video
        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened():
            raise FileNotFoundError(video)
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.minimum, self.maximum = minimum, maximum
        self.side = side
        self.corners = bool(corners)
        self.sideline_supports = bool(sideline_supports)
        if self.corners and self.sideline_supports:
            raise ValueError("Kies hoekpunten of zijlijnsteun, niet beide.")
        self.required_points = 2 if self.corners or self.sideline_supports else 3
        self.reference_corners = tuple(reference_corners)
        self.time = float(np.clip(start, minimum, maximum))
        self.frame = self._read()
        self.points: list[tuple[float, float]] = []
        self.zoom = 1.0
        self.center: tuple[float, float] | None = None
        self.mapping = (0.0, 0.0, 1.0, 0.0, 0.0)
        self.done = False
        self.status = "Kies een beeld waarop dezelfde 8v8-achterlijn met meerdere hoedjes zichtbaar is."

    def _read(self):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, round(self.time * self.fps))
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"Frame {self.time:.1f}s niet leesbaar")
        return frame

    def run(self):
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.WIDTH, self.HEIGHT)
        cv2.setMouseCallback(self.WINDOW, self._mouse)
        try:
            while not self.done:
                cv2.imshow(self.WINDOW, self._render())
                key = cv2.waitKeyEx(30)
                if key in (27, ord("q"), ord("Q")):
                    raise KeyboardInterrupt
                if key in (ord("u"), ord("U"), 8, 127) and self.points:
                    self.points.pop()
                elif key in (ord(","), 2424832) and not self.points:
                    self._shift(-1.0)
                elif key in (ord("."), 2555904) and not self.points:
                    self._shift(1.0)
                elif key in (ord("+"), ord("=")):
                    self.zoom = min(8.0, self.zoom * 1.25)
                elif key in (ord("-"), ord("_")):
                    self.zoom = max(1.0, self.zoom / 1.25)
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
                elif key in (10, 13) and len(self.points) == self.required_points:
                    self.done = True
        finally:
            self.capture.release()
            cv2.destroyWindow(self.WINDOW)
        values = np.asarray(self.points)
        if self.corners:
            return {
                "schema_version": 1,
                "video_name": self.video.name,
                "frame_number": int(round(self.time * self.fps)),
                "time_seconds": self.time,
                "role": f"8v8_{self.side}_end_line_corners",
                "field_x_m": 0.0 if self.side == "left" else 64.0,
                "rear_corner": values[0].tolist(),
                "front_corner": values[1].tolist(),
                "provenance": "human_reviewed",
            }
        if self.sideline_supports:
            return {
                "schema_version": 1,
                "video_name": self.video.name,
                "frame_number": int(round(self.time * self.fps)),
                "time_seconds": self.time,
                "role": f"8v8_{self.side}_sideline_supports",
                "field_x_m": 0.0 if self.side == "left" else 64.0,
                "rear_sideline_support": values[0].tolist(),
                "front_sideline_support": values[1].tolist(),
                "provenance": "human_reviewed",
            }
        center = values.mean(axis=0)
        _u, _s, vh = np.linalg.svd(values - center)
        normal = np.asarray((-vh[0, 1], vh[0, 0]))
        equation = np.asarray((*normal, -normal @ center))
        equation /= np.linalg.norm(equation[:2])
        errors = np.abs(np.column_stack((values, np.ones(3))) @ equation)
        return {
            "schema_version": 1, "video_name": self.video.name,
            "frame_number": int(round(self.time * self.fps)), "time_seconds": self.time,
            "role": f"8v8_{self.side}_end_line_transverse_to_sidelines",
            "field_x_m": 0.0 if self.side == "left" else 64.0,
            "points": values.tolist(), "equation": equation.tolist(),
            "rms_error_px": float(np.sqrt(np.mean(errors ** 2))),
            "maximum_error_px": float(errors.max()),
        }

    def _shift(self, delta):
        self.time = float(np.clip(self.time + delta, self.minimum, self.maximum))
        self.frame = self._read()
        self.zoom, self.center = 1.0, None

    def _pan(self, dx: float, dy: float) -> None:
        if self.zoom <= 1.0:
            self.status = "Zoom eerst in; bij volledig beeld is verplaatsen niet nodig."
            return
        height, width = self.frame.shape[:2]
        crop_width, crop_height = width / self.zoom, height / self.zoom
        center_x, center_y = self.center or (width / 2.0, height / 2.0)
        self.center = (
            float(np.clip(center_x + dx * crop_width, crop_width / 2.0, width - crop_width / 2.0)),
            float(np.clip(center_y + dy * crop_height, crop_height / 2.0, height - crop_height / 2.0)),
        )

    def _crop(self):
        h, w = self.frame.shape[:2]
        cw, ch = w / self.zoom, h / self.zoom
        cx, cy = self.center or (w / 2, h / 2)
        x0 = float(np.clip(cx - cw / 2, 0, w - cw)); y0 = float(np.clip(cy - ch / 2, 0, h - ch))
        return self.frame[int(y0):int(y0 + ch), int(x0):int(x0 + cw)], (x0, y0, cw, ch)

    def _mouse(self, event, x, y, flags, _data):
        if event == cv2.EVENT_MOUSEWHEEL:
            self.zoom = min(8.0, self.zoom * 1.25) if flags > 0 else max(1.0, self.zoom / 1.25)
            return
        if event != cv2.EVENT_LBUTTONDOWN or len(self.points) >= self.required_points:
            return
        x0, y0, scale, ox, oy = self.mapping
        point = (x0 + (x - ox) / scale, y0 + (y - oy) / scale)
        h, w = self.frame.shape[:2]
        if 0 <= point[0] < w and 0 <= point[1] < h:
            if self.sideline_supports and any(
                np.linalg.norm(np.subtract(point, corner)) < 25.0
                for corner in self.reference_corners
            ):
                self.status = "Dit is een hoekpunt. Klik een VOLGEND hoedje verder de zijlijn op."
                return
            self.points.append(point)

    def _render(self):
        canvas = np.full((self.HEIGHT, self.WIDTH, 3), 24, np.uint8)
        crop, (x0, y0, cw, ch) = self._crop()
        scale = min((self.WIDTH - self.PANEL) / cw, self.HEIGHT / ch)
        shown = cv2.resize(crop, (round(cw * scale), round(ch * scale)))
        ox = self.PANEL + (self.WIDTH - self.PANEL - shown.shape[1]) // 2; oy = (self.HEIGHT - shown.shape[0]) // 2
        canvas[oy:oy + shown.shape[0], ox:ox + shown.shape[1]] = shown
        self.mapping = (x0, y0, scale, ox, oy)
        for point in self.points:
            p = (round(ox + (point[0] - x0) * scale), round(oy + (point[1] - y0) * scale))
            cv2.circle(canvas, p, 8, (255, 0, 255), -1)
        for point in self.reference_corners:
            p = (round(ox + (point[0] - x0) * scale), round(oy + (point[1] - y0) * scale))
            cv2.circle(canvas, p, 12, (0, 140, 255), 4, cv2.LINE_AA)
        side_label = "LINKER" if self.side == "left" else "RECHTER"
        if self.corners:
            instructions = (
                "Klik exact de TWEE hoekhoedjes:",
                "1. achterste hoek van de achterlijn",
                "2. voorste hoek van de achterlijn",
                "Niet zomaar punten op de lijn klikken.",
            )
        elif self.sideline_supports:
            instructions = (
                "Klik exact TWEE volgende hoedjes:",
                "1. op de ACHTERSTE zijlijn",
                "2. op de VOORSTE zijlijn",
                "Kies per lijn een hoedje vanaf de hoek het veld in.",
            )
        else:
            instructions = (
                "Klik 3 verspreide punten op",
                f"DEZELFDE {side_label} achterlijn die",
                "haaks op de 8v8-zijlijnen staat.",
                "Gebruik betrouwbare hoedjes of zichtbare lijnpunten.",
            )
        lines = (f"{side_label} 8v8-ACHTERLIJN", "", *instructions, "", "Geen 11v11-middenlijn klikken.", "", f"Tijd: {self.time:.1f}s", f"Punten: {len(self.points)}/{self.required_points}", "", ",/. ander moment", "Muiswiel of +/- zoom", "Pijltjes of W/A/S/D bewegen", "U ongedaan | Enter opslaan", "Esc afbreken")
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (18, 35 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 230, 255) if i in (0, 2) else (235, 235, 235), 1 if i else 2, cv2.LINE_AA)
        return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True); parser.add_argument("--format", default="8v8")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--corners", action="store_true", help="Klik de twee echte hoekhoedjes in plaats van drie lijnpunten.")
    mode.add_argument("--sideline-supports", action="store_true", help="Klik een extra hoedje op beide zijlijnen.")
    parser.add_argument("--start", type=float, default=969.5); parser.add_argument("--minimum", type=float, default=951.0); parser.add_argument("--maximum", type=float, default=997.5)
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    reference_corners = ()
    if args.sideline_supports:
        corner_path = (
            PROJECT_ROOT / "output" / "pitch_bootstrap"
            / f"{video.stem}_{args.format}_manual_8v8_{args.side}_endline_corners.json"
        )
        corner_data = json.loads(corner_path.read_text(encoding="utf-8"))
        reference_corners = (
            tuple(corner_data["rear_corner"]),
            tuple(corner_data["front_corner"]),
        )
    result = EndLineCollector(
        video, args.start, args.minimum, args.maximum, args.side,
        args.corners, args.sideline_supports, reference_corners,
    ).run()
    suffix = (
        "sideline_supports" if args.sideline_supports
        else "endline_corners" if args.corners
        else "endline"
    )
    output = PROJECT_ROOT / "output" / "pitch_bootstrap" / f"{video.stem}_{args.format}_manual_8v8_{args.side}_{suffix}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    detail = (
        "2 menselijke zijlijnsteunpunten" if args.sideline_supports
        else "2 menselijke hoekpunten" if args.corners
        else f"RMS {result['rms_error_px']:.2f}px"
    )
    print(f"8v8-achterlijn opgeslagen: {output} | {detail}")


if __name__ == "__main__":
    main()
