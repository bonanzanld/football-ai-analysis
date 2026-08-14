from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from football_ai.calibration.bootstrap.goal_seed import load_goal_seeds
from football_ai.calibration.goal_structure_observation import (
    GoalStructureLine,
    GoalStructureObservation,
    save_goal_structure_observations,
    load_goal_structure_observations,
)


class GoalStructureCollector:
    WINDOW = "Football AI - doelstructuur"
    CANVAS = (1600, 900)
    PANEL_WIDTH = 430
    POINTS_PER_LINE = 5
    TARGETS = (
        ("far_post", "VERSTE PAAL", "Klik van grondcontact naar boven."),
        ("crossbar", "LAT", "Klik van de verste naar de dichtstbijzijnde paal."),
        ("near_post", "DICHTSTBIJZIJNDE PAAL", "Klik van boven naar het grondcontact."),
        ("goal_line", "WITTE DOELLIJN", "Klik verspreid langs de witte lijn door beide palen."),
    )

    def __init__(self, video: Path, seeds) -> None:
        self.video = video
        self.seeds = tuple(seeds)
        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened():
            raise FileNotFoundError(video)
        self.goal_index = 0
        self.target_index = 0
        self.pending: list[tuple[float, float]] = []
        self.lines: list[GoalStructureLine] = []
        self.results: list[GoalStructureObservation] = []
        self.zoom = 1.0
        self.center: tuple[float, float] | None = None
        self.frame = self._read_current()
        self.view_rect = (self.PANEL_WIDTH, 0, self.CANVAS[0] - self.PANEL_WIDTH, self.CANVAS[1])
        self.mapping = (0.0, 0.0, 1.0, 0.0, 0.0)
        self.status = "Klik vijf verspreide punten; kleine afwijkingen binnen de lijndikte zijn toegestaan."

    def run(self) -> tuple[GoalStructureObservation, ...]:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, *self.CANVAS)
        cv2.setMouseCallback(self.WINDOW, self._mouse)
        try:
            while self.goal_index < len(self.seeds):
                cv2.imshow(self.WINDOW, self._render())
                key = cv2.waitKeyEx(30)
                if key in (27, ord("q"), ord("Q")):
                    raise KeyboardInterrupt("Doelstructuurinvoer afgebroken.")
                if key in (ord("u"), ord("U"), 8, 127):
                    self._undo()
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
        finally:
            self.capture.release()
            cv2.destroyWindow(self.WINDOW)
        return tuple(self.results)

    def _read_current(self) -> np.ndarray:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.seeds[self.goal_index].frame_number)
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Referentieframe kon niet worden gelezen.")
        return frame

    def _zoom(self, factor: float) -> None:
        height, width = self.frame.shape[:2]
        self.center = self.center or (width / 2.0, height / 2.0)
        self.zoom = float(np.clip(self.zoom * factor, 1.0, 10.0))

    def _pan(self, dx: float, dy: float) -> None:
        if self.zoom <= 1.0:
            self.status = "Zoom eerst in; bij volledig beeld is pannen niet nodig."
            return
        height, width = self.frame.shape[:2]
        cx, cy = self.center or (width / 2.0, height / 2.0)
        crop_w, crop_h = width / self.zoom, height / self.zoom
        self.center = (
            float(np.clip(cx + dx * crop_w, crop_w / 2.0, width - crop_w / 2.0)),
            float(np.clip(cy + dy * crop_h, crop_h / 2.0, height - crop_h / 2.0)),
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
            self._zoom(1.25 if flags > 0 else 0.8)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x0, y0, scale, ox, oy = self.mapping
        if x < ox or y < oy:
            return
        point = (x0 + (x - ox) / scale, y0 + (y - oy) / scale)
        height, width = self.frame.shape[:2]
        if not (0.0 <= point[0] < width and 0.0 <= point[1] < height):
            return
        self.pending.append(point)
        if len(self.pending) == self.POINTS_PER_LINE:
            name = self.TARGETS[self.target_index][0]
            self.lines.append(GoalStructureLine.fit(name, tuple(self.pending)))
            self.pending.clear()
            self.target_index += 1
            if self.target_index == len(self.TARGETS):
                seed = self.seeds[self.goal_index]
                self.results.append(GoalStructureObservation(seed.goal_id, seed.frame_number, seed.time_seconds, tuple(self.lines)))
                self.goal_index += 1
                if self.goal_index < len(self.seeds):
                    self.target_index, self.lines = 0, []
                    self.zoom, self.center = 1.0, None
                    self.frame = self._read_current()
                    self.status = "Volgend doel. Begin opnieuw met de verste paal."
            else:
                self.status = "Lijn opgeslagen. Ga verder met het volgende onderdeel."

    def _undo(self) -> None:
        if self.pending:
            self.pending.pop()
        elif self.lines:
            line = self.lines.pop()
            self.target_index -= 1
            self.pending = list(line.points[:-1])
        else:
            self.status = "Er is niets om ongedaan te maken."

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
        for line in self.lines:
            points = [self._display(item, x0, y0, scale, ox, oy) for item in line.points]
            cv2.polylines(canvas, [np.asarray(points)], False, (0, 255, 255), 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(canvas, point, 5, (255, 0, 255), -1, cv2.LINE_AA)
        for point in self.pending:
            cv2.circle(canvas, self._display(point, x0, y0, scale, ox, oy), 6, (255, 0, 255), -1, cv2.LINE_AA)
        name, title, direction = self.TARGETS[self.target_index]
        texts = (
            "DOELSTRUCTUUR 5 x 2 METER",
            f"Doel {self.seeds[self.goal_index].goal_id} | onderdeel {self.target_index + 1}/4",
            "",
            title,
            direction,
            f"Punt {len(self.pending) + 1}/5",
            "",
            "Muiswiel of +/-: zoomen",
            "Pijltjes of W/A/S/D: beeld bewegen",
            "0: volledig beeld",
            "U: laatste klik ongedaan",
            "Esc: afbreken",
            "",
            "Klik ongeveer midden op paal/lijn.",
            "Een kleine marge wordt gemiddeld.",
            "",
            self.status,
        )
        for index, text in enumerate(texts):
            color = (0, 230, 255) if index in (0, 3, 4, 5) else (235, 235, 235)
            cv2.putText(canvas, text, (16, 34 + index * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1 if index else 2, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _display(point, x0, y0, scale, ox, oy):
        return int(round(ox + (point[0] - x0) * scale)), int(round(oy + (point[1] - y0) * scale))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verzamel robuuste lijnen op beide wedstrijdgoals.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--goal", choices=("A", "B"))
    parser.add_argument("--time", type=float)
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    seeds = load_goal_seeds(output_dir / f"{prefix}_goal_seeds.json")
    if args.goal is not None:
        selected = next(seed for seed in seeds if seed.goal_id == args.goal)
        if args.time is not None:
            capture = cv2.VideoCapture(str(video))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            capture.release()
            if fps <= 0.0:
                raise RuntimeError("Video-FPS kon niet worden gelezen.")
            selected = replace(
                selected,
                frame_number=int(round(args.time * fps)),
                time_seconds=float(args.time),
            )
        seeds = (selected,)
    observations = GoalStructureCollector(video, seeds).run()
    output = output_dir / f"{prefix}_goal_structure_lines.json"
    if args.goal is not None and output.exists():
        previous = load_goal_structure_observations(output)
        by_goal = {item.goal_id: item for item in previous}
        by_goal.update({item.goal_id: item for item in observations})
        observations = tuple(by_goal[key] for key in ("A", "B") if key in by_goal)
    save_goal_structure_observations(observations, output)
    print(f"Doelstructuurlijnen opgeslagen: {output}")
    for goal in observations:
        print(f"Doel {goal.goal_id}:")
        for line in goal.lines:
            print(f"  {line.name}: RMS {line.rms_error_px:.2f}px | max {line.maximum_error_px:.2f}px")


if __name__ == "__main__":
    main()
