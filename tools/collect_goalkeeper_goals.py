from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.classification.goalkeeper_goal_reference import (
    GoalkeeperGoalReference,
    save_goalkeeper_goal_references,
)
from football_ai.calibration.manual_ui_controls import (
    CANCEL_KEYS,
    FINISH_KEYS,
    PAN_KEY_DIRECTIONS,
    RESET_VIEW_KEYS,
    UNDO_KEYS,
    ZOOM_IN_KEYS,
    ZOOM_OUT_KEYS,
    mouse_wheel_direction,
)


class GoalReferenceApp:
    WINDOW = "Football AI - doelen voor keeperdetectie"
    WIDTH = 1600
    HEIGHT = 900
    PANEL = 390

    def __init__(
        self,
        video_path: Path,
        maximum_seconds: float,
        team_a_name: str,
        team_b_name: str,
    ) -> None:
        self.video_path = video_path
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Video kon niet worden geopend: {video_path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = min(frame_count - 1, round(maximum_seconds * self.fps))
        step = max(1, round(self.fps))
        self.samples = list(range(0, end_frame + 1, step))
        self.index = 0
        self.team_names = (team_a_name, team_b_name)
        self.active_team: int | None = None
        self.pending: list[tuple[float, float]] = []
        self.references: dict[int, GoalkeeperGoalReference] = {}
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.image_rect = (self.PANEL, 0, self.WIDTH - self.PANEL, self.HEIGHT)
        self.scale = 1.0
        self.origin = (0, 0)
        self.crop = (0, 0)
        self.zoom = 1.0
        self.view_center = [0.5, 0.5]
        self.status = "Zoek een beeld waarop een wedstrijdgoal duidelijk zichtbaar is."
        self._load()

    def run(self) -> tuple[GoalkeeperGoalReference, ...]:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.WIDTH, self.HEIGHT)
        cv2.setMouseCallback(self.WINDOW, self._mouse)
        try:
            while True:
                cv2.imshow(self.WINDOW, self._render())
                key = cv2.waitKeyEx(30)
                if key in (*CANCEL_KEYS, ord("q"), ord("Q")):
                    raise RuntimeError("Doelreferentie afgebroken.")
                if key in FINISH_KEYS:
                    if not self.references:
                        self.status = "Geef minimaal één doel aan voordat je afrondt."
                    else:
                        return tuple(self.references[key] for key in sorted(self.references))
                elif key in (ord("1"), ord("2")):
                    self.active_team = int(chr(key)) - 1
                    self.pending = []
                    self.status = (
                        f"{self.team_names[self.active_team]} verdedigt dit doel. "
                        "Klik eerst de VERSTE paal vanuit de camera, daarna de "
                        "DICHTSTBIJZIJNDE paal. Klik steeds waar de paal de grond raakt."
                    )
                elif key in UNDO_KEYS:
                    if self.pending:
                        self.pending.pop()
                    elif self.active_team is not None:
                        self.references.pop(self.active_team, None)
                    self.status = "Laatste doelinvoer ongedaan gemaakt."
                elif key in (ord("p"), ord("P")):
                    self._move(-1)
                elif key in (ord("n"), ord("N")):
                    self._move(1)
                elif key in ZOOM_IN_KEYS:
                    self._change_zoom(1.25)
                elif key in ZOOM_OUT_KEYS:
                    self._change_zoom(0.8)
                elif key in RESET_VIEW_KEYS:
                    self.zoom = 1.0
                    self.view_center = [0.5, 0.5]
                elif key in PAN_KEY_DIRECTIONS:
                    self._pan(*PAN_KEY_DIRECTIONS[key])
        finally:
            self.capture.release()
            cv2.destroyWindow(self.WINDOW)

    def _move(self, step: int) -> None:
        self.index = min(max(self.index + step, 0), len(self.samples) - 1)
        self.pending = []
        self.zoom = 1.0
        self.view_center = [0.5, 0.5]
        self._load()

    def _change_zoom(self, factor: float) -> None:
        self.zoom = min(8.0, max(1.0, self.zoom * factor))

    def _pan(self, horizontal: int, vertical: int) -> None:
        if self.zoom <= 1.0:
            self.status = "Zoom eerst in; bij volledig beeld is verplaatsen niet nodig."
            return
        step = 0.12 / self.zoom
        self.view_center[0] = min(1.0, max(0.0, self.view_center[0] + horizontal * step))
        self.view_center[1] = min(1.0, max(0.0, self.view_center[1] + vertical * step))

    def _load(self) -> None:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.samples[self.index])
        success, frame = self.capture.read()
        if success:
            self.frame = frame

    def _mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        if event == cv2.EVENT_MOUSEWHEEL:
            ox, oy = self.origin
            crop_x, crop_y = self.crop
            height, width = self.frame.shape[:2]
            image_x = (x - ox + crop_x) / self.scale
            image_y = (y - oy + crop_y) / self.scale
            if 0 <= image_x < width and 0 <= image_y < height:
                self.view_center = [image_x / width, image_y / height]
            self._change_zoom(1.25 if mouse_wheel_direction(flags) > 0 else 0.8)
            return
        if event != cv2.EVENT_LBUTTONDOWN or self.active_team is None:
            return
        ox, oy = self.origin
        crop_x, crop_y = self.crop
        image_x = (x - ox + crop_x) / self.scale
        image_y = (y - oy + crop_y) / self.scale
        height, width = self.frame.shape[:2]
        if not (0 <= image_x < width and 0 <= image_y < height):
            return
        self.pending.append((float(image_x), float(image_y)))
        if len(self.pending) == 2:
            frame_number = self.samples[self.index]
            self.references[self.active_team] = GoalkeeperGoalReference(
                frame_number=frame_number,
                time_seconds=frame_number / self.fps,
                defending_team_id=self.active_team,
                first_post=self.pending[0],
                second_post=self.pending[1],
            )
            self.status = (
                f"Doel voor {self.team_names[self.active_team]} opgeslagen. "
                "De verste en dichtstbijzijnde paal zijn opgeslagen. "
                "Zoek eventueel het andere doel, of druk Enter."
            )
            self.pending = []

    def _render(self) -> np.ndarray:
        canvas = np.full((self.HEIGHT, self.WIDTH, 3), 24, dtype=np.uint8)
        target_width = self.WIDTH - self.PANEL
        base_scale = min(target_width / self.frame.shape[1], self.HEIGHT / self.frame.shape[0])
        scale = base_scale * self.zoom
        shown = cv2.resize(
            self.frame,
            (round(self.frame.shape[1] * scale), round(self.frame.shape[0] * scale)),
        )
        crop_x = min(
            max(round(self.view_center[0] * shown.shape[1] - target_width / 2), 0),
            max(0, shown.shape[1] - target_width),
        )
        crop_y = min(
            max(round(self.view_center[1] * shown.shape[0] - self.HEIGHT / 2), 0),
            max(0, shown.shape[0] - self.HEIGHT),
        )
        visible_width = min(target_width, shown.shape[1])
        visible_height = min(self.HEIGHT, shown.shape[0])
        ox = self.PANEL + (target_width - visible_width) // 2
        oy = (self.HEIGHT - visible_height) // 2
        self.scale = scale
        self.origin = (ox, oy)
        self.crop = (crop_x, crop_y)
        canvas[oy:oy + visible_height, ox:ox + visible_width] = shown[
            crop_y:crop_y + visible_height,
            crop_x:crop_x + visible_width,
        ]
        self._draw_panel(canvas)
        for team_id, reference in self.references.items():
            if reference.frame_number != self.samples[self.index]:
                continue
            points = [reference.first_post, reference.second_post]
            color = (255, 120, 0) if team_id == 0 else (0, 0, 255)
            rendered = [
                (round(x * scale) - crop_x + ox, round(y * scale) - crop_y + oy)
                for x, y in points
            ]
            cv2.line(canvas, rendered[0], rendered[1], color, 4, cv2.LINE_AA)
            for point_index, point in enumerate(rendered):
                cv2.circle(canvas, point, 9, color, -1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    "VERSTE PAAL" if point_index == 0 else "DICHTSTBIJZIJNDE PAAL",
                    (point[0] + 12, point[1] - 12),
                    0,
                    0.48,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        for point in self.pending:
            rendered = (
                round(point[0] * scale) - crop_x + ox,
                round(point[1] * scale) - crop_y + oy,
            )
            cv2.circle(canvas, rendered, 9, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "VERSTE PAAL",
                (rendered[0] + 12, rendered[1] - 12),
                0,
                0.48,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return canvas

    def _draw_panel(self, canvas: np.ndarray) -> None:
        lines = [
            ("DOELEN VOOR KEEPERDETECTIE", (255, 255, 255)),
            (f"Beeld {self.index + 1}/{len(self.samples)} | {self.samples[self.index] / self.fps:.1f}s", (0, 255, 255)),
            ("", (255, 255, 255)),
            (f"1 = doel verdedigd door {self.team_names[0]}", (255, 160, 40)),
            (f"2 = doel verdedigd door {self.team_names[1]}", (80, 80, 255)),
            ("", (255, 255, 255)),
            ("VASTE KLIKVOLGORDE:", (0, 255, 255)),
            ("1. VERSTE doelpaal vanaf camera", (0, 255, 255)),
            ("2. DICHTSTBIJZIJNDE doelpaal", (0, 255, 255)),
            ("Klik waar iedere paal de GROND raakt", (255, 255, 255)),
            ("", (255, 255, 255)),
            (f"Zoom: {self.zoom:.1f}x", (200, 200, 200)),
            ("Muiswiel of +/- = in-/uitzoomen", (200, 200, 200)),
            ("Pijlen of W/A/S/D = beeld bewegen", (200, 200, 200)),
            ("0 = volledig beeld", (200, 200, 200)),
            ("P/N = vorig/volgend beeld", (200, 200, 200)),
            ("U = ongedaan maken", (200, 200, 200)),
            ("Enter = opslaan", (200, 200, 200)),
            ("Q/Esc = afbreken", (200, 200, 200)),
        ]
        y = 42
        for text, color in lines:
            cv2.putText(canvas, text, (18, y), 0, 0.48, color, 1, cv2.LINE_AA)
            y += 34
        for team_id, reference in sorted(self.references.items()):
            cv2.putText(
                canvas,
                f"OPGESLAGEN: {self.team_names[team_id]} bij {reference.time_seconds:.1f}s",
                (18, y + 12),
                0,
                0.43,
                (80, 255, 80),
                1,
                cv2.LINE_AA,
            )
            y += 30
        status_lines = [self.status[index:index + 48] for index in range(0, len(self.status), 48)]
        for index, text in enumerate(status_lines[:4]):
            cv2.putText(canvas, text, (18, 790 + index * 24), 0, 0.42, (0, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wijs lokale doelen aan voor keepersdetectie.")
    parser.add_argument("--video", required=True, help="Bestandsnaam in videos/ of volledig pad.")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--team-a-name", default="Brandevoort (groen-wit)")
    parser.add_argument("--team-b-name", default="Brabantia (rood-blauw)")
    args = parser.parse_args()
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / "videos" / video_path
    references = GoalReferenceApp(
        video_path,
        args.seconds,
        args.team_a_name,
        args.team_b_name,
    ).run()
    output = PROJECT_ROOT / "output" / "entities" / f"{video_path.stem}_goal_references.json"
    save_goalkeeper_goal_references(video_path.name, references, output)
    print(f"Doelreferenties opgeslagen: {output}")
    for item in references:
        print(
            f"{args.team_a_name if item.defending_team_id == 0 else args.team_b_name}: "
            f"frame {item.frame_number} ({item.time_seconds:.1f}s)"
        )


if __name__ == "__main__":
    main()
