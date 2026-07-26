from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class SidelineAnchor:
    camera_state: int
    view_position: float
    frame_number: int
    time_seconds: float
    rear_point: tuple[float, float] | None
    front_point: tuple[float, float] | None

    def to_dict(self) -> dict:
        return {
            "camera_state": self.camera_state,
            "view_position": self.view_position,
            "frame_number": self.frame_number,
            "time_seconds": self.time_seconds,
            "rear_point": list(self.rear_point) if self.rear_point is not None else None,
            "front_point": list(self.front_point) if self.front_point is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SidelineAnchor":
        return cls(
            camera_state=int(data["camera_state"]),
            view_position=float(data["view_position"]),
            frame_number=int(data["frame_number"]),
            time_seconds=float(data["time_seconds"]),
            rear_point=(tuple(map(float, data["rear_point"])) if data.get("rear_point") is not None else None),
            front_point=(tuple(map(float, data["front_point"])) if data.get("front_point") is not None else None),
        )


def save_sideline_anchors(anchors: tuple[SidelineAnchor, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "anchors": [item.to_dict() for item in anchors]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_sideline_anchors(path: Path) -> tuple[SidelineAnchor, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(SidelineAnchor.from_dict(item) for item in payload["anchors"])


def select_intermediate_camera_states(
    bootstrap_report: dict,
    excluded_states: set[int],
) -> list[dict]:
    states = [
        item for item in bootstrap_report["camera_states"]
        if int(item["camera_state"]) not in excluded_states
    ]
    return sorted(states, key=lambda item: float(item.get("view_position", 0.5)))


class SidelineAnchorApp:
    WINDOW = "Football AI - tussenankers zijlijnen"

    def __init__(
        self,
        video_path: Path,
        bootstrap_path: Path,
        excluded_states: set[int],
    ) -> None:
        report = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        self.states = select_intermediate_camera_states(report, excluded_states)
        if not self.states:
            raise RuntimeError("Geen tussenliggende camerastanden gevonden.")
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Video kon niet worden geopend: {video_path}")
        self.index = 0
        self.frame: np.ndarray | None = None
        self.points: dict[int, list[tuple[float, float] | None]] = {}
        self.zoom = 1.0
        self.zoom_center: tuple[float, float] | None = None
        self.viewport_rect = (400, 0, 1200, 900)
        self.image_rect = self.viewport_rect
        self.view_origin = (0.0, 0.0)
        self.view_scale = 1.0
        self.status = "Klik op de VERRE zijlijn, of druk S wanneer die niet zichtbaar is."
        self._load()

    def run(self) -> tuple[SidelineAnchor, ...]:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 1600, 900)
        cv2.setMouseCallback(self.WINDOW, self._mouse)
        while True:
            cv2.imshow(self.WINDOW, self._render())
            key = cv2.waitKeyEx(30)
            if key == 27:
                self._close()
                raise RuntimeError("Tussenankerselectie afgebroken.")
            if key in (2424832, 65361, ord("p"), ord("P")):
                self._move(-1)
            elif key in (2555904, 65363, ord("n"), ord("N")):
                self._move(1)
            elif key in (8, 127, ord("u"), ord("U")):
                points = self.points.setdefault(self._state_id(), [])
                if points:
                    points.pop()
                    self._set_instruction()
            elif key in (ord("s"), ord("S")):
                points = self.points.setdefault(self._state_id(), [])
                if len(points) < 2:
                    points.append(None)
                    self._set_instruction()
            elif key in (ord("+"), ord("=")):
                self.zoom = min(8.0, self.zoom * 1.25)
            elif key in (ord("-"), ord("_")):
                self.zoom = max(1.0, self.zoom * 0.8)
            elif key == ord("0"):
                self.zoom, self.zoom_center = 1.0, None
            elif key in (10, 13):
                if all(len(self.points.get(int(item["camera_state"]), [])) == 2 for item in self.states):
                    anchors = tuple(self._create_anchor(item) for item in self.states)
                    self._close()
                    return anchors
                missing = sum(len(self.points.get(int(item["camera_state"]), [])) < 2 for item in self.states)
                self.status = f"Nog {missing} camerastand(en) niet compleet. Gebruik P/N om ze te openen."

    def _state_id(self) -> int:
        return int(self.states[self.index]["camera_state"])

    def _create_anchor(self, state: dict) -> SidelineAnchor:
        points = self.points[int(state["camera_state"])]
        return SidelineAnchor(
            camera_state=int(state["camera_state"]),
            view_position=float(state.get("view_position", 0.5)),
            frame_number=int(state["representative_frame_number"]),
            time_seconds=float(state["representative_time_seconds"]),
            rear_point=points[0],
            front_point=points[1],
        )

    def _move(self, amount: int) -> None:
        self.index = (self.index + amount) % len(self.states)
        self.zoom, self.zoom_center = 1.0, None
        self._load()
        self._set_instruction()

    def _load(self) -> None:
        frame_number = int(self.states[self.index]["representative_frame_number"])
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = self.capture.read()
        if not success:
            raise RuntimeError(f"Frame {frame_number} kon niet worden gelezen.")
        self.frame = frame

    def _set_instruction(self) -> None:
        count = len(self.points.get(self._state_id(), []))
        self.status = (
            "Klik één punt of hoedje op de VERRE zijlijn, of druk S wanneer die niet zichtbaar is."
            if count == 0 else
            "Klik één punt of hoedje op de NABIJE zijlijn, of druk S wanneer die niet zichtbaar is."
            if count == 1 else
            "Deze stand is compleet. Druk N voor de volgende stand."
        )

    def _mouse(self, event: int, x: int, y: int, flags: int, _data: object) -> None:
        if self.frame is None:
            return
        if event == cv2.EVENT_MOUSEWHEEL:
            point = self._canvas_to_frame(x, y)
            if point is not None:
                self.zoom_center = point
                delta = np.int16((flags >> 16) & 0xFFFF).item()
                self.zoom = min(8.0, self.zoom * 1.25) if delta > 0 else max(1.0, self.zoom * 0.8)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        point = self._canvas_to_frame(x, y)
        if point is None:
            return
        points = self.points.setdefault(self._state_id(), [])
        if len(points) < 2:
            points.append(point)
        self._set_instruction()

    def _render(self) -> np.ndarray:
        assert self.frame is not None
        canvas = np.full((900, 1600, 3), 24, np.uint8)
        state = self.states[self.index]
        current_points = self.points.get(self._state_id(), [])
        def observation_status(index: int) -> str:
            if len(current_points) <= index:
                return "nog nodig"
            return "niet zichtbaar" if current_points[index] is None else "opgeslagen"
        completed = sum(len(self.points.get(int(item["camera_state"]), [])) == 2 for item in self.states)
        lines = (
            "TUSSENANKERS ZIJLIJNEN",
            f"Stand {self.index + 1}/{len(self.states)} | camera {self._state_id()}",
            f"Tijd {float(state['representative_time_seconds']):.1f}s",
            f"Compleet: {completed}/{len(self.states)}",
            f"Verre zijlijn: {observation_status(0)}",
            f"Nabije zijlijn: {observation_status(1)}",
            "",
            "KLIK 1: VERRE zijlijn",
            "De lijn aan de overkant van het veld.",
            "Klik een zichtbaar hoedje of lijnpunt.",
            "",
            "KLIK 2: NABIJE zijlijn",
            "De lijn aan de kant van de camera.",
            "Klik een zichtbaar hoedje of lijnpunt.",
            "",
            "Klik NIET op de middenlijn.",
            "Klik NIET op een strafschopgebied.",
            "S: huidige zijlijn is NIET ZICHTBAAR",
            "",
            "P/N of pijlen: vorige/volgende",
            "U: laatste klik ongedaan",
            "Muiswiel of +/-: zoom",
            "0: volledig beeld",
            "Enter: opslaan als alles compleet is",
            "Esc: afbreken",
        )
        for index, text in enumerate(lines):
            color = (0, 230, 255) if text.startswith("KLIK") else (235, 235, 235)
            cv2.putText(canvas, text, (18, 34 + 28 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        for index, text in enumerate(self._wrap(self.status, 43)):
            cv2.putText(canvas, text, (18, 820 + 25 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
        self._draw_image(canvas)
        return canvas

    def _draw_image(self, canvas: np.ndarray) -> None:
        assert self.frame is not None
        annotated = self.frame.copy()
        points = self.points.get(self._state_id(), [])
        labels = ("VERRE ZIJLIJN", "NABIJE ZIJLIJN")
        colors = ((255, 255, 0), (0, 165, 255))
        for index, point in enumerate(points):
            if point is None:
                continue
            position = tuple(np.round(point).astype(int))
            cv2.drawMarker(annotated, position, colors[index], cv2.MARKER_CROSS, 22, 3, cv2.LINE_AA)
            cv2.putText(annotated, labels[index], (position[0] + 12, position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[index], 2, cv2.LINE_AA)
        x0, y0, width, height = self.viewport_rect
        frame_height, frame_width = annotated.shape[:2]
        crop_width, crop_height = frame_width / self.zoom, frame_height / self.zoom
        center = self.zoom_center or (frame_width / 2.0, frame_height / 2.0)
        left = float(np.clip(center[0] - crop_width / 2.0, 0.0, frame_width - crop_width))
        top = float(np.clip(center[1] - crop_height / 2.0, 0.0, frame_height - crop_height))
        crop = annotated[int(top):int(round(top + crop_height)), int(left):int(round(left + crop_width))]
        scale = min(width / crop.shape[1], height / crop.shape[0])
        shown = cv2.resize(crop, (int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))))
        px = x0 + (width - shown.shape[1]) // 2
        py = y0 + (height - shown.shape[0]) // 2
        canvas[py:py + shown.shape[0], px:px + shown.shape[1]] = shown
        self.image_rect = (px, py, shown.shape[1], shown.shape[0])
        self.view_origin, self.view_scale = (left, top), scale

    def _canvas_to_frame(self, x: int, y: int) -> tuple[float, float] | None:
        left, top, width, height = self.image_rect
        if not (left <= x < left + width and top <= y < top + height):
            return None
        return (
            self.view_origin[0] + (x - left) / self.view_scale,
            self.view_origin[1] + (y - top) / self.view_scale,
        )

    @staticmethod
    def _wrap(text: str, width: int) -> list[str]:
        lines, current = [], ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if len(candidate) > width and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines[-3:]

    def _close(self) -> None:
        cv2.destroyWindow(self.WINDOW)
        self.capture.release()
