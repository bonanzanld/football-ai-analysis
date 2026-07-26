from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import cv2
import numpy as np

from football_ai.calibration.bootstrap.goal_detection import measure_backline_support


def fit_average_support_line(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Fit one ground line while allowing normal click and marking-width variation."""
    samples = np.asarray(points, dtype=np.float64)
    if samples.shape != (3, 2):
        raise ValueError("Geef precies drie globale aanwijzingen voor de gemiddelde lijn.")
    center = samples.mean(axis=0)
    _, singular_values, axes = np.linalg.svd(samples - center, full_matrices=False)
    if singular_values[0] < 40.0:
        raise ValueError("De aanwijzingen liggen te dicht bij elkaar. Kies punten verder uit elkaar.")
    direction = axes[0]
    positions = (samples - center) @ direction
    start = center + positions.min() * direction
    end = center + positions.max() * direction
    residuals = (samples - center) @ axes[1]
    rms_error = float(np.sqrt(np.mean(np.square(residuals))))
    return tuple(map(float, start)), tuple(map(float, end)), rms_error


@dataclass(frozen=True, slots=True)
class GoalSeed:
    goal_id: str
    frame_number: int
    time_seconds: float
    camera_state: int
    view_position: float
    first_ground: tuple[float, float]
    second_ground: tuple[float, float]
    goal_width_m: float
    backline_support: float
    rear_corner: tuple[float, float] | None = None
    front_corner: tuple[float, float] | None = None
    rear_sideline_support: tuple[float, float] | None = None
    front_sideline_support: tuple[float, float] | None = None
    front_sideline_support_end: tuple[float, float] | None = None
    front_sideline_observations: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "frame_number": self.frame_number,
            "time_seconds": self.time_seconds,
            "camera_state": self.camera_state,
            "view_position": self.view_position,
            "first_ground": list(self.first_ground),
            "second_ground": list(self.second_ground),
            "goal_width_m": self.goal_width_m,
            "backline_support": self.backline_support,
            "rear_corner": list(self.rear_corner) if self.rear_corner is not None else None,
            "front_corner": list(self.front_corner) if self.front_corner is not None else None,
            "rear_sideline_support": list(self.rear_sideline_support) if self.rear_sideline_support is not None else None,
            "front_sideline_support": list(self.front_sideline_support) if self.front_sideline_support is not None else None,
            "front_sideline_support_end": list(self.front_sideline_support_end) if self.front_sideline_support_end is not None else None,
            "front_sideline_observations": [list(point) for point in self.front_sideline_observations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoalSeed":
        def optional_point(name: str) -> tuple[float, float] | None:
            value = data.get(name)
            return tuple(map(float, value)) if value is not None else None

        return cls(
            goal_id=str(data["goal_id"]),
            frame_number=int(data["frame_number"]),
            time_seconds=float(data["time_seconds"]),
            camera_state=int(data["camera_state"]),
            view_position=float(data["view_position"]),
            first_ground=tuple(map(float, data["first_ground"])),
            second_ground=tuple(map(float, data["second_ground"])),
            goal_width_m=float(data["goal_width_m"]),
            backline_support=float(data["backline_support"]),
            rear_corner=optional_point("rear_corner"),
            front_corner=optional_point("front_corner"),
            rear_sideline_support=optional_point("rear_sideline_support"),
            front_sideline_support=optional_point("front_sideline_support"),
            front_sideline_support_end=optional_point("front_sideline_support_end"),
            front_sideline_observations=tuple(
                tuple(map(float, point))
                for point in data.get("front_sideline_observations", ())
            ),
        )


def build_goal_sample_window(
    goal_times: tuple[float, float],
    *,
    fps: float,
    frame_count: int,
    radius_seconds: float = 12.0,
    step_seconds: float = 2.0,
) -> list[dict]:
    """Create a small navigable sample set around known left/right views."""
    if fps <= 0.0 or frame_count <= 0 or step_seconds <= 0.0:
        raise ValueError("Ongeldige videogegevens voor alternatieve doelsamples.")
    duration = (frame_count - 1) / fps
    samples = []
    state = 0
    for view_position, center in zip((0.0, 1.0), goal_times):
        offsets = np.arange(-radius_seconds, radius_seconds + 0.5 * step_seconds, step_seconds)
        for offset in offsets:
            time_seconds = float(np.clip(center + offset, 0.0, duration))
            samples.append(
                {
                    "frame_number": int(round(time_seconds * fps)),
                    "time_seconds": time_seconds,
                    "camera_state": state,
                    "view_position": view_position,
                }
            )
            state += 1
    unique = {item["frame_number"]: item for item in samples}
    return sorted(unique.values(), key=lambda item: (item["view_position"], item["time_seconds"]))


class GoalSeedApp:
    WINDOW = "Football AI - optionele doelbevestiging"

    def __init__(
        self,
        video_path: Path,
        bootstrap_path: Path,
        goal_width_m: float = 5.0,
        match_format: str = "8v8",
        fallback_goal_times: tuple[float, float] | None = None,
    ) -> None:
        self.video_path = video_path
        self.goal_width_m = goal_width_m
        self.match_format = match_format
        if bootstrap_path.exists():
            report = json.loads(bootstrap_path.read_text(encoding="utf-8"))
            positions = {
                int(item["camera_state"]): float(item.get("view_position", 0.5))
                for item in report["camera_states"]
            }
            self.samples = [
                {**item, "view_position": positions[int(item["camera_state"])]}
                for item in report["samples"]
                if positions[int(item["camera_state"])] <= 0.30
                or positions[int(item["camera_state"])] >= 0.70
            ]
        elif fallback_goal_times is not None:
            probe = cv2.VideoCapture(str(video_path))
            if not probe.isOpened():
                raise FileNotFoundError(f"Video kon niet worden geopend: {video_path}")
            fps = float(probe.get(cv2.CAP_PROP_FPS))
            frame_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
            probe.release()
            self.samples = build_goal_sample_window(
                fallback_goal_times, fps=fps, frame_count=frame_count
            )
        else:
            raise FileNotFoundError(
                f"Bootstrap ontbreekt: {bootstrap_path}. Geef alternatieve doeltijden door."
            )
        if not self.samples:
            raise RuntimeError("Bootstrap bevat geen uiterste camerastanden.")
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Video kon niet worden geopend: {video_path}")
        self.sample_index = 0
        self.frame: np.ndarray | None = None
        self.mode = "A"
        self.target = "posts"
        self.pending: list[tuple[float, float]] = []
        self.seeds: dict[str, GoalSeed] = {}
        self.zoom = 1.0
        self.zoom_center: tuple[float, float] | None = None
        self.image_rect = (400, 0, 1200, 900)
        self.view_origin = (0.0, 0.0)
        self.view_scale = 1.0
        self.status = (
            "Kies A voor het LINKER wedstrijdgoal of B voor het RECHTER wedstrijdgoal, "
            "allebei gezien vanuit de camera."
        )
        self.click_feedback: tuple[tuple[float, float], bool, str] | None = None
        self._load_sample()
    def run(self) -> tuple[GoalSeed, GoalSeed]:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 1600, 900)
        cv2.setMouseCallback(self.WINDOW, self._mouse)
        while True:
            cv2.imshow(self.WINDOW, self._render())
            key = cv2.waitKeyEx(30)
            if key == 27:
                cv2.destroyWindow(self.WINDOW)
                self.capture.release()
                raise RuntimeError("Doelbevestiging afgebroken.")
            if key in (2424832, 65361, ord("p"), ord("P")):
                self._move(-1)
            elif key in (2555904, 65363, ord("n"), ord("N")):
                self._move(1)
            elif key in (ord("a"), ord("A")):
                self.mode, self.target, self.pending = "A", "posts", []
                self.click_feedback = None
                self.status = (
                    "DOEL A = LINKS VANUIT DE CAMERA. Klik eerst het grondcontact van de "
                    "VERSTE paal en daarna dat van de DICHTSTBIJZIJNDE paal."
                )
            elif key in (ord("b"), ord("B")):
                self.mode, self.target, self.pending = "B", "posts", []
                self.click_feedback = None
                self.status = (
                    "DOEL B = RECHTS VANUIT DE CAMERA. Klik eerst het grondcontact van de "
                    "VERSTE paal en daarna dat van de DICHTSTBIJZIJNDE paal."
                )
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                corner_targets = {
                    ord("1"): ("A", "rear", "LINKSACHTER"),
                    ord("2"): ("A", "front", "LINKSVOOR"),
                    ord("3"): ("B", "rear", "RECHTSACHTER"),
                    ord("4"): ("B", "front", "RECHTSVOOR"),
                }
                self.mode, self.target, label = corner_targets[key]
                self.pending = []
                self.click_feedback = None
                self.status = f"{label}: klik het midden van de hoekmarkering."
            elif key in (ord("5"), ord("6")):
                if key == ord("5"):
                    self.status = (
                        "Geen klik nodig: de VERRE zijlijn wordt automatisch uit het "
                        "grondvlak en de andere achterlijn berekend."
                    )
                    continue
                self.target = "front_support"
                self.pending = []
                label = "NABIJE 5,5M-/HOEDJESLIJN"
                self.status = (
                    f"Doel {self.mode} | {label}: klik "
                    + ("een zichtbaar lijnpunt of hoedje." if self.target == "rear_support" else
                       "DRIE globale aanwijzingen op de brede 5,5m-/hoedjeslijn.")
                )
            elif key in (8, 127, ord("u"), ord("U")):
                if self.pending:
                    self.pending.pop()
                elif self.target in ("rear", "front", "rear_support", "front_support") and self.mode in self.seeds:
                    field_name = {
                        "rear": "rear_corner",
                        "front": "front_corner",
                        "rear_support": "rear_sideline_support",
                        "front_support": "front_sideline_support",
                    }[self.target]
                    changes = {field_name: None}
                    if self.target == "front_support":
                        changes["front_sideline_support_end"] = None
                    self.seeds[self.mode] = replace(self.seeds[self.mode], **changes)
                else:
                    self.seeds.pop(self.mode, None)
            elif key in (ord("+"), ord("=")):
                self.zoom = min(8.0, self.zoom * 1.25)
            elif key in (ord("-"), ord("_")):
                self.zoom = max(1.0, self.zoom * 0.8)
            elif key == ord("0"):
                self.zoom, self.zoom_center = 1.0, None
            elif key in (10, 13):
                complete = {"A", "B"} <= self.seeds.keys() and all(
                    (seed.rear_corner is not None or seed.front_corner is not None)
                    and seed.front_sideline_support is not None
                    and seed.front_sideline_support_end is not None
                    for seed in self.seeds.values()
                )
                if complete:
                    cv2.destroyWindow(self.WINDOW)
                    self.capture.release()
                    return self.seeds["A"], self.seeds["B"]
                self.status = (
                    "Per doel zijn palen, minimaal een hoek en DRIE aanwijzingen op de "
                    "nabije 5,5m-/hoedjeslijn vereist."
                )

    def _move(self, amount: int) -> None:
        self.sample_index = (self.sample_index + amount) % len(self.samples)
        self.pending = []
        self.click_feedback = None
        self.zoom, self.zoom_center = 1.0, None
        self._load_sample()

    def _load_sample(self) -> None:
        sample = self.samples[self.sample_index]
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(sample["frame_number"]))
        success, frame = self.capture.read()
        if not success:
            raise RuntimeError(f"Frame {sample['frame_number']} kon niet worden gelezen.")
        self.frame = frame

    def _mouse(self, event: int, x: int, y: int, flags: int, _data: object) -> None:
        if self.frame is None:
            return
        if event == cv2.EVENT_MOUSEWHEEL:
            point = self._canvas_to_frame(x, y)
            if point is not None:
                self.zoom_center = point
                delta = np.int16((flags >> 16) & 0xFFFF).item()
                if delta > 0:
                    self.zoom = min(8.0, self.zoom * 1.25)
                elif delta < 0:
                    self.zoom = max(1.0, self.zoom * 0.8)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        point = self._canvas_to_frame(x, y)
        if point is None:
            return
        if self.target in ("rear", "front", "rear_support", "front_support"):
            seed = self.seeds.get(self.mode)
            if seed is None:
                self.status = f"Klik eerst de twee palen van Doel {self.mode}."
                self.click_feedback = (point, False, "Eerst doelpalen")
                return
            sample_frame = int(self.samples[self.sample_index]["frame_number"])
            if seed.frame_number != sample_frame:
                self.status = "Kies voor dit hoekpunt hetzelfde sample als de doelpalen."
                self.click_feedback = (point, False, "Verkeerd sample")
                return
            if self.target in ("rear_support", "front_support"):
                first = np.asarray(seed.first_ground, dtype=np.float64)
                second = np.asarray(seed.second_ground, dtype=np.float64)
                direction = second - first
                offset = np.asarray(point) - first
                cross_value = direction[0] * offset[1] - direction[1] * offset[0]
                distance = abs(float(cross_value)) / max(float(np.linalg.norm(direction)), 1e-6)
                if distance < 8.0:
                    self.status = "Dit punt ligt nog op de achterlijn. Klik verderop langs de zijlijn."
                    self.click_feedback = (point, False, "Nog op achterlijn")
                    return
            if self.target == "front_support":
                self.pending.append(point)
                click_count = len(self.pending)
                if click_count < 3:
                    self.click_feedback = (point, True, f"Aanwijzing {click_count} opgeslagen")
                    self.status = (
                        f"Aanwijzing {click_count}/3 opgeslagen. Klik nog een globaal punt "
                        "op dezelfde brede kalklijn of hoedjeslijn; pixelprecisie is niet nodig."
                    )
                    return
                observations = tuple(self.pending)
                self.pending = []
                try:
                    first_support, last_support, rms_error = fit_average_support_line(
                        observations
                    )
                except ValueError as error:
                    self.status = str(error)
                    self.click_feedback = (point, False, "Opnieuw")
                    return
                self.seeds[self.mode] = replace(
                    seed,
                    front_sideline_support=first_support,
                    front_sideline_support_end=last_support,
                    front_sideline_observations=observations,
                )
                self.click_feedback = (point, True, "Gemiddelde lijn opgeslagen")
                other = "B" if self.mode == "A" else "A"
                self.status = (
                    f"Gemiddelde 5,5m-/hoedjeslijn bij Doel {self.mode} opgeslagen "
                    f"(spreiding {rms_error:.1f}px). "
                    f"Kies Doel {other}, of Enter wanneer alles compleet is."
                )
                return
            field_name = {
                "rear": "rear_corner",
                "front": "front_corner",
                "rear_support": "rear_sideline_support",
                "front_support": "front_sideline_support",
            }[self.target]
            self.seeds[self.mode] = replace(seed, **{field_name: point})
            self.click_feedback = (point, True, "Opgeslagen")
            label = {
                ("A", "rear"): "LINKSACHTER",
                ("A", "front"): "LINKSVOOR",
                ("B", "rear"): "RECHTSACHTER",
                ("B", "front"): "RECHTSVOOR",
                ("A", "rear_support"): "ACHTERSTE ZIJLIJN BIJ DOEL A",
                ("A", "front_support"): "VOORSTE ZIJLIJN BIJ DOEL A",
                ("B", "rear_support"): "ACHTERSTE ZIJLIJN BIJ DOEL B",
                ("B", "front_support"): "VOORSTE ZIJLIJN BIJ DOEL B",
            }[(self.mode, self.target)]
            if self.target in ("rear", "front"):
                self.target = "front_support"
                self.status = (
                    f"{label} opgeslagen. Geef nu DRIE globale aanwijzingen op de "
                    "NABIJE brede 5,5m-lijn en/of dezelfde rij hoedjes."
                )
            else:
                other = "B" if self.mode == "A" else "A"
                self.status = f"Beide zijlijnen bij Doel {self.mode} opgeslagen. Kies Doel {other}, of Enter wanneer alles compleet is."
            return
        self.pending.append(point)
        if len(self.pending) == 2:
            sample = self.samples[self.sample_index]
            support = measure_backline_support(self.frame, self.pending[0], self.pending[1])
            self.seeds[self.mode] = GoalSeed(
                self.mode,
                int(sample["frame_number"]),
                float(sample["time_seconds"]),
                int(sample["camera_state"]),
                float(sample["view_position"]),
                self.pending[0],
                self.pending[1],
                self.goal_width_m,
                support,
            )
            self.pending = []
            corner_keys = "1/2" if self.mode == "A" else "3/4"
            self.status = (
                f"Doel {self.mode} opgeslagen | witte achterlijnsteun {support:.0%}. "
                + (
                    f"WAARSCHUWING: steun is laag. Kies nu hoek {corner_keys}."
                    if support < 0.25
                    else f"Kies nu minimaal een hoek met toets {corner_keys}."
                )
            )

    def _render(self) -> np.ndarray:
        canvas = np.full((900, 1600, 3), 25, np.uint8)
        sample = self.samples[self.sample_index]

        def seed_status(goal_id: str) -> str:
            seed = self.seeds.get(goal_id)
            if seed is None:
                return "palen nog nodig"
            rear = "A" if seed.rear_corner is not None else "-"
            front = "V" if seed.front_corner is not None else "-"
            near_line = "V2" if seed.front_sideline_support_end is not None else "-"
            return f"palen | hoek {rear}/{front} | nabije lijn {near_line}"

        lines = [
            f"DOELBEVESTIGING {self.match_format}",
            "A = LINKER wedstrijdgoal vanuit camera",
            "B = RECHTER wedstrijdgoal vanuit camera",
            "Dit blijft gelden terwijl de camera draait.",
            "",
            f"Sample {self.sample_index + 1}/{len(self.samples)}",
            f"Tijd {float(sample['time_seconds']):.1f}s | stand {sample['camera_state']}",
            f"NU: {self._goal_label(self.mode)}",
            f"KLIK: {self._target_label()}",
            f"Doel A: {seed_status('A')}",
            f"Doel B: {seed_status('B')}",
            "",
            "P/N of pijlen: ander sample",
            "A: selecteer LINKER wedstrijdgoal",
            "B: selecteer RECHTER wedstrijdgoal",
            "Palen: eerst VERST, daarna DICHTSTBIJ",
            "Klik waar iedere paal de grond raakt",
            f"Kies het {self.goal_width_m:g}m WEDSTRIJDDOEL",
            "Negeer reserve-/trainingsdoelen",
            "1 = LINKER goal, verste veldhoek",
            "2 = LINKER goal, hoek aan camerazijde",
            "3 = RECHTER goal, verste veldhoek",
            "4 = RECHTER goal, hoek aan camerazijde",
            "Minimaal 1 hoek per doelzijde",
            "Na hoek volgt automatisch:",
            "DRIE globale punten op nabije lijn",
            "Hoedjes en brede kalklijn mogen gecombineerd",
            "Verre zijlijn wordt automatisch berekend",
            "Muiswiel of +/-: zoom",
            "0: volledig beeld",
            "U: ongedaan maken",
            "Enter: afronden",
        ]
        for index, text in enumerate(lines):
            color = (0, 230, 255) if text.startswith(("A =", "B =", "NU:", "KLIK:")) else (240, 240, 240)
            cv2.putText(canvas, text, (20, 34 + index * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        status_words = self.status.split()
        status_lines: list[str] = []
        current = ""
        for word in status_words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > 42 and current:
                status_lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            status_lines.append(current)
        for index, status_line in enumerate(status_lines[-3:]):
            cv2.putText(
                canvas,
                status_line,
                (20, 810 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
        self._draw_frame(canvas)
        return canvas

    def _target_label(self) -> str:
        if self.target == "posts":
            return (
                "PAAL 1: grondcontact VERSTE paal"
                if len(self.pending) == 0 else
                "PAAL 2: grondcontact DICHTSTBIJZIJNDE paal"
            )
        return {
            "rear": "ACHTERSTE veldhoek (verste zijde)",
            "front": "VOORSTE veldhoek (camera-zijde)",
            "rear_support": "punt op VERRE zijlijn",
            "front_support": (
                f"NABIJE 5,5m-/hoedjeslijn: AANWIJZING {len(self.pending) + 1}/3"
            ),
        }.get(self.target, self.target)

    @staticmethod
    def _goal_label(goal_id: str) -> str:
        return "DOEL A = LINKS vanuit camera" if goal_id == "A" else "DOEL B = RECHTS vanuit camera"

    def _draw_frame(self, canvas: np.ndarray) -> None:
        assert self.frame is not None
        annotated = self.frame.copy()
        sample_frame_number = int(self.samples[self.sample_index]["frame_number"])
        visible_points: list[tuple[tuple[float, float], str]] = [
            (point, self.mode) for point in self.pending
        ]
        for goal_id, seed in self.seeds.items():
            if seed.frame_number != sample_frame_number:
                continue
            visible_points.extend(
                ((seed.first_ground, goal_id), (seed.second_ground, goal_id))
            )
            if seed.rear_corner is not None:
                visible_points.append((seed.rear_corner, "LA" if goal_id == "A" else "RA"))
            if seed.front_corner is not None:
                visible_points.append((seed.front_corner, "LV" if goal_id == "A" else "RV"))
            if seed.rear_sideline_support is not None:
                visible_points.append((seed.rear_sideline_support, "ZA"))
            if seed.front_sideline_support is not None:
                visible_points.append((seed.front_sideline_support, "ZV"))
            if seed.front_sideline_support_end is not None:
                visible_points.append((seed.front_sideline_support_end, "ZV2"))
                cv2.line(
                    annotated,
                    tuple(np.round(seed.front_sideline_support).astype(int)),
                    tuple(np.round(seed.front_sideline_support_end).astype(int)),
                    (255, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
            if seed.rear_corner is not None or seed.front_corner is not None:
                rear_end, front_end = estimate_backline_endpoints(
                    seed.first_ground,
                    seed.second_ground,
                    seed.goal_width_m,
                    42.5,
                    seed.rear_corner,
                    seed.front_corner,
                )
                cv2.line(
                    annotated,
                    tuple(np.round(rear_end).astype(int)),
                    tuple(np.round(front_end).astype(int)),
                    (0, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                for endpoint in (rear_end, front_end):
                    cv2.drawMarker(
                        annotated,
                        tuple(np.round(endpoint).astype(int)),
                        (0, 255, 255),
                        cv2.MARKER_TILTED_CROSS,
                        18,
                        3,
                        cv2.LINE_AA,
                    )
                for corner, support in (
                    (rear_end, seed.rear_sideline_support),
                    (front_end, seed.front_sideline_support),
                ):
                    if support is not None:
                        cv2.line(
                            annotated,
                            tuple(np.round(corner).astype(int)),
                            tuple(np.round(support).astype(int)),
                            (255, 255, 0),
                            3,
                            cv2.LINE_AA,
                        )
                if self.mode == goal_id and self.target in ("rear_support", "front_support"):
                    active_corner = rear_end if self.target == "rear_support" else front_end
                    active_label = "ACHTERSTE" if self.target == "rear_support" else "VOORSTE"
                    active_px = tuple(np.round(active_corner).astype(int))
                    cv2.circle(annotated, active_px, 24, (255, 255, 0), 4, cv2.LINE_AA)
                    cv2.rectangle(annotated, (12, 58), (760, 108), (20, 20, 20), -1)
                    cv2.putText(
                        annotated,
                        f"{active_label} ZIJLIJN: klik vanaf de omcirkelde hoek richting het andere doel",
                        (24, 91),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.63,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
            cv2.line(
                annotated,
                tuple(np.round(seed.first_ground).astype(int)),
                tuple(np.round(seed.second_ground).astype(int)),
                (255, 0, 255),
                3,
                cv2.LINE_AA,
            )
        for point, goal_id in visible_points:
            center = tuple(np.round(point).astype(int))
            cv2.circle(annotated, center, 10, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(annotated, center, 14, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                annotated,
                goal_id,
                (center[0] + 14, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if self.click_feedback is not None:
            point, accepted, reason = self.click_feedback
            center = tuple(np.round(point).astype(int))
            color = (255, 255, 0) if accepted else (0, 0, 255)
            cv2.drawMarker(annotated, center, color, cv2.MARKER_TILTED_CROSS, 28, 4, cv2.LINE_AA)
            cv2.putText(annotated, reason, (center[0] + 16, center[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        height, width = annotated.shape[:2]
        view_width, view_height = width / self.zoom, height / self.zoom
        center = self.zoom_center or (width / 2.0, height / 2.0)
        origin_x = min(max(0.0, center[0] - view_width / 2.0), width - view_width)
        origin_y = min(max(0.0, center[1] - view_height / 2.0), height - view_height)
        crop = annotated[int(origin_y):int(origin_y + view_height), int(origin_x):int(origin_x + view_width)]
        scale = min(1200 / crop.shape[1], 900 / crop.shape[0])
        shown = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
        x, y = 400, (900 - shown.shape[0]) // 2
        canvas[y:y + shown.shape[0], x:x + shown.shape[1]] = shown
        self.image_rect = (x, y, shown.shape[1], shown.shape[0])
        self.view_origin, self.view_scale = (origin_x, origin_y), scale

    def _canvas_to_frame(self, x: int, y: int) -> tuple[float, float] | None:
        rx, ry, rw, rh = self.image_rect
        if not (rx <= x < rx + rw and ry <= y < ry + rh):
            return None
        return (
            self.view_origin[0] + (x - rx) / self.view_scale,
            self.view_origin[1] + (y - ry) / self.view_scale,
        )


def save_goal_seeds(
    seeds: tuple[GoalSeed, GoalSeed],
    path: Path,
    match_format: str = "8v8",
) -> None:
    from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
    from football_ai.calibration.bootstrap.seeded_field_contour import build_seeded_field_contour

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = create_detection_profile(match_format)
    contour = build_seeded_field_contour(seeds, profile)
    path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "match_format": profile.match_format.value,
                "goals": [item.to_dict() for item in seeds],
                "field_contour": contour.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_goal_seeds(path: Path) -> tuple[GoalSeed, GoalSeed]:
    data = json.loads(path.read_text(encoding="utf-8"))
    seeds = tuple(GoalSeed.from_dict(item) for item in data["goals"])
    if len(seeds) != 2:
        raise ValueError("Doel-seedbestand moet precies twee doelen bevatten.")
    return seeds


def estimate_backline_endpoints(
    first_post: tuple[float, float],
    second_post: tuple[float, float],
    goal_width_m: float,
    pitch_width_m: float,
    rear_corner: tuple[float, float] | None = None,
    front_corner: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Project both backline ends using ordered posts and optional known corners.

    The first post must be the rear-side post and the second the camera-side post.
    One observed corner is enough to solve the 1D projective mapping. Without a
    corner, the function retains the old local-affine estimate for compatibility.
    """
    if goal_width_m <= 0.0 or pitch_width_m <= goal_width_m:
        raise ValueError("Veld- en doelbreedte moeten positief en logisch zijn.")
    if rear_corner is not None and front_corner is not None:
        return rear_corner, front_corner
    first = np.asarray(first_post, dtype=np.float64)
    second = np.asarray(second_post, dtype=np.float64)
    goal_vector = second - first
    if np.linalg.norm(goal_vector) < 1e-6:
        raise ValueError("De twee doelpaalpunten mogen niet samenvallen.")
    if rear_corner is None and front_corner is None:
        centre = (first + second) / 2.0
        half_backline = goal_vector * (pitch_width_m / goal_width_m) / 2.0
        rear = centre - half_backline
        front = centre + half_backline
        return tuple(rear.tolist()), tuple(front.tolist())

    line_length = float(np.linalg.norm(goal_vector))
    line_direction = goal_vector / line_length
    rear_post_m = (pitch_width_m - goal_width_m) / 2.0
    front_post_m = rear_post_m + goal_width_m
    observations = [(rear_post_m, 0.0), (front_post_m, line_length)]
    if rear_corner is not None:
        observations.append((0.0, float(np.dot(np.asarray(rear_corner) - first, line_direction))))
    if front_corner is not None:
        observations.append((pitch_width_m, float(np.dot(np.asarray(front_corner) - first, line_direction))))
    matrix = np.asarray([[metres, 1.0, -pixels * metres] for metres, pixels in observations])
    target = np.asarray([pixels for _, pixels in observations])
    a, b, c = np.linalg.lstsq(matrix, target, rcond=None)[0]

    def project(metres: float) -> tuple[float, float]:
        denominator = c * metres + 1.0
        if abs(denominator) < 1e-8:
            raise ValueError("Hoekpuntconfiguratie leidt tot een instabiele projectie.")
        pixels = (a * metres + b) / denominator
        return tuple((first + line_direction * pixels).tolist())

    return project(0.0), project(pitch_width_m)


def create_goal_seed_preview(
    video_path: Path,
    seeds: tuple[GoalSeed, GoalSeed],
    pitch_width_m: float = 42.5,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    tiles: list[np.ndarray] = []
    for seed in seeds:
        capture.set(cv2.CAP_PROP_POS_FRAMES, seed.frame_number)
        success, frame = capture.read()
        if not success:
            continue
        backline_start, backline_end = estimate_backline_endpoints(
            seed.first_ground,
            seed.second_ground,
            seed.goal_width_m,
            pitch_width_m,
            seed.rear_corner,
            seed.front_corner,
        )
        cv2.line(
            frame,
            tuple(np.round(backline_start).astype(int)),
            tuple(np.round(backline_end).astype(int)),
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
        for endpoint in (backline_start, backline_end):
            cv2.drawMarker(
                frame,
                tuple(np.round(endpoint).astype(int)),
                (0, 255, 255),
                cv2.MARKER_TILTED_CROSS,
                18,
                3,
                cv2.LINE_AA,
            )
        for point in (seed.first_ground, seed.second_ground):
            cv2.circle(frame, tuple(np.round(point).astype(int)), 8, (255, 0, 255), -1, cv2.LINE_AA)
        corner_labels = (
            (("linksachter", seed.rear_corner), ("linksvoor", seed.front_corner))
            if seed.goal_id == "A"
            else (("rechtsachter", seed.rear_corner), ("rechtsvoor", seed.front_corner))
        )
        for label, point in corner_labels:
            if point is None:
                continue
            centre = tuple(np.round(point).astype(int))
            cv2.circle(frame, centre, 9, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, label, (centre[0] + 10, centre[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
        for label, corner, support in (
            ("zijlijn achter", backline_start, seed.rear_sideline_support),
            ("zijlijn voor", backline_end, seed.front_sideline_support),
        ):
            if support is None:
                continue
            corner_px = tuple(np.round(corner).astype(int))
            support_px = tuple(np.round(support).astype(int))
            cv2.line(frame, corner_px, support_px, (255, 255, 0), 3, cv2.LINE_AA)
            cv2.circle(frame, support_px, 8, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(frame, label, (support_px[0] + 10, support_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"DOEL {seed.goal_id} | frame {seed.frame_number} | lijnsteun {seed.backline_support:.0%} | gele lijn ~ {pitch_width_m:g}m",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(cv2.resize(frame, (640, 360)))
    capture.release()
    if len(tiles) != 2:
        raise RuntimeError("QA-preview vereist twee leesbare doelframes.")
    return np.hstack(tiles)
