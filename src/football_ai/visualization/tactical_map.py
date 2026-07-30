from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from football_ai.analysis.entity_timeline import TimelineEntity
from football_ai.analysis.possession import PossessionObservation, PossessionState
from football_ai.detection.ball_tracking import BallObservation
from football_ai.tracking.entity_corrections import EntityRole, TeamAssignment


TEAM_COLORS = {
    TeamAssignment.TEAM_A: (255, 130, 20),
    TeamAssignment.TEAM_B: (30, 70, 255),
}


@dataclass(frozen=True, slots=True)
class MapPoint:
    x: float
    y: float


class CoordinateProjector(Protocol):
    """Verwisselbare omzetting van beeldpunten naar een genormaliseerde veldkaart."""

    name: str
    metric: bool

    def project(self, point: tuple[float, float], frame_size: tuple[int, int]) -> MapPoint:
        ...


@dataclass(frozen=True, slots=True)
class CameraRelativeProjector:
    """Niet-metrische kaart voor video's waarvoor nog geen veldprojectie beschikbaar is."""

    horizon_ratio: float = 0.30
    depth_exponent: float = 1.22
    name: str = "camera-relatief"
    metric: bool = False

    def project(self, point: tuple[float, float], frame_size: tuple[int, int]) -> MapPoint:
        width, height = frame_size
        x = float(np.clip(point[0] / max(width, 1), 0.0, 1.0))
        horizon = self.horizon_ratio * height
        depth = (point[1] - horizon) / max(height - horizon, 1.0)
        y = float(np.clip(depth, 0.0, 1.0) ** self.depth_exponent)
        return MapPoint(x, y)


@dataclass
class GoalkeeperAnchoredProjector:
    """Tijdelijke, niet-metrische kaart die zichtbare keepers als doelanker gebruikt.

    De omzetting bewaart de volgorde van alle beeldpunten. Een keeper links of
    rechts in beeld wordt nabij het bijbehorende doel geplaatst; zijn diepte
    wordt rond het midden van dat doel verankerd. Zonder zichtbare keeper valt
    de projector terug op de gewone camera-relatieve omzetting.
    """

    camera_height_m: float = 3.75
    goal_offset: float = 0.055
    goal_center_y: float = 0.50
    name: str = "keeper-verankerd (geschat)"
    metric: bool = False

    def __post_init__(self) -> None:
        # De hoogte is alleen een grove diepte-prior. Zonder lens- en
        # camerahoekkalibratie mag deze omzetting nadrukkelijk niet metrisch
        # worden geïnterpreteerd.
        exponent = float(np.clip(1.0 + 0.06 * self.camera_height_m, 1.12, 1.35))
        self._fallback = CameraRelativeProjector(depth_exponent=exponent)
        self._horizontal_anchors: tuple[tuple[float, float], ...] = ()
        self._depth_anchor: tuple[float, float] | None = None

    @property
    def anchored(self) -> bool:
        return bool(self._horizontal_anchors)

    def update(
        self,
        entities: Sequence[TimelineEntity],
        frame_size: tuple[int, int],
    ) -> None:
        """Gebruik uitsluitend keepers die in het huidige frame zichtbaar zijn."""

        width, _ = frame_size
        keepers = [item for item in entities if item.role is EntityRole.GOALKEEPER]
        if not keepers:
            self._horizontal_anchors = ()
            self._depth_anchor = None
            return

        normalized = sorted(
            (
                float(np.clip(item.footpoint[0] / max(width, 1), 0.0, 1.0)),
                self._fallback.project(item.footpoint, frame_size).y,
            )
            for item in keepers
        )
        left = [item for item in normalized if item[0] < 0.5]
        right = [item for item in normalized if item[0] >= 0.5]
        anchors: list[tuple[float, float]] = []
        if left:
            anchors.append((min(left, key=lambda item: item[0])[0], self.goal_offset))
        if right:
            anchors.append((max(right, key=lambda item: item[0])[0], 1.0 - self.goal_offset))
        self._horizontal_anchors = tuple(sorted(anchors))
        self._depth_anchor = (
            float(np.median([item[1] for item in normalized])),
            self.goal_center_y,
        )

    def project(self, point: tuple[float, float], frame_size: tuple[int, int]) -> MapPoint:
        raw = self._fallback.project(point, frame_size)
        if not self.anchored:
            return raw
        x = _piecewise_anchor(raw.x, self._horizontal_anchors)
        y = (
            _piecewise_anchor(raw.y, (self._depth_anchor,))
            if self._depth_anchor is not None
            else raw.y
        )
        return MapPoint(x, y)


class TacticalMapRenderer:
    def __init__(self, projector: CoordinateProjector, trail_length: int = 18) -> None:
        self.projector = projector
        self._trails: dict[tuple[str, int], deque[MapPoint]] = defaultdict(
            lambda: deque(maxlen=trail_length)
        )

    def draw(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        entities: Sequence[TimelineEntity],
        ball: BallObservation | None,
        possession: PossessionObservation,
        frame_size: tuple[int, int],
    ) -> None:
        x1, y1, x2, y2 = rect
        _draw_pitch(canvas, rect)
        cv2.putText(
            canvas,
            f"2D {self.projector.name.upper()}",
            (x1, max(18, y1 - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (205, 220, 205),
            1,
            cv2.LINE_AA,
        )

        current_keys: set[tuple[str, int]] = set()
        projected: dict[int, MapPoint] = {}
        for entity in entities:
            key = ("identity", entity.identity_id) if entity.identity_id is not None else ("track", entity.track_id)
            point = self.projector.project(entity.footpoint, frame_size)
            current_keys.add(key)
            projected[entity.track_id] = point
            self._trails[key].append(point)
            trail = self._trails[key]
            if len(trail) > 1:
                pixels = np.array([_map_pixel(item, rect) for item in trail], dtype=np.int32)
                cv2.polylines(canvas, [pixels], False, (115, 145, 115), 1, cv2.LINE_AA)

            pixel = _map_pixel(point, rect)
            team_color = TEAM_COLORS.get(entity.team, (180, 180, 180))
            if entity.role is EntityRole.GOALKEEPER:
                cv2.rectangle(
                    canvas,
                    (pixel[0] - 6, pixel[1] - 6),
                    (pixel[0] + 6, pixel[1] + 6),
                    (210, 70, 210),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.rectangle(canvas, (pixel[0] - 7, pixel[1] - 7), (pixel[0] + 7, pixel[1] + 7), team_color, 2)
            else:
                cv2.circle(canvas, pixel, 6, team_color, -1, cv2.LINE_AA)
                cv2.circle(canvas, pixel, 7, (245, 245, 245), 1, cv2.LINE_AA)
            if possession.track_id == entity.track_id:
                cv2.circle(canvas, pixel, 11, (0, 255, 255), 2, cv2.LINE_AA)

        for key in list(self._trails):
            if key not in current_keys and len(self._trails[key]) > 1:
                self._trails[key].popleft()

        ball_point: MapPoint | None = None
        inferred = False
        if ball is not None and ball.confidence >= 0.15:
            ball_point = self.projector.project(ball.center, frame_size)
        elif possession.track_id is not None and possession.state is PossessionState.INFERRED:
            ball_point = projected.get(possession.track_id)
            inferred = ball_point is not None
        if ball_point is not None:
            pixel = _map_pixel(ball_point, rect)
            if inferred:
                for start in range(0, 360, 60):
                    cv2.ellipse(canvas, pixel, (7, 7), 0, start, start + 30, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(canvas, pixel, 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, pixel, 6, (0, 0, 0), 1, cv2.LINE_AA)


class TeamHeatmapAccumulator:
    def __init__(self, projector: CoordinateProjector, grid_size: tuple[int, int] = (96, 64)) -> None:
        self.projector = projector
        self.grid_size = grid_size
        self._maps = {
            TeamAssignment.TEAM_A: np.zeros((grid_size[1], grid_size[0]), dtype=np.float32),
            TeamAssignment.TEAM_B: np.zeros((grid_size[1], grid_size[0]), dtype=np.float32),
        }

    def add(self, entities: Sequence[TimelineEntity], frame_size: tuple[int, int]) -> None:
        for entity in entities:
            heatmap = self._maps.get(entity.team)
            if heatmap is None:
                continue
            point = self.projector.project(entity.footpoint, frame_size)
            x = min(self.grid_size[0] - 1, max(0, int(round(point.x * (self.grid_size[0] - 1)))))
            y = min(self.grid_size[1] - 1, max(0, int(round(point.y * (self.grid_size[1] - 1)))))
            heatmap[y, x] += 1.0

    def save(self, output_dir: Path, prefix: str, team_names: dict[str, str]) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for team in (TeamAssignment.TEAM_A, TeamAssignment.TEAM_B):
            path = output_dir / f"{prefix}_{team.value}_camera_relative_heatmap.jpg"
            image = self._render(team, team_names.get(team.value, team.value))
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"Heatmap kon niet worden opgeslagen: {path}")
            paths.append(path)
        return paths[0], paths[1]

    def _render(self, team: TeamAssignment, name: str) -> np.ndarray:
        width, height = 960, 680
        image = np.full((height, width, 3), (18, 28, 18), dtype=np.uint8)
        rect = (55, 95, width - 55, height - 55)
        _draw_pitch(image, rect)
        heatmap = cv2.GaussianBlur(self._maps[team], (0, 0), 3.2)
        maximum = float(heatmap.max())
        if maximum > 0:
            normalized = np.uint8(np.clip(heatmap / maximum * 255.0, 0, 255))
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
            colored = cv2.resize(colored, (rect[2] - rect[0], rect[3] - rect[1]), interpolation=cv2.INTER_CUBIC)
            alpha = cv2.resize(normalized, (rect[2] - rect[0], rect[3] - rect[1]), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
            alpha = (alpha * 0.78)[..., None]
            roi = image[rect[1]:rect[3], rect[0]:rect[2]].astype(np.float32)
            image[rect[1]:rect[3], rect[0]:rect[2]] = np.uint8(roi * (1.0 - alpha) + colored * alpha)
            _draw_pitch(image, rect, lines_only=True)
        cv2.putText(image, f"HEATMAP {name}"[:48], (55, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.90, TEAM_COLORS[team], 2, cv2.LINE_AA)
        cv2.putText(image, f"{self.projector.name.capitalize()} - niet metrisch", (55, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (205, 220, 205), 1, cv2.LINE_AA)
        return image


def _piecewise_anchor(
    value: float,
    anchors: Sequence[tuple[float, float]],
) -> float:
    """Monotone stukgewijze lineaire omzetting door opgegeven ankerpunten."""

    points = [(0.0, 0.0), *sorted(anchors), (1.0, 1.0)]
    value = float(np.clip(value, 0.0, 1.0))
    for (source_a, target_a), (source_b, target_b) in zip(points, points[1:]):
        if value <= source_b:
            span = source_b - source_a
            if span <= 1e-9:
                return float(np.clip(target_b, 0.0, 1.0))
            ratio = (value - source_a) / span
            return float(np.clip(target_a + ratio * (target_b - target_a), 0.0, 1.0))
    return 1.0


def _map_pixel(point: MapPoint, rect: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = rect
    return (
        int(round(x1 + point.x * (x2 - x1))),
        int(round(y1 + point.y * (y2 - y1))),
    )


def _draw_pitch(image: np.ndarray, rect: tuple[int, int, int, int], lines_only: bool = False) -> None:
    x1, y1, x2, y2 = rect
    if not lines_only:
        cv2.rectangle(image, (x1, y1), (x2, y2), (38, 112, 49), -1)
    color = (235, 235, 235)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    middle = (x1 + x2) // 2
    cv2.line(image, (middle, y1), (middle, y2), color, 1, cv2.LINE_AA)
    radius = max(8, int(round((y2 - y1) * 0.13)))
    cv2.circle(image, (middle, (y1 + y2) // 2), radius, color, 1, cv2.LINE_AA)
    goal_depth = max(5, int(round((x2 - x1) * 0.025)))
    goal_half = max(8, int(round((y2 - y1) * 0.12)))
    center_y = (y1 + y2) // 2
    cv2.rectangle(image, (x1 - goal_depth, center_y - goal_half), (x1, center_y + goal_half), color, 2)
    cv2.rectangle(image, (x2, center_y - goal_half), (x2 + goal_depth, center_y + goal_half), color, 2)
