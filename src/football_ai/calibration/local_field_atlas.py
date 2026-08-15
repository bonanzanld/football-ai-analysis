from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from football_ai.calibration.manual_midfield_line import ManualMidfieldLine
from football_ai.calibration.manual_parallel_lines import ManualParallelLineReference
from football_ai.calibration.field_topology import BOUNDARY_CORNERS, ground_corners


BOUNDARY_NAMES = ("end_line_a", "sideline_rear", "end_line_b", "sideline_front")


@dataclass(frozen=True, slots=True)
class VisibleBoundarySegment:
    name: str
    image_start: tuple[float, float]
    image_end: tuple[float, float]
    source_patch: str
    confidence: float
    status: str


@dataclass(frozen=True, slots=True)
class VisibleFieldEvidence:
    patch_id: str
    visible_polygon: tuple[tuple[float, float], ...]
    boundary_segments: tuple[VisibleBoundarySegment, ...]
    boundary_status: dict[str, str]
    frame_coverage: float


@dataclass(frozen=True, slots=True)
class LocalFieldPatch:
    patch_id: str
    anchor_frame: int
    ground_to_anchor: np.ndarray
    support_polygon: tuple[tuple[float, float], ...]
    confidence: float
    source: str
    verified_boundaries: tuple[str, ...] = ()
    inferred_boundaries: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        matrix = np.asarray(self.ground_to_anchor, dtype=np.float64)
        polygon = np.asarray(self.support_polygon, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("Een lokaal veldvlak vereist een eindige 3x3-homography.")
        if polygon.shape[0] < 3 or polygon.shape[1:] != (2,) or not np.all(np.isfinite(polygon)):
            raise ValueError("Een lokaal veldvlak vereist een geldige steunpolygoon.")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("Atlasvertrouwen moet tussen 0 en 1 liggen.")
        if any(item not in BOUNDARY_NAMES for item in self.verified_boundaries):
            raise ValueError("Atlasvlak bevat een onbekende bevestigde veldgrens.")
        if any(item not in BOUNDARY_NAMES for item in self.inferred_boundaries):
            raise ValueError("Atlasvlak bevat een onbekende berekende veldgrens.")
        if set(self.verified_boundaries) & set(self.inferred_boundaries):
            raise ValueError("Een veldgrens kan niet tegelijk bevestigd en berekend zijn.")
        object.__setattr__(self, "ground_to_anchor", matrix / matrix[2, 2])

    def contains(self, ground_point: tuple[float, float]) -> bool:
        polygon = np.asarray(self.support_polygon, dtype=np.float32)
        return cv2.pointPolygonTest(polygon, ground_point, False) >= 0.0

    def project(
        self,
        ground_point: tuple[float, float],
        anchor_to_frame: np.ndarray | None = None,
    ) -> tuple[float, float]:
        matrix = self.ground_to_anchor
        if anchor_to_frame is not None:
            transform = np.asarray(anchor_to_frame, dtype=np.float64)
            if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
                raise ValueError("Anker-naar-frame-transformatie moet eindig en 3x3 zijn.")
            matrix = transform @ matrix
        point = matrix @ np.asarray((*ground_point, 1.0), dtype=np.float64)
        if abs(float(point[2])) < 1e-12:
            raise ValueError("Atlaspunt projecteert naar oneindig.")
        return float(point[0] / point[2]), float(point[1] / point[2])

    def to_dict(self) -> dict:
        return {
            "patch_id": self.patch_id,
            "anchor_frame": self.anchor_frame,
            "ground_to_anchor": self.ground_to_anchor.tolist(),
            "support_polygon": [list(item) for item in self.support_polygon],
            "confidence": self.confidence,
            "source": self.source,
            "verified_boundaries": list(self.verified_boundaries),
            "inferred_boundaries": list(self.inferred_boundaries),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LocalFieldPatch":
        return cls(
            str(data["patch_id"]),
            int(data["anchor_frame"]),
            np.asarray(data["ground_to_anchor"], dtype=np.float64),
            tuple(tuple(map(float, item)) for item in data["support_polygon"]),
            float(data["confidence"]),
            str(data["source"]),
            tuple(map(str, data.get("verified_boundaries", ()))),
            tuple(map(str, data.get("inferred_boundaries", ()))),
        )


@dataclass(frozen=True, slots=True)
class LocalFieldAtlas:
    video_name: str
    match_format: str
    pitch_length_m: float
    pitch_width_m: float
    patches: tuple[LocalFieldPatch, ...]
    manual_midfield_line: ManualMidfieldLine | None = None
    manual_parallel_lines: ManualParallelLineReference | None = None

    def __post_init__(self) -> None:
        if self.pitch_length_m <= 0.0 or self.pitch_width_m <= 0.0:
            raise ValueError("Atlasafmetingen moeten positief zijn.")
        if not self.patches:
            raise ValueError("Een veldatlas vereist minimaal een lokaal vlak.")
        if len({item.patch_id for item in self.patches}) != len(self.patches):
            raise ValueError("Lokale atlasvlakken moeten unieke namen hebben.")

    def covering_patches(self, ground_point: tuple[float, float]) -> tuple[LocalFieldPatch, ...]:
        return tuple(item for item in self.patches if item.contains(ground_point))

    @property
    def complete_field_coverage(self) -> bool:
        tolerance = 1e-6
        intervals = sorted(
            (
                min(point[0] for point in patch.support_polygon),
                max(point[0] for point in patch.support_polygon),
            )
            for patch in self.patches
            if min(point[1] for point in patch.support_polygon) <= tolerance
            and max(point[1] for point in patch.support_polygon)
            >= self.pitch_width_m - tolerance
        )
        if not intervals or intervals[0][0] > tolerance:
            return False
        covered_until = intervals[0][1]
        for start, end in intervals[1:]:
            if start > covered_until + tolerance:
                return False
            covered_until = max(covered_until, end)
        return covered_until >= self.pitch_length_m - tolerance

    def blended_projection(
        self,
        ground_point: tuple[float, float],
        anchor_to_frame: dict[str, np.ndarray],
        runtime_quality: dict[str, float] | None = None,
    ) -> tuple[tuple[float, float], float, tuple[str, ...]]:
        projected = []
        weights = []
        identifiers = []
        for patch in self.covering_patches(ground_point):
            transform = anchor_to_frame.get(patch.patch_id)
            if transform is None:
                continue
            quality = 1.0 if runtime_quality is None else runtime_quality.get(patch.patch_id, 0.0)
            weight = patch.confidence * max(0.0, float(quality))
            if weight <= 0.0:
                continue
            projected.append(patch.project(ground_point, transform))
            weights.append(weight)
            identifiers.append(patch.patch_id)
        if not projected:
            raise ValueError("Geen actief lokaal vlak dekt dit veldpunt.")
        points = np.asarray(projected, dtype=np.float64)
        weight_values = np.asarray(weights, dtype=np.float64)
        blended = np.average(points, axis=0, weights=weight_values)
        disagreement = float(
            np.sqrt(np.average(np.sum(np.square(points - blended), axis=1), weights=weight_values))
        )
        return (float(blended[0]), float(blended[1])), disagreement, tuple(identifiers)

    def visible_evidence(
        self,
        patch_id: str,
        frame_size: tuple[int, int],
        anchor_to_frame: np.ndarray | None = None,
    ) -> VisibleFieldEvidence:
        patch = next((item for item in self.patches if item.patch_id == patch_id), None)
        if patch is None:
            raise KeyError(f"Onbekend lokaal atlasvlak: {patch_id}")
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("Frame-afmetingen moeten positief zijn.")
        projected_polygon = np.asarray(
            [patch.project(point, anchor_to_frame) for point in patch.support_polygon],
            dtype=np.float32,
        )
        frame_polygon = np.asarray(
            ((0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)),
            dtype=np.float32,
        )
        visible_polygon: tuple[tuple[float, float], ...] = ()
        if cv2.isContourConvex(projected_polygon.reshape(-1, 1, 2)):
            area, intersection = cv2.intersectConvexConvex(projected_polygon, frame_polygon)
            if area > 0.0 and intersection is not None:
                visible_polygon = tuple(tuple(map(float, item)) for item in intersection.reshape(-1, 2))
        minimum_x = min(item[0] for item in patch.support_polygon)
        maximum_x = max(item[0] for item in patch.support_polygon)
        corners = ground_corners(self.pitch_length_m, self.pitch_width_m)
        boundaries = {
            name: (corners[first], corners[second])
            for name, (first, second) in BOUNDARY_CORNERS.items()
            if name.startswith("sideline_")
        }
        tolerance = 1e-6
        if abs(minimum_x) <= tolerance:
            first, second = BOUNDARY_CORNERS["end_line_a"]
            boundaries["end_line_a"] = (corners[first], corners[second])
        if abs(maximum_x - self.pitch_length_m) <= tolerance:
            first, second = BOUNDARY_CORNERS["end_line_b"]
            boundaries["end_line_b"] = (corners[first], corners[second])
        segments = []
        status = {name: "UNKNOWN" for name in BOUNDARY_NAMES}
        rectangle = (0, 0, width, height)
        for name, (ground_start, ground_end) in boundaries.items():
            if name in patch.verified_boundaries:
                boundary_status = "VISIBLE"
            elif name in patch.inferred_boundaries:
                boundary_status = "INFERRED"
            else:
                continue
            image_start = patch.project(ground_start, anchor_to_frame)
            image_end = patch.project(ground_end, anchor_to_frame)
            clipped, first, second = cv2.clipLine(
                rectangle,
                tuple(np.rint(image_start).astype(int)),
                tuple(np.rint(image_end).astype(int)),
            )
            if not clipped or np.linalg.norm(np.asarray(second) - np.asarray(first)) < 8.0:
                continue
            status[name] = boundary_status
            segments.append(
                VisibleBoundarySegment(
                    name, tuple(map(float, first)), tuple(map(float, second)),
                    patch.patch_id, patch.confidence, boundary_status,
                )
            )
        coverage = (
            abs(float(cv2.contourArea(np.asarray(visible_polygon, dtype=np.float32))))
            / float(width * height)
            if len(visible_polygon) >= 3 else 0.0
        )
        return VisibleFieldEvidence(
            patch.patch_id, visible_polygon, tuple(segments), status, coverage
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "video_name": self.video_name,
            "match_format": self.match_format,
            "pitch_length_m": self.pitch_length_m,
            "pitch_width_m": self.pitch_width_m,
            "patches": [item.to_dict() for item in self.patches],
            "complete_field_coverage": self.complete_field_coverage,
            "manual_midfield_line": (
                None if self.manual_midfield_line is None else self.manual_midfield_line.to_dict()
            ),
            "manual_parallel_lines": (
                None if self.manual_parallel_lines is None else self.manual_parallel_lines.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LocalFieldAtlas":
        return cls(
            str(data["video_name"]), str(data["match_format"]),
            float(data["pitch_length_m"]), float(data["pitch_width_m"]),
            tuple(LocalFieldPatch.from_dict(item) for item in data["patches"]),
            (
                None
                if data.get("manual_midfield_line") is None
                else ManualMidfieldLine.from_dict(data["manual_midfield_line"])
            ),
            (
                None
                if data.get("manual_parallel_lines") is None
                else ManualParallelLineReference.from_dict(data["manual_parallel_lines"])
            ),
        )


def save_local_field_atlas(atlas: LocalFieldAtlas, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(atlas.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_local_field_atlas(path: Path) -> LocalFieldAtlas:
    return LocalFieldAtlas.from_dict(json.loads(path.read_text(encoding="utf-8")))


def align_patch_to_front_sideline(
    ground_to_image: np.ndarray,
    goal_id: str,
    pitch_length_m: float,
    pitch_width_m: float,
    observed_points: np.ndarray,
    direction_vanishing_point: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Keep the end line fixed while aligning the camera-side boundary evidence."""
    matrix = np.asarray(ground_to_image, dtype=np.float64)
    points = np.asarray(observed_points, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Zijlijnverfijning vereist een geldige grondhomography.")
    if goal_id not in ("A", "B"):
        raise ValueError("Zijlijnverfijning vereist Doel A of Doel B.")
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
        raise ValueError("Zijlijnverfijning vereist minimaal twee waargenomen lijnpunten.")
    center = np.mean(points, axis=0)
    if direction_vanishing_point is None:
        _u, _s, vh = np.linalg.svd(points - center)
        direction = vh[0]
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        line = np.asarray((normal[0], normal[1], -float(normal @ center)))
    else:
        vanishing = np.asarray(direction_vanishing_point, dtype=np.float64)
        if vanishing.shape != (2,) or not np.all(np.isfinite(vanishing)):
            raise ValueError("Het verdwijnpunt voor de zijlijnen moet eindig zijn.")
        end_x = 0.0 if goal_id == "A" else pitch_length_m
        fixed_corner = matrix @ np.asarray((end_x, pitch_width_m, 1.0))
        fixed_corner /= fixed_corner[2]
        line = np.cross(fixed_corner, (*vanishing, 1.0)).astype(np.float64)
    line /= np.linalg.norm(line[:2])
    if direction_vanishing_point is not None:
        vanishing_h = np.asarray((*direction_vanishing_point, 1.0), dtype=np.float64)
        old_direction = matrix[:, 0]
        initial_scale = float(vanishing_h @ old_direction / (vanishing_h @ vanishing_h))
        samples_inward = np.linspace(0.0, pitch_length_m * 0.62, 7)
        sample_x = samples_inward if goal_id == "A" else pitch_length_m - samples_inward
        scale_ground = np.vstack(
            (
                np.column_stack((sample_x, np.zeros_like(sample_x), np.ones_like(sample_x))),
                np.column_stack(
                    (sample_x, np.full_like(sample_x, pitch_width_m), np.ones_like(sample_x))
                ),
            )
        )
        old_projected = (matrix @ scale_ground.T).T
        old_projected = old_projected[:, :2] / old_projected[:, 2:3]

        def with_scale(value: float) -> np.ndarray:
            candidate = matrix.copy()
            candidate[:, 0] = value * vanishing_h
            if goal_id == "B":
                candidate[:, 2] += pitch_length_m * (old_direction - candidate[:, 0])
            return candidate

        def scale_residual(values: np.ndarray) -> np.ndarray:
            candidate = with_scale(float(values[0]))
            projected = (candidate @ scale_ground.T).T
            valid = np.abs(projected[:, 2]) > 1e-8
            if not np.all(valid):
                return np.full(old_projected.size, 10000.0)
            projected = projected[:, :2] / projected[:, 2:3]
            return (projected - old_projected).reshape(-1)

        scale_result = least_squares(
            scale_residual, np.asarray((initial_scale,)), loss="soft_l1", f_scale=20.0
        )
        refined = with_scale(float(scale_result.x[0]))
        refined /= refined[2, 2]
        distances = points @ line[:2] + line[2]
        return refined, float(np.sqrt(np.mean(np.square(distances))))
    inward = np.linspace(0.0, pitch_length_m * 0.62, 7)
    ground_x = inward if goal_id == "A" else pitch_length_m - inward
    ground = np.column_stack((ground_x, np.full_like(ground_x, pitch_width_m), np.ones_like(ground_x)))

    def world_adjustment(scale: float, shear: float) -> np.ndarray:
        if goal_id == "A":
            return np.asarray(((scale, 0.0, 0.0), (shear, 1.0, 0.0), (0.0, 0.0, 1.0)))
        return np.asarray(
            (
                (scale, 0.0, pitch_length_m * (1.0 - scale)),
                (-shear, 1.0, shear * pitch_length_m),
                (0.0, 0.0, 1.0),
            )
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        scale, shear = parameters
        candidate = matrix @ world_adjustment(scale, shear)
        projected = (candidate @ ground.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        distances = projected @ line[:2] + line[2]
        vanishing = candidate[:, 0]
        vanishing_distance = (
            float(line @ (vanishing / vanishing[2]))
            if abs(float(vanishing[2])) > 1e-9 else 1000.0
        )
        regularisation = np.asarray(((scale - 1.0) * 2.0, shear * 2.0))
        return np.concatenate((distances, (vanishing_distance * 8.0,), regularisation))

    result = least_squares(
        residual, np.asarray((1.0, 0.0)), bounds=((0.35, -1.5), (2.5, 1.5)),
        loss="soft_l1", f_scale=2.0,
    )
    refined = matrix @ world_adjustment(float(result.x[0]), float(result.x[1]))
    refined /= refined[2, 2]
    distances = residual(result.x)[:len(ground)]
    return refined, float(np.sqrt(np.mean(np.square(distances))))


def anchor_patch_to_measured_endline(
    ground_to_image: np.ndarray,
    goal_id: str,
    pitch_length_m: float,
    pitch_width_m: float,
    rear_corner: tuple[float, float],
    front_corner: tuple[float, float],
    direction_vanishing_point: tuple[float, float],
    front_sideline_points: np.ndarray | None = None,
    inward_distance_m: float = 12.0,
) -> np.ndarray:
    """Bind a local ground plane to the measured end-line corners and its VP.

    The camera solve remains useful for metric depth, but it may slightly move the
    two physical corners because most observations lie on the vertical goal
    plane.  A field boundary must never inherit that error.  This reconstruction
    pins the end line to the measured corners and carries the old metric depth
    along two rays that meet at the independently measured longitudinal
    vanishing point.
    """
    matrix = np.asarray(ground_to_image, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Hoekverankering vereist een geldige grondhomography.")
    if goal_id not in ("A", "B"):
        raise ValueError("Hoekverankering vereist Doel A of Doel B.")
    if not 0.0 < inward_distance_m < pitch_length_m:
        raise ValueError("De lokale diepte moet binnen het speelveld liggen.")
    vanishing = np.asarray(direction_vanishing_point, dtype=np.float64)
    if vanishing.shape != (2,) or not np.all(np.isfinite(vanishing)):
        raise ValueError("Hoekverankering vereist een eindig verdwijnpunt.")

    end_x = 0.0 if goal_id == "A" else pitch_length_m
    inner_x = inward_distance_m if goal_id == "A" else pitch_length_m - inward_distance_m
    world = np.asarray(
        ((end_x, 0.0), (end_x, pitch_width_m), (inner_x, 0.0), (inner_x, pitch_width_m)),
        dtype=np.float64,
    )
    old = _project_ground_points(matrix, world)
    measured = (np.asarray(rear_corner, dtype=np.float64), np.asarray(front_corner, dtype=np.float64))
    destinations = [measured[0], measured[1]]
    for index, endpoint in enumerate(measured):
        old_end = old[index]
        old_inner = old[index + 2]
        old_axis = old_end - vanishing
        denominator = float(old_axis @ old_axis)
        if denominator < 1e-8:
            raise ValueError("Verdwijnpunt valt samen met een gemeten veldhoek.")
        ratio = float((old_inner - vanishing) @ old_axis / denominator)
        # The inner sample must remain on the finite field side of both the
        # end line and the longitudinal horizon.  A goal-plane-only pose can
        # otherwise suggest a negative/very small ratio and make the ground
        # homography cross infinity before midfield (the visible V-shape bug).
        minimum_ratio = 1.0 - 0.85 * inward_distance_m / pitch_length_m
        toward_ratio = float(np.clip(ratio, minimum_ratio, 0.985))
        if front_sideline_points is not None:
            sideline = np.asarray(front_sideline_points, dtype=np.float64)
            if sideline.ndim != 2 or sideline.shape[1:] != (2,) or len(sideline) < 2:
                raise ValueError("Richtingscontrole vereist minimaal twee zijlijnpunten.")
            observed_direction = np.mean(sideline, axis=0) - measured[1]
            away_direction = measured[1] - vanishing
            goes_away = float(observed_direction @ away_direction) > 0.0
        else:
            goes_away = ratio > 1.0
        if goes_away:
            # Retain the old perspective magnitude but reverse it onto the
            # observed field side. This is what connects the corner toward the
            # other goal instead of folding back toward the horizon.
            delta = float(np.clip(abs(1.0 - ratio), 0.015, 0.22))
            ratio = 1.0 + delta
        else:
            ratio = toward_ratio
        destinations.append(vanishing + ratio * (endpoint - vanishing))

    anchored = cv2.getPerspectiveTransform(
        world.astype(np.float32), np.asarray(destinations, dtype=np.float32)
    ).astype(np.float64)
    if abs(float(anchored[2, 2])) < 1e-10:
        raise ValueError("De verankerde veldprojectie is numeriek instabiel.")
    return anchored / anchored[2, 2]


def _project_ground_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (matrix @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) < 1e-9):
        raise ValueError("Grondpunt projecteert naar oneindig.")
    return projected[:, :2] / projected[:, 2:3]
