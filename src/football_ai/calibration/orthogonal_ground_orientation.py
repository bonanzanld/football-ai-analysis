from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

import numpy as np

from football_ai.calibration.ground_line_evidence import GroundLineFamily


@dataclass(frozen=True, slots=True)
class ImageLineObservation:
    family: GroundLineFamily
    start: tuple[float, float]
    end: tuple[float, float]
    weight: float = 1.0
    ground_offset_m: float | None = None
    source_id: str | None = None

    def equation(self) -> np.ndarray:
        line = np.cross((*self.start, 1.0), (*self.end, 1.0)).astype(np.float64)
        normal = float(np.linalg.norm(line[:2]))
        if normal < 1e-9:
            raise ValueError("Beeldlijn moet twee verschillende punten hebben.")
        return line / normal


@dataclass(frozen=True, slots=True)
class OrthogonalGroundOrientation:
    longitudinal_vanishing_point: tuple[float, float]
    transverse_vanishing_point: tuple[float, float]
    horizon_line: tuple[float, float, float]
    focal_length_pixels: float
    longitudinal_inliers: int
    transverse_inliers: int
    longitudinal_rms_degrees: float
    transverse_rms_degrees: float
    line_diversity: dict

    def to_dict(self) -> dict:
        return {
            "longitudinal_vanishing_point": list(self.longitudinal_vanishing_point),
            "transverse_vanishing_point": list(self.transverse_vanishing_point),
            "horizon_line": list(self.horizon_line),
            "focal_length_pixels": self.focal_length_pixels,
            "longitudinal_inliers": self.longitudinal_inliers,
            "transverse_inliers": self.transverse_inliers,
            "longitudinal_rms_degrees": self.longitudinal_rms_degrees,
            "transverse_rms_degrees": self.transverse_rms_degrees,
            "line_diversity": self.line_diversity,
        }


@dataclass(frozen=True, slots=True)
class LineCluster:
    family: GroundLineFamily
    representative: ImageLineObservation
    observation_count: int
    source_count: int
    mean_ground_offset_m: float | None

    def to_dict(self) -> dict:
        return {
            "family": self.family.value,
            "observation_count": self.observation_count,
            "source_count": self.source_count,
            "mean_ground_offset_m": self.mean_ground_offset_m,
            "representative": {
                "start": list(self.representative.start),
                "end": list(self.representative.end),
                "weight": self.representative.weight,
                "source_id": self.representative.source_id,
            },
        }


@dataclass(frozen=True, slots=True)
class FarGoalObservation2D:
    """Optional metric/vertical QA for a visible full-size 11v11 goal."""

    left_bottom: tuple[float, float]
    right_bottom: tuple[float, float]
    left_top: tuple[float, float]
    right_top: tuple[float, float]
    goal_width_m: float = 7.32
    goal_height_m: float = 2.44

    def __post_init__(self) -> None:
        points = np.asarray(
            (self.left_bottom, self.right_bottom, self.left_top, self.right_top),
            dtype=np.float64,
        )
        if points.shape != (4, 2) or not np.all(np.isfinite(points)):
            raise ValueError("Verre-doelobservatie vereist vier eindige beeldpunten.")
        if self.goal_width_m <= 0.0 or self.goal_height_m <= 0.0:
            raise ValueError("Doelafmetingen moeten positief zijn.")

    def to_dict(self) -> dict:
        return {
            "left_bottom": list(self.left_bottom),
            "right_bottom": list(self.right_bottom),
            "left_top": list(self.left_top),
            "right_top": list(self.right_top),
            "goal_width_m": self.goal_width_m,
            "goal_height_m": self.goal_height_m,
        }


def estimate_orthogonal_ground_orientation(
    observations: tuple[ImageLineObservation, ...],
    frame_size: tuple[int, int],
    angular_threshold_degrees: float = 2.5,
    minimum_lines_per_family: int = 3,
) -> OrthogonalGroundOrientation:
    if minimum_lines_per_family < 2:
        raise ValueError("Minimaal twee lijnen per richtingsfamilie vereist.")
    width, height = frame_size
    clusters = cluster_physical_lines(observations)
    families = {
        family: tuple(cluster.representative for cluster in clusters[family])
        for family in GroundLineFamily
    }
    diversity = summarize_line_diversity(clusters)
    for family, items in families.items():
        family_diversity = diversity[family.value]
        if len(items) < minimum_lines_per_family:
            raise ValueError(
                f"Minimaal {minimum_lines_per_family} verschillende lijnen vereist voor {family.value}."
            )
        if not family_diversity["sufficient_spread"]:
            raise ValueError(
                f"De lijnen voor {family.value} liggen te dicht bij elkaar; "
                "minimaal 5 meter of 100 pixels spreiding vereist."
            )
    longitudinal = _robust_vanishing_point(families[GroundLineFamily.LONGITUDINAL], angular_threshold_degrees)
    transverse = _robust_vanishing_point(families[GroundLineFamily.TRANSVERSE], angular_threshold_degrees)
    first = np.asarray((*longitudinal[0], 1.0), dtype=np.float64)
    second = np.asarray((*transverse[0], 1.0), dtype=np.float64)
    horizon = np.cross(first, second)
    horizon /= max(float(np.linalg.norm(horizon[:2])), 1e-12)
    center = np.asarray((width / 2.0, height / 2.0), dtype=np.float64)
    focal_squared = -float((np.asarray(longitudinal[0]) - center) @ (np.asarray(transverse[0]) - center))
    if focal_squared <= (0.15 * width) ** 2:
        raise ValueError("De twee lijnfamilies leveren geen fysiek geldige 90-graden-cameraoriëntatie.")
    focal = float(np.sqrt(focal_squared))
    if not 0.2 * width <= focal <= 5.0 * width:
        raise ValueError(f"Geschatte brandpuntsafstand is onrealistisch ({focal:.0f} px).")
    return OrthogonalGroundOrientation(
        longitudinal_vanishing_point=longitudinal[0],
        transverse_vanishing_point=transverse[0],
        horizon_line=tuple(map(float, horizon)),
        focal_length_pixels=focal,
        longitudinal_inliers=longitudinal[1],
        transverse_inliers=transverse[1],
        longitudinal_rms_degrees=longitudinal[2],
        transverse_rms_degrees=transverse[2],
        line_diversity=diversity,
    )


def transform_line_observation(
    observation: ImageLineObservation,
    source_to_target: np.ndarray,
) -> ImageLineObservation:
    transform = np.asarray(source_to_target, dtype=np.float64)
    points = np.asarray((observation.start, observation.end), dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(2)))
    projected = (transform @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) < 1e-12):
        raise ValueError("Lijn projecteert naar oneindig.")
    projected = projected[:, :2] / projected[:, 2:3]
    return ImageLineObservation(
        observation.family,
        tuple(map(float, projected[0])),
        tuple(map(float, projected[1])),
        observation.weight,
        observation.ground_offset_m,
        observation.source_id,
    )


def _robust_vanishing_point(
    observations: tuple[ImageLineObservation, ...],
    threshold_degrees: float,
) -> tuple[tuple[float, float], int, float]:
    lines = [item.equation() for item in observations]
    best: tuple[int, float, np.ndarray, np.ndarray] | None = None
    for first, second in combinations(lines, 2):
        point = np.cross(first, second)
        if abs(float(point[2])) < 1e-8:
            continue
        point /= point[2]
        residuals = np.asarray([_angular_residual(item, point[:2]) for item in observations])
        mask = residuals <= threshold_degrees
        count = int(np.count_nonzero(mask))
        rms = float(np.sqrt(np.mean(np.square(residuals[mask])))) if count else float("inf")
        if best is None or count > best[0] or (count == best[0] and rms < best[1]):
            best = count, rms, point, mask
    if best is None or best[0] < 3:
        raise ValueError("Lijnfamilie bepaalt geen eindig verdwijnpunt.")
    selected = [line for line, keep in zip(lines, best[3]) if keep]
    weights = [item.weight for item, keep in zip(observations, best[3]) if keep]
    system = np.asarray([line * np.sqrt(weight) for line, weight in zip(selected, weights)])
    _u, _s, vh = np.linalg.svd(system)
    point = vh[-1]
    if abs(float(point[2])) < 1e-9:
        raise ValueError("Verdwijnpunt ligt numeriek op oneindig.")
    point /= point[2]
    residuals = np.asarray([_angular_residual(item, point[:2]) for item, keep in zip(observations, best[3]) if keep])
    return (float(point[0]), float(point[1])), best[0], float(np.sqrt(np.mean(np.square(residuals))))


def _angular_residual(observation: ImageLineObservation, point: np.ndarray) -> float:
    start, end = np.asarray(observation.start), np.asarray(observation.end)
    midpoint = (start + end) / 2.0
    direction = end - start
    toward_vanishing = point - midpoint
    denominator = float(np.linalg.norm(direction) * np.linalg.norm(toward_vanishing))
    if denominator < 1e-9:
        return 90.0
    cosine = np.clip(abs(float(direction @ toward_vanishing)) / denominator, 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def cluster_physical_lines(
    observations: tuple[ImageLineObservation, ...],
) -> dict[GroundLineFamily, tuple[LineCluster, ...]]:
    result: dict[GroundLineFamily, tuple[LineCluster, ...]] = {}
    for family in GroundLineFamily:
        items = tuple(item for item in observations if item.family == family)
        parents = list(range(len(items)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parents[second_root] = first_root

        for first_index, second_index in combinations(range(len(items)), 2):
            if _same_physical_line(items[first_index], items[second_index]):
                union(first_index, second_index)
        groups: dict[int, list[ImageLineObservation]] = {}
        for index, item in enumerate(items):
            groups.setdefault(find(index), []).append(item)
        family_clusters = []
        for members in groups.values():
            representative = max(members, key=lambda item: item.weight)
            offsets = [item.ground_offset_m for item in members if item.ground_offset_m is not None]
            sources = {item.source_id for item in members if item.source_id is not None}
            family_clusters.append(
                LineCluster(
                    family,
                    representative,
                    len(members),
                    len(sources),
                    float(np.mean(offsets)) if offsets else None,
                )
            )
        result[family] = tuple(sorted(family_clusters, key=lambda item: item.representative.weight, reverse=True))
    return result


def summarize_line_diversity(
    clusters: dict[GroundLineFamily, tuple[LineCluster, ...]],
) -> dict[str, dict]:
    result = {}
    for family, family_clusters in clusters.items():
        offsets = [item.mean_ground_offset_m for item in family_clusters if item.mean_ground_offset_m is not None]
        metric_span = float(max(offsets) - min(offsets)) if len(offsets) >= 2 else 0.0
        image_separation = 0.0
        for first, second in combinations(family_clusters, 2):
            image_separation = max(
                image_separation,
                _symmetric_line_separation(first.representative, second.representative),
            )
        result[family.value] = {
            "cluster_count": len(family_clusters),
            "metric_offset_span_m": metric_span,
            "image_separation_px": image_separation,
            "sufficient_spread": metric_span >= 5.0 or image_separation >= 100.0,
            "clusters": [item.to_dict() for item in family_clusters],
        }
    return result


def _same_physical_line(first: ImageLineObservation, second: ImageLineObservation) -> bool:
    if (
        first.ground_offset_m is not None
        and second.ground_offset_m is not None
        and abs(first.ground_offset_m - second.ground_offset_m) < 1.5
    ):
        return True
    first_equation, second_equation = first.equation(), second.equation()
    angle = np.degrees(
        np.arccos(np.clip(abs(float(first_equation[:2] @ second_equation[:2])), 0.0, 1.0))
    )
    return angle < 3.0 and _symmetric_line_separation(first, second) < 35.0


def _symmetric_line_separation(first: ImageLineObservation, second: ImageLineObservation) -> float:
    first_midpoint = (np.asarray(first.start) + np.asarray(first.end)) / 2.0
    second_midpoint = (np.asarray(second.start) + np.asarray(second.end)) / 2.0
    first_equation, second_equation = first.equation(), second.equation()
    return max(
        abs(float(first_equation @ np.asarray((*second_midpoint, 1.0)))),
        abs(float(second_equation @ np.asarray((*first_midpoint, 1.0)))),
    )
