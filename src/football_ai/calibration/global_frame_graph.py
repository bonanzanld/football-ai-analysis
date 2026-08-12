from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from football_ai.calibration.bootstrap.white_line_detection import extract_white_pitch_mask
from football_ai.calibration.ground_line_evidence import GroundLineFamily


@dataclass(frozen=True, slots=True)
class FrameGraphNode:
    node_id: str
    frame_number: int
    time_seconds: float


@dataclass(frozen=True, slots=True)
class FrameGraphEdge:
    source_id: str
    target_id: str
    source_to_target: np.ndarray
    matches: int
    inliers: int
    inlier_ratio: float
    source_coverage: float
    target_coverage: float
    median_error_px: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.source_to_target, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("Framegraph-edge vereist een eindige 3x3-homography.")
        object.__setattr__(self, "source_to_target", matrix / matrix[2, 2])


@dataclass(frozen=True, slots=True)
class GlobalFrameGraphSolution:
    reference_id: str
    node_to_reference: dict[str, np.ndarray]
    edge_rms_px: float
    maximum_edge_error_px: float
    connected_nodes: tuple[str, ...]
    rejected_nodes: tuple[str, ...]
    used_edges: int


@dataclass(frozen=True, slots=True)
class GroundDirectionConstraint:
    node_id: str
    family: GroundLineFamily
    image_start: tuple[float, float]
    image_end: tuple[float, float]
    weight: float = 1.0

    def equation(self) -> np.ndarray:
        line = np.cross((*self.image_start, 1.0), (*self.image_end, 1.0)).astype(np.float64)
        normal = float(np.linalg.norm(line[:2]))
        if normal < 1e-9:
            raise ValueError("Richtingsvoorwaarde vereist twee verschillende beeldpunten.")
        return line / normal


@dataclass(frozen=True, slots=True)
class AbsoluteGroundConstraint:
    """Pins a graph node to an independently calibrated ground projection."""

    node_id: str
    ground_to_image: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        matrix = np.asarray(self.ground_to_image, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("Absoluut grondanker vereist een eindige 3x3-homography.")
        object.__setattr__(self, "ground_to_image", matrix / matrix[2, 2])


@dataclass(frozen=True, slots=True)
class AbsoluteGroundPointConstraint:
    node_id: str
    ground_points: np.ndarray
    image_points: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        ground = np.asarray(self.ground_points, dtype=np.float64)
        image = np.asarray(self.image_points, dtype=np.float64)
        if ground.shape != image.shape or ground.ndim != 2 or ground.shape[1] != 2:
            raise ValueError("Absoluut puntanker vereist evenveel 2D-grond- als beeldpunten.")
        if len(ground) < 1 or not np.all(np.isfinite(ground)) or not np.all(np.isfinite(image)):
            raise ValueError("Absoluut puntanker bevat ongeldige punten.")
        object.__setattr__(self, "ground_points", ground)
        object.__setattr__(self, "image_points", image)


@dataclass(frozen=True, slots=True)
class AbsoluteGroundLineConstraint:
    """Pins known metric ground points to one observed image line."""

    node_id: str
    ground_points: np.ndarray
    image_line: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        ground = np.asarray(self.ground_points, dtype=np.float64)
        line = np.asarray(self.image_line, dtype=np.float64)
        if ground.ndim != 2 or ground.shape[1] != 2 or len(ground) < 2:
            raise ValueError("Absolute grondlijn vereist minstens twee metrische punten.")
        if line.shape != (3,) or not np.all(np.isfinite(ground)) or not np.all(np.isfinite(line)):
            raise ValueError("Absolute grondlijn bevat ongeldige waarden.")
        normal = float(np.linalg.norm(line[:2]))
        if normal < 1e-9:
            raise ValueError("Absolute beeldlijn is degeneraat.")
        object.__setattr__(self, "ground_points", ground)
        object.__setattr__(self, "image_line", line / normal)


def homography_local_scale_ratio(
    matrix: np.ndarray,
    point: tuple[float, float] = (640.0, 360.0),
) -> float:
    """Return isotropic area-scale proxy of a homography near one image point."""

    homography = np.asarray(matrix, dtype=np.float64)
    if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        raise ValueError("Scale ratio requires a finite 3x3 homography")
    x, y = (float(value) for value in point)
    a, b, c = homography[0]
    d, e, f = homography[1]
    g, h, i = homography[2]
    denominator = g * x + h * y + i
    if abs(float(denominator)) < 1e-12:
        raise ValueError("Homography maps scale sample to infinity")
    numerator_x = a * x + b * y + c
    numerator_y = d * x + e * y + f
    jacobian = np.asarray(
        (
            (
                (a * denominator - g * numerator_x) / denominator**2,
                (b * denominator - h * numerator_x) / denominator**2,
            ),
            (
                (d * denominator - g * numerator_y) / denominator**2,
                (e * denominator - h * numerator_y) / denominator**2,
            ),
        ),
        dtype=np.float64,
    )
    determinant = abs(float(np.linalg.det(jacobian)))
    if not np.isfinite(determinant) or determinant < 1e-12:
        raise ValueError("Homography has degenerate local scale")
    return float(np.sqrt(determinant))


def connected_frame_graph_components(
    nodes: tuple[FrameGraphNode, ...],
    edges: tuple[FrameGraphEdge, ...],
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic connected node groups after edge filtering."""

    adjacency = {node.node_id: set() for node in nodes}
    for edge in edges:
        if edge.source_id not in adjacency or edge.target_id not in adjacency:
            raise ValueError("Framegraph edge references an unknown node")
        adjacency[edge.source_id].add(edge.target_id)
        adjacency[edge.target_id].add(edge.source_id)
    remaining = set(adjacency)
    components = []
    while remaining:
        start = min(remaining)
        component = {start}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbour in sorted(adjacency[current]):
                if neighbour in component:
                    continue
                component.add(neighbour)
                queue.append(neighbour)
        remaining -= component
        components.append(tuple(sorted(component)))
    components.sort(key=lambda item: (-len(item), item))
    return tuple(components)


def select_maximum_quality_tree(
    nodes: tuple[FrameGraphNode, ...],
    edges: tuple[FrameGraphEdge, ...],
) -> tuple[FrameGraphEdge, ...]:
    """Select the strongest cycle-free ground connections across all nodes."""
    parents = {item.node_id: item.node_id for item in nodes}

    def find(node_id: str) -> str:
        while parents[node_id] != node_id:
            parents[node_id] = parents[parents[node_id]]
            node_id = parents[node_id]
        return node_id

    def union(first: str, second: str) -> bool:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return False
        parents[second_root] = first_root
        return True

    def quality(edge: FrameGraphEdge) -> float:
        coverage = np.sqrt(max(edge.source_coverage * edge.target_coverage, 0.0))
        return (
            edge.inliers
            * edge.inlier_ratio
            * coverage
            / max(1.0 + edge.median_error_px, 1e-6)
        )

    selected = []
    for edge in sorted(edges, key=quality, reverse=True):
        if union(edge.source_id, edge.target_id):
            selected.append(edge)
    return tuple(selected)


def select_cycle_consistent_edges(
    tree_solution: GlobalFrameGraphSolution,
    tree_edges: tuple[FrameGraphEdge, ...],
    candidate_edges: tuple[FrameGraphEdge, ...],
    maximum_error_px: float = 8.0,
) -> tuple[FrameGraphEdge, ...]:
    """Keep the connected backbone and only loop edges that agree with it."""
    tree_keys = {
        (edge.source_id, edge.target_id)
        for edge in tree_edges
    }
    samples = np.asarray(
        (
            (0.0, 0.0), (640.0, 0.0), (1280.0, 0.0),
            (0.0, 360.0), (640.0, 360.0), (1280.0, 360.0),
            (0.0, 720.0), (640.0, 720.0), (1280.0, 720.0),
        ),
        dtype=np.float64,
    )
    selected = []
    for edge in candidate_edges:
        key = (edge.source_id, edge.target_id)
        if key in tree_keys:
            selected.append(edge)
            continue
        if (
            edge.source_id not in tree_solution.node_to_reference
            or edge.target_id not in tree_solution.node_to_reference
        ):
            continue
        direct = _project(samples, tree_solution.node_to_reference[edge.source_id])
        via_edge = _project(
            _project(samples, edge.source_to_target),
            tree_solution.node_to_reference[edge.target_id],
        )
        error = float(np.sqrt(np.mean(np.sum(np.square(direct - via_edge), axis=1))))
        if np.isfinite(error) and error <= maximum_error_px:
            selected.append(edge)
    return tuple(selected)


def estimate_frame_edge(
    source_id: str,
    target_id: str,
    source: np.ndarray,
    target: np.ndarray,
) -> FrameGraphEdge:
    return _estimate_frame_edge(source_id, target_id, source, target, None, None)


def estimate_ground_frame_edge(
    source_id: str,
    target_id: str,
    source: np.ndarray,
    target: np.ndarray,
) -> FrameGraphEdge:
    """Estimate camera motion from features on the shared grass plane only."""
    source_grass, _source_white = extract_white_pitch_mask(source)
    target_grass, _target_white = extract_white_pitch_mask(target)
    kernel = np.ones((11, 11), dtype=np.uint8)
    source_mask = cv2.erode(source_grass, kernel)
    target_mask = cv2.erode(target_grass, kernel)
    if (
        np.count_nonzero(source_mask) < 0.12 * source_mask.size
        or np.count_nonzero(target_mask) < 0.12 * target_mask.size
    ):
        raise ValueError("Onvoldoende zichtbaar gras voor een grondvlakverbinding.")
    return _estimate_frame_edge(
        source_id,
        target_id,
        source,
        target,
        source_mask,
        target_mask,
    )


def _estimate_frame_edge(
    source_id: str,
    target_id: str,
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray | None,
    target_mask: np.ndarray | None,
) -> FrameGraphEdge:
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=10)
    first_points, first_descriptors = orb.detectAndCompute(
        cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), source_mask
    )
    second_points, second_descriptors = orb.detectAndCompute(
        cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), target_mask
    )
    if first_descriptors is None or second_descriptors is None:
        raise ValueError("Onvoldoende kenmerken voor framegraph-verbinding.")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(first_descriptors, second_descriptors, k=2)
    good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < 0.72 * b.distance]
    if len(good) < 35:
        raise ValueError("Minder dan 35 betrouwbare frame-overeenkomsten.")
    source_points = np.float32([first_points[item.queryIdx].pt for item in good])
    target_points = np.float32([second_points[item.trainIdx].pt for item in good])
    matrix, mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 3.5)
    if matrix is None or mask is None:
        raise ValueError("Framegraph-homography kon niet worden bepaald.")
    selected = mask.ravel().astype(bool)
    inliers = int(np.count_nonzero(selected))
    ratio = inliers / len(good)
    source_coverage = _coverage(source_points[selected], source.shape[1], source.shape[0])
    target_coverage = _coverage(target_points[selected], target.shape[1], target.shape[0])
    projected = cv2.perspectiveTransform(source_points[selected].reshape(1, -1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - target_points[selected], axis=1)
    if inliers < 30 or ratio < 0.40 or min(source_coverage, target_coverage) < 0.055:
        raise ValueError("Framegraph-verbinding heeft onvoldoende verspreide inliers.")
    return FrameGraphEdge(
        source_id, target_id, matrix, len(good), inliers, float(ratio),
        source_coverage, target_coverage, float(np.median(errors)),
    )


def solve_global_frame_graph(
    nodes: tuple[FrameGraphNode, ...],
    edges: tuple[FrameGraphEdge, ...],
    reference_id: str,
    _pruning_rounds: int = 6,
    direction_constraints: tuple[GroundDirectionConstraint, ...] = (),
    reference_ground_to_image: np.ndarray | None = None,
    absolute_ground_constraints: tuple[AbsoluteGroundConstraint, ...] = (),
    absolute_ground_point_constraints: tuple[AbsoluteGroundPointConstraint, ...] = (),
    absolute_ground_line_constraints: tuple[AbsoluteGroundLineConstraint, ...] = (),
) -> GlobalFrameGraphSolution:
    node_ids = {item.node_id for item in nodes}
    if reference_id not in node_ids:
        raise ValueError("Referentienode ontbreekt in de framegraph.")
    adjacency: dict[str, list[tuple[str, np.ndarray]]] = {item: [] for item in node_ids}
    for edge in edges:
        if edge.source_id not in node_ids or edge.target_id not in node_ids:
            raise ValueError("Framegraph-edge verwijst naar een onbekende node.")
        adjacency[edge.source_id].append((edge.target_id, edge.source_to_target))
        adjacency[edge.target_id].append((edge.source_id, np.linalg.inv(edge.source_to_target)))
    transforms = {reference_id: np.eye(3, dtype=np.float64)}
    queue = [reference_id]
    while queue:
        current = queue.pop(0)
        for neighbour, current_to_neighbour in adjacency[current]:
            if neighbour in transforms:
                continue
            transforms[neighbour] = transforms[current] @ np.linalg.inv(current_to_neighbour)
            transforms[neighbour] /= transforms[neighbour][2, 2]
            queue.append(neighbour)
    connected = tuple(sorted(transforms))
    rejected = tuple(sorted(node_ids - transforms.keys()))
    variable_ids = tuple(item for item in connected if item != reference_id)
    index = {node_id: position for position, node_id in enumerate(variable_ids)}

    def pack(matrix: np.ndarray) -> np.ndarray:
        normalised = matrix / matrix[2, 2]
        return normalised.reshape(-1)[:8]

    def unpack(values: np.ndarray) -> dict[str, np.ndarray]:
        result = {reference_id: np.eye(3, dtype=np.float64)}
        for node_id, position in index.items():
            flat = np.append(values[position * 8:(position + 1) * 8], 1.0)
            result[node_id] = flat.reshape(3, 3)
        return result

    initial = np.concatenate([pack(transforms[item]) for item in variable_ids]) if variable_ids else np.empty(0)
    samples = np.asarray(((0.0, 0.0), (640.0, 0.0), (1280.0, 0.0), (0.0, 360.0), (640.0, 360.0), (1280.0, 360.0), (0.0, 720.0), (640.0, 720.0), (1280.0, 720.0)), dtype=np.float64)
    connected_edges = tuple(edge for edge in edges if edge.source_id in transforms and edge.target_id in transforms)
    active_constraints = tuple(
        item for item in direction_constraints if item.node_id in transforms
    )
    active_absolute = tuple(
        item for item in absolute_ground_constraints if item.node_id in transforms
    )
    active_points = tuple(
        item for item in absolute_ground_point_constraints if item.node_id in transforms
    )
    active_lines = tuple(
        item for item in absolute_ground_line_constraints if item.node_id in transforms
    )
    if (active_constraints or active_absolute or active_points or active_lines) and reference_ground_to_image is None:
        raise ValueError("Grondvoorwaarden vereisen de grondhomography van het referentieframe.")
    reference_h = None
    if reference_ground_to_image is not None:
        reference_h = np.asarray(reference_ground_to_image, dtype=np.float64)
        if reference_h.shape != (3, 3) or not np.all(np.isfinite(reference_h)):
            raise ValueError("Referentie-grondhomography moet een eindige 3x3-matrix zijn.")

    def direction_residual(current: dict[str, np.ndarray], constraint: GroundDirectionConstraint) -> float:
        assert reference_h is not None
        reference_vanishing = reference_h[:, 0 if constraint.family is GroundLineFamily.LONGITUDINAL else 1]
        node_vanishing = np.linalg.inv(current[constraint.node_id]) @ reference_vanishing
        if abs(float(node_vanishing[2])) < 1e-9:
            return 100.0
        point = node_vanishing / node_vanishing[2]
        return float(constraint.equation() @ point) * np.sqrt(max(constraint.weight, 0.05))

    def residual(values: np.ndarray) -> np.ndarray:
        current = unpack(values)
        parts = []
        for edge in connected_edges:
            direct = _project(samples, current[edge.source_id])
            via_edge = _project(_project(samples, edge.source_to_target), current[edge.target_id])
            scale = np.sqrt(max(edge.inliers * edge.inlier_ratio, 1.0)) / 20.0
            parts.append((direct - via_edge).reshape(-1) * scale)
        if active_constraints:
            parts.append(
                np.asarray(
                    [direction_residual(current, item) for item in active_constraints],
                    dtype=np.float64,
                )
            )
        if active_absolute:
            ground_samples = np.asarray(
                ((0.0, 0.0), (64.0, 0.0), (0.0, 42.5), (64.0, 42.5), (32.0, 21.25)),
                dtype=np.float64,
            )
            for constraint in active_absolute:
                predicted = _project(
                    ground_samples,
                    np.linalg.inv(current[constraint.node_id]) @ reference_h,
                )
                observed = _project(ground_samples, constraint.ground_to_image)
                parts.append(
                    (predicted - observed).reshape(-1)
                    * np.sqrt(max(constraint.weight, 0.05))
                )
        for constraint in active_points:
            predicted = _project(
                constraint.ground_points,
                np.linalg.inv(current[constraint.node_id]) @ reference_h,
            )
            parts.append(
                (predicted - constraint.image_points).reshape(-1)
                * np.sqrt(max(constraint.weight, 0.05))
            )
        for constraint in active_lines:
            predicted = _project(
                constraint.ground_points,
                np.linalg.inv(current[constraint.node_id]) @ reference_h,
            )
            homogeneous = np.column_stack((predicted, np.ones(len(predicted))))
            parts.append(
                (homogeneous @ constraint.image_line)
                * np.sqrt(max(constraint.weight, 0.05))
            )
        return np.concatenate(parts) if parts else np.empty(0)

    if len(initial) and connected_edges:
        residuals_per_edge = len(samples) * 2
        edge_rows = len(connected_edges) * residuals_per_edge
        absolute_rows = len(active_absolute) * 10
        point_rows = sum(len(item.ground_points) * 2 for item in active_points)
        line_rows = sum(len(item.ground_points) for item in active_lines)
        sparsity = lil_matrix(
            (edge_rows + len(active_constraints) + absolute_rows + point_rows + line_rows, len(initial)),
            dtype=np.int8,
        )
        for edge_index, edge in enumerate(connected_edges):
            row_start = edge_index * residuals_per_edge
            row_end = row_start + residuals_per_edge
            for node_id in (edge.source_id, edge.target_id):
                if node_id == reference_id:
                    continue
                column_start = index[node_id] * 8
                sparsity[row_start:row_end, column_start:column_start + 8] = 1
        for constraint_index, constraint in enumerate(active_constraints):
            if constraint.node_id == reference_id:
                continue
            column_start = index[constraint.node_id] * 8
            sparsity[edge_rows + constraint_index, column_start:column_start + 8] = 1
        absolute_start = edge_rows + len(active_constraints)
        for constraint_index, constraint in enumerate(active_absolute):
            if constraint.node_id == reference_id:
                continue
            column_start = index[constraint.node_id] * 8
            row_start = absolute_start + constraint_index * 10
            sparsity[row_start:row_start + 10, column_start:column_start + 8] = 1
        point_start = absolute_start + absolute_rows
        for constraint in active_points:
            rows = len(constraint.ground_points) * 2
            if constraint.node_id != reference_id:
                column_start = index[constraint.node_id] * 8
                sparsity[point_start:point_start + rows, column_start:column_start + 8] = 1
            point_start += rows
        line_start = absolute_start + absolute_rows + point_rows
        for constraint in active_lines:
            rows = len(constraint.ground_points)
            if constraint.node_id != reference_id:
                column_start = index[constraint.node_id] * 8
                sparsity[line_start:line_start + rows, column_start:column_start + 8] = 1
            line_start += rows
        optimum = least_squares(
            residual,
            initial,
            jac_sparsity=sparsity.tocsr(),
            tr_solver="lsmr",
            x_scale="jac",
            loss="soft_l1",
            f_scale=3.0,
            max_nfev=150,
        )
        transforms = unpack(optimum.x)
    edge_errors = _edge_consistency_errors(transforms, connected_edges, samples)
    if _pruning_rounds > 0 and edge_errors:
        retained = tuple(
            edge for edge, error in zip(connected_edges, edge_errors)
            if error <= 8.0
        )
        if len(retained) < len(connected_edges) and retained:
            pruned = solve_global_frame_graph(
                nodes,
                retained,
                reference_id,
                _pruning_rounds - 1,
                direction_constraints,
                reference_ground_to_image,
                absolute_ground_constraints,
                absolute_ground_point_constraints,
                absolute_ground_line_constraints,
            )
            required_nodes = {
                item.node_id
                for item in (
                    *active_constraints,
                    *active_absolute,
                    *active_points,
                    *active_lines,
                )
            }
            if required_nodes <= set(pruned.connected_nodes):
                return pruned
    errors = np.asarray(edge_errors if edge_errors else (0.0,), dtype=np.float64)
    return GlobalFrameGraphSolution(
        reference_id,
        transforms,
        float(np.sqrt(np.mean(np.square(errors)))),
        float(np.max(errors)),
        connected,
        rejected,
        len(connected_edges),
    )


def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (homography @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:3]


def _coverage(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3:
        return 0.0
    return abs(float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))) / max(float(width * height), 1.0)


def _edge_consistency_errors(
    transforms: dict[str, np.ndarray],
    edges: tuple[FrameGraphEdge, ...],
    samples: np.ndarray,
) -> list[float]:
    errors = []
    for edge in edges:
        direct = _project(samples, transforms[edge.source_id])
        via_edge = _project(_project(samples, edge.source_to_target), transforms[edge.target_id])
        errors.append(float(np.sqrt(np.mean(np.sum(np.square(direct - via_edge), axis=1)))))
    return errors
