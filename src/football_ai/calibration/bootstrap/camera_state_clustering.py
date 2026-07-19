from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from football_ai.calibration.bootstrap.frame_sampler import BootstrapFrameSample


@dataclass(frozen=True, slots=True)
class CameraStateCluster:
    cluster_id: int
    sample_indices: tuple[int, ...]
    representative_sample_index: int
    mean_distance: float
    maximum_distance: float
    support_ratio: float
    stable: bool


@dataclass(frozen=True, slots=True)
class CameraStateClustering:
    labels: tuple[int, ...]
    clusters: tuple[CameraStateCluster, ...]
    separation_score: float


def cluster_camera_states(
    samples: list[BootstrapFrameSample],
    requested_cluster_count: int = 5,
) -> CameraStateClustering:
    """Groepeer frames op globale beeldinhoud, robuust tegen kleine spelers."""
    if len(samples) < 3:
        raise ValueError("Minimaal drie frames nodig voor cameraclustering.")
    features = np.vstack([_camera_view_descriptor(item.frame) for item in samples])
    features = _standardize(features)
    unique_count = len(np.unique(np.round(features, decimals=6), axis=0))
    cluster_count = min(max(1, requested_cluster_count), len(samples), unique_count)
    labels, centers = _deterministic_kmeans(features, cluster_count)
    clusters: list[CameraStateCluster] = []
    for cluster_id in range(cluster_count):
        indices = np.flatnonzero(labels == cluster_id)
        distances = np.linalg.norm(features[indices] - centers[cluster_id], axis=1)
        representative = int(indices[int(np.argmin(distances))])
        clusters.append(
            CameraStateCluster(
                cluster_id=cluster_id + 1,
                sample_indices=tuple(int(index) for index in indices),
                representative_sample_index=representative,
                mean_distance=float(np.mean(distances)),
                maximum_distance=float(np.max(distances)),
                support_ratio=len(indices) / len(samples),
                stable=(len(indices) / len(samples)) >= 0.05,
            )
        )
    own_distances = np.linalg.norm(features - centers[labels], axis=1)
    if cluster_count == 1:
        return CameraStateClustering(
            labels=tuple(1 for _item in samples),
            clusters=tuple(clusters),
            separation_score=0.0,
        )
    other_distances = np.full(len(samples), np.inf, dtype=np.float64)
    for cluster_id in range(cluster_count):
        distances = np.linalg.norm(features - centers[cluster_id], axis=1)
        other_distances = np.minimum(
            other_distances,
            np.where(labels == cluster_id, np.inf, distances),
        )
    separation = np.mean(
        np.clip((other_distances - own_distances) / np.maximum(other_distances, 1e-9), 0.0, 1.0)
    )
    return CameraStateClustering(
        labels=tuple(int(value) + 1 for value in labels),
        clusters=tuple(clusters),
        separation_score=float(separation),
    )


def _camera_view_descriptor(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140).astype(np.float32) / 255.0
    # Lage resolutie onderdrukt spelers, maar behoudt tribunes, bomen, doelen
    # en de globale ligging van het veld in beeld.
    return np.concatenate([lab.reshape(-1), edges.reshape(-1) * 0.35])


def _standardize(features: np.ndarray) -> np.ndarray:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1e-5] = 1.0
    return (features - mean) / scale


def _deterministic_kmeans(
    features: np.ndarray,
    cluster_count: int,
    maximum_iterations: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    first = int(np.argmax(np.linalg.norm(features - mean, axis=1)))
    centers = [features[first].copy()]
    while len(centers) < cluster_count:
        distances = np.min(
            np.vstack([np.linalg.norm(features - center, axis=1) for center in centers]),
            axis=0,
        )
        centers.append(features[int(np.argmax(distances))].copy())
    center_array = np.vstack(centers)
    labels = np.zeros(len(features), dtype=np.int32)
    for _iteration in range(maximum_iterations):
        distances = np.stack(
            [np.linalg.norm(features - center, axis=1) for center in center_array],
            axis=1,
        )
        new_labels = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels) and _iteration > 0:
            break
        labels = new_labels
        for cluster_id in range(cluster_count):
            members = features[labels == cluster_id]
            if len(members):
                center_array[cluster_id] = np.mean(members, axis=0)
    return labels, center_array
