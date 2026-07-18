from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldLineDefinition:
    key: int
    name: str
    equation: tuple[float, float, float]
    instruction: str


@dataclass(frozen=True)
class LinePointObservation:
    line_key: int
    image_point: tuple[float, float]


@dataclass(frozen=True)
class FittedImageLine:
    equation: tuple[float, float, float]
    inlier_mask: tuple[bool, ...]
    rms_error_pixels: float


def fit_image_line_robustly(
    image_points: np.ndarray,
    distance_threshold_pixels: float = 4.0,
) -> FittedImageLine:
    """Pas robuust een beeldlijn door globale gebruikersklikken."""
    points = _points(image_points, "image_points")
    if len(points) < 3:
        raise ValueError("Klik minimaal drie punten op deze veldlijn.")
    if distance_threshold_pixels <= 0.0:
        raise ValueError("De afstandsdrempel moet groter zijn dan nul.")

    best_mask: np.ndarray | None = None
    best_rms = float("inf")
    for first_index in range(len(points) - 1):
        for second_index in range(first_index + 1, len(points)):
            line = _line_through_points(
                points[first_index],
                points[second_index],
            )
            if line is None:
                continue
            distances = np.abs(_homogeneous(points) @ line)
            mask = distances <= distance_threshold_pixels
            count = int(np.count_nonzero(mask))
            if count < 2:
                continue
            rms = float(np.sqrt(np.mean(np.square(distances[mask]))))
            if (
                best_mask is None
                or count > int(np.count_nonzero(best_mask))
                or (count == int(np.count_nonzero(best_mask)) and rms < best_rms)
            ):
                best_mask = mask
                best_rms = rms

    if best_mask is None or int(np.count_nonzero(best_mask)) < 3:
        raise ValueError(
            "De klikken vormen nog geen betrouwbare rechte lijn. "
            "Klik minimaal drie goed verspreide punten opnieuw."
        )

    line = _fit_line_tls(points[best_mask])
    distances = np.abs(_homogeneous(points) @ line)
    inliers = distances <= distance_threshold_pixels
    if int(np.count_nonzero(inliers)) < 3:
        raise ValueError("Minder dan drie lijnklikken zijn betrouwbaar.")
    line = _fit_line_tls(points[inliers])
    distances = np.abs(_homogeneous(points) @ line)
    inliers = distances <= distance_threshold_pixels
    rms = float(np.sqrt(np.mean(np.square(distances[inliers]))))
    return FittedImageLine(
        equation=(float(line[0]), float(line[1]), float(line[2])),
        inlier_mask=tuple(bool(value) for value in inliers),
        rms_error_pixels=rms,
    )


def filter_line_observations(
    observations: list[LinePointObservation],
    distance_threshold_pixels: float = 4.0,
) -> list[LinePointObservation]:
    """Verwijder duidelijke misklikken per bekende veldlijn."""
    filtered: list[LinePointObservation] = []
    for line_key in sorted({item.line_key for item in observations}):
        group = [item for item in observations if item.line_key == line_key]
        if len(group) < 3:
            filtered.extend(group)
            continue
        fitted = fit_image_line_robustly(
            np.asarray([item.image_point for item in group], dtype=np.float64),
            distance_threshold_pixels,
        )
        filtered.extend(
            item for item, is_inlier in zip(group, fitted.inlier_mask)
            if is_inlier
        )
    return filtered


def estimate_homography_with_line_constraints(
    image_points: np.ndarray,
    pitch_points: np.ndarray,
    line_observations: list[LinePointObservation],
    line_definitions: dict[int, FieldLineDefinition],
) -> np.ndarray:
    """Schat image-to-pitch met exacte punten en punten op bekende lijnen."""
    image = _points(image_points, "image_points")
    pitch = _points(pitch_points, "pitch_points")
    if len(image) != len(pitch):
        raise ValueError(
            "image_points en pitch_points moeten evenveel punten bevatten."
        )
    line_observations = filter_line_observations(line_observations)

    line_image = np.asarray(
        [observation.image_point for observation in line_observations],
        dtype=np.float64,
    ).reshape(-1, 2)
    all_image = (
        np.vstack([image, line_image])
        if len(line_image)
        else image
    )
    if len(all_image) < 4:
        raise ValueError("Er zijn onvoldoende observaties voor een homography.")

    world_samples = _world_normalisation_samples(
        pitch,
        line_observations,
        line_definitions,
    )
    image_transform = _normalisation_transform(all_image)
    world_transform = _normalisation_transform(world_samples)
    normalised_image = _transform_points(image, image_transform)
    normalised_pitch = _transform_points(pitch, world_transform)

    rows: list[np.ndarray] = []
    for image_point, pitch_point in zip(
        normalised_image,
        normalised_pitch,
    ):
        u, v = image_point
        x, y = pitch_point
        rows.append(
            np.array([-u, -v, -1.0, 0.0, 0.0, 0.0, x*u, x*v, x])
        )
        rows.append(
            np.array([0.0, 0.0, 0.0, -u, -v, -1.0, y*u, y*v, y])
        )

    inverse_world_transpose = np.linalg.inv(world_transform).T
    for observation in line_observations:
        definition = line_definitions.get(observation.line_key)
        if definition is None:
            raise ValueError(f"Onbekende veldlijn: {observation.line_key}.")
        point = _transform_points(
            np.asarray([observation.image_point], dtype=np.float64),
            image_transform,
        )[0]
        line = inverse_world_transpose @ np.asarray(
            definition.equation,
            dtype=np.float64,
        )
        line /= max(np.linalg.norm(line[:2]), 1e-12)
        homogeneous_point = np.array([point[0], point[1], 1.0])
        rows.append(np.kron(line, homogeneous_point))

    system = np.asarray(rows, dtype=np.float64)
    if system.shape[0] < 8 or np.linalg.matrix_rank(system) < 8:
        raise ValueError(
            "De observaties bepalen geen unieke homography. Voeg punten toe "
            "op meer verschillende veldlijnen of exacte landmarks."
        )

    _u, _singular_values, vh = np.linalg.svd(system)
    normalised_homography = vh[-1].reshape(3, 3)
    homography = (
        np.linalg.inv(world_transform)
        @ normalised_homography
        @ image_transform
    )
    if abs(homography[2, 2]) < 1e-12:
        raise ValueError("De geschatte homography heeft een ongeldige schaal.")
    homography /= homography[2, 2]
    if not np.all(np.isfinite(homography)):
        raise ValueError("De geschatte homography bevat ongeldige waarden.")
    return homography


def create_boundary_line_definitions(
    pitch_width: float,
    pitch_length: float,
) -> dict[int, FieldLineDefinition]:
    return {
        1: FieldLineDefinition(
            1,
            "Doellijn A (kaart links)",
            (0.0, 1.0, 0.0),
            "Klik 3+ punten op de korte lijn aan de linkerkant van de kaart.",
        ),
        2: FieldLineDefinition(
            2,
            "Doellijn B (kaart rechts)",
            (0.0, 1.0, -pitch_length),
            "Klik 3+ punten op de korte lijn aan de rechterkant van de kaart.",
        ),
        3: FieldLineDefinition(
            3,
            "Zijlijn boven (op kaart)",
            (1.0, 0.0, 0.0),
            "Klik 3+ verspreide punten op de bovenste lange zijlijn.",
        ),
        4: FieldLineDefinition(
            4,
            "Zijlijn onder (op kaart)",
            (1.0, 0.0, -pitch_width),
            "Klik 3+ verspreide punten op de onderste lange zijlijn.",
        ),
        5: FieldLineDefinition(
            5,
            "Middenlijn",
            (0.0, 1.0, -pitch_length / 2.0),
            "Klik 3+ verspreide punten op de middenlijn.",
        ),
    }


def _points(points: np.ndarray, name: str) -> np.ndarray:
    converted = np.asarray(points, dtype=np.float64)
    if converted.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if converted.ndim != 2 or converted.shape[1] != 2:
        raise ValueError(f"{name} moet de vorm (n, 2) hebben.")
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} bevat ongeldige waarden.")
    return converted


def _homogeneous(points: np.ndarray) -> np.ndarray:
    return np.column_stack([points, np.ones(len(points), dtype=np.float64)])


def _line_through_points(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray | None:
    line = np.cross(
        np.array([first[0], first[1], 1.0]),
        np.array([second[0], second[1], 1.0]),
    )
    normal = float(np.linalg.norm(line[:2]))
    return None if normal < 1e-9 else line / normal


def _fit_line_tls(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    _u, _singular_values, vh = np.linalg.svd(points - centroid)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    return np.array([normal[0], normal[1], -normal @ centroid])


def _world_normalisation_samples(
    pitch_points: np.ndarray,
    observations: list[LinePointObservation],
    definitions: dict[int, FieldLineDefinition],
) -> np.ndarray:
    samples = [point for point in pitch_points]
    for observation in observations:
        line = definitions[observation.line_key].equation
        a, b, c = line
        if abs(a) > abs(b):
            samples.append(np.array([-c/a, 0.0]))
            samples.append(np.array([-c/a, 1.0]))
        else:
            samples.append(np.array([0.0, -c/b]))
            samples.append(np.array([1.0, -c/b]))
    return np.asarray(samples, dtype=np.float64)


def _normalisation_transform(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    mean_distance = float(np.mean(distances))
    if mean_distance < 1e-9:
        raise ValueError("Observaties hebben onvoldoende ruimtelijke spreiding.")
    scale = np.sqrt(2.0) / mean_distance
    return np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.copy()
    homogeneous = np.column_stack([points, np.ones(len(points))])
    transformed = (transform @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]
