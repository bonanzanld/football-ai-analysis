from __future__ import annotations

import numpy as np


def constrain_homography_to_camera_rotation(
    image_homography: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """Project an image homography onto H = K R K^-1 for a fixed camera."""
    homography = np.asarray(image_homography, dtype=np.float64)
    intrinsics = np.asarray(camera_matrix, dtype=np.float64)
    if homography.shape != (3, 3) or intrinsics.shape != (3, 3):
        raise ValueError("Rotatieprojectie vereist twee 3x3-matrices.")
    if not np.all(np.isfinite(homography)) or not np.all(np.isfinite(intrinsics)):
        raise ValueError("Rotatieprojectie vereist eindige matrices.")
    if abs(float(np.linalg.det(intrinsics))) < 1e-12:
        raise ValueError("Camera-intrinsics zijn singulier.")
    normalized = np.linalg.inv(intrinsics) @ homography @ intrinsics
    scale = float(np.cbrt(abs(np.linalg.det(normalized))))
    if not np.isfinite(scale) or scale < 1e-12:
        raise ValueError("Beeldhomography kan niet als camerarotatie worden genormaliseerd.")
    normalized /= scale
    u, _singular, vh = np.linalg.svd(normalized)
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = np.sign(np.linalg.det(u @ vh))
    rotation = u @ correction @ vh
    constrained = intrinsics @ rotation @ np.linalg.inv(intrinsics)
    if abs(float(constrained[2, 2])) < 1e-12:
        raise ValueError("Camerarotatie projecteert het beeldvlak naar oneindig.")
    return constrained / constrained[2, 2]
