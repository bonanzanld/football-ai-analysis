from __future__ import annotations

import cv2
import numpy as np


def extract_shirt_feature(
    frame: np.ndarray,
    bounding_box: np.ndarray,
) -> np.ndarray | None:
    """
    Haalt een kleurprofiel uit het shirtgebied van een speler.

    Alleen het centrale bovenlichaam wordt gebruikt, zodat gras,
    benen, huid en achtergrond zo weinig mogelijk invloed hebben.
    """
    frame_height, frame_width = frame.shape[:2]

    x1, y1, x2, y2 = bounding_box.astype(int)

    x1 = max(
        0,
        min(x1, frame_width - 1),
    )
    x2 = max(
        0,
        min(x2, frame_width),
    )
    y1 = max(
        0,
        min(y1, frame_height - 1),
    )
    y2 = max(
        0,
        min(y2, frame_height),
    )

    box_width = x2 - x1
    box_height = y2 - y1

    if box_width < 8 or box_height < 16:
        return None

    shirt_x1 = x1 + int(
        box_width * 0.15
    )
    shirt_x2 = x2 - int(
        box_width * 0.15
    )

    shirt_y1 = y1 + int(
        box_height * 0.18
    )
    shirt_y2 = y1 + int(
        box_height * 0.55
    )

    if (
        shirt_x2 <= shirt_x1
        or shirt_y2 <= shirt_y1
    ):
        return None

    shirt_crop = frame[
        shirt_y1:shirt_y2,
        shirt_x1:shirt_x2,
    ]

    if shirt_crop.size == 0:
        return None

    hsv_crop = cv2.cvtColor(
        shirt_crop,
        cv2.COLOR_BGR2HSV,
    )

    saturation = hsv_crop[:, :, 1]
    value = hsv_crop[:, :, 2]

    valid_mask = (
        (value > 35)
        & (
            (saturation > 20)
            | (value > 110)
        )
    ).astype(np.uint8) * 255

    valid_pixel_count = int(
        cv2.countNonZero(valid_mask)
    )

    if valid_pixel_count < 10:
        return None

    hue_histogram = cv2.calcHist(
        [hsv_crop],
        [0],
        valid_mask,
        [18],
        [0, 180],
    )

    saturation_histogram = cv2.calcHist(
        [hsv_crop],
        [1],
        valid_mask,
        [8],
        [0, 256],
    )

    value_histogram = cv2.calcHist(
        [hsv_crop],
        [2],
        valid_mask,
        [8],
        [0, 256],
    )

    feature = np.concatenate(
        [
            hue_histogram.flatten(),
            saturation_histogram.flatten(),
            value_histogram.flatten(),
        ]
    ).astype(np.float32)

    feature_sum = float(
        feature.sum()
    )

    if feature_sum <= 0:
        return None

    feature /= feature_sum

    return feature