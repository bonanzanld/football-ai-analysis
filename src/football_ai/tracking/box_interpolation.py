from __future__ import annotations

from typing import Any


Box = tuple[float, float, float, float]


def observations_with_short_gaps(
    track: Any,
    maximum_gap_frames: int = 8,
    maximum_center_step_ratio: float = 0.45,
) -> list[tuple[int, Box]]:
    """Return observed boxes plus conservative interpolation of short gaps.

    A missing detector result must not immediately make a known player vanish.
    We only bridge a gap when the same technical track exists on both sides and
    its box size and displacement remain physically plausible. Nothing is
    extrapolated before the first or after the final real observation.
    """

    observed = [
        (int(frame), tuple(float(value) for value in box))
        for frame, box in zip(track.observation_frames, track.boxes, strict=True)
    ]
    if len(observed) < 2:
        return observed

    result: list[tuple[int, Box]] = []
    for (first_frame, first_box), (second_frame, second_box) in zip(
        observed,
        observed[1:],
        strict=False,
    ):
        result.append((first_frame, first_box))
        missing = second_frame - first_frame - 1
        if missing <= 0 or missing > maximum_gap_frames:
            continue
        if not _gap_is_plausible(
            first_box,
            second_box,
            second_frame - first_frame,
            maximum_center_step_ratio,
        ):
            continue
        for offset in range(1, missing + 1):
            amount = offset / float(second_frame - first_frame)
            result.append(
                (
                    first_frame + offset,
                    tuple(
                        start + amount * (end - start)
                        for start, end in zip(first_box, second_box, strict=True)
                    ),
                )
            )
    result.append(observed[-1])
    return result


def _gap_is_plausible(
    first: Box,
    second: Box,
    frame_distance: int,
    maximum_center_step_ratio: float,
) -> bool:
    first_width = max(1.0, first[2] - first[0])
    first_height = max(1.0, first[3] - first[1])
    second_width = max(1.0, second[2] - second[0])
    second_height = max(1.0, second[3] - second[1])
    width_ratio = second_width / first_width
    height_ratio = second_height / first_height
    if not 0.60 <= width_ratio <= 1.67 or not 0.60 <= height_ratio <= 1.67:
        return False

    first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
    second_center = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
    displacement = (
        (second_center[0] - first_center[0]) ** 2
        + (second_center[1] - first_center[1]) ** 2
    ) ** 0.5
    reference_height = (first_height + second_height) / 2.0
    allowed = frame_distance * max(12.0, reference_height * maximum_center_step_ratio)
    return displacement <= allowed
