from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CameraKind(StrEnum):
    UNKNOWN = "unknown"
    XBOTGO_FALCON = "xbotgo_falcon"
    IPHONE = "iphone"
    OTHER = "other"


class ZoomMode(StrEnum):
    FIXED = "fixed"
    AUTOMATIC = "automatic"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CameraProfile:
    """Optional camera knowledge used only as an optimisation prior."""

    kind: CameraKind = CameraKind.UNKNOWN
    display_name: str = "Onbekende camera"
    zoom_mode: ZoomMode = ZoomMode.UNKNOWN
    horizontal_fov_degrees: float | None = None
    expected_height_m: tuple[float, float] | None = None
    supports_multiple_lenses: bool = False

    def __post_init__(self) -> None:
        if self.horizontal_fov_degrees is not None and not 10.0 <= self.horizontal_fov_degrees <= 180.0:
            raise ValueError("De horizontale beeldhoek moet tussen 10 en 180 graden liggen.")
        if self.expected_height_m is not None:
            low, high = self.expected_height_m
            if low <= 0.0 or high < low:
                raise ValueError("Het verwachte hoogtebereik is ongeldig.")


def create_camera_profile(
    kind: CameraKind | str = CameraKind.UNKNOWN,
    *,
    zoom_mode: ZoomMode | str = ZoomMode.UNKNOWN,
) -> CameraProfile:
    camera_kind = CameraKind(kind)
    selected_zoom = ZoomMode(zoom_mode)
    if camera_kind is CameraKind.XBOTGO_FALCON:
        return CameraProfile(
            kind=camera_kind,
            display_name="XbotGo Falcon",
            zoom_mode=selected_zoom,
            horizontal_fov_degrees=106.0,
            expected_height_m=(3.0, 5.0),
            supports_multiple_lenses=True,
        )
    if camera_kind is CameraKind.IPHONE:
        return CameraProfile(
            kind=camera_kind,
            display_name="Apple iPhone",
            zoom_mode=selected_zoom,
            supports_multiple_lenses=True,
        )
    if camera_kind is CameraKind.OTHER:
        return CameraProfile(kind=camera_kind, display_name="Andere camera", zoom_mode=selected_zoom)
    return CameraProfile(zoom_mode=selected_zoom)
