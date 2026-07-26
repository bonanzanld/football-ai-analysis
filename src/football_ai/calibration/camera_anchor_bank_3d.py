from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from football_ai.calibration.bootstrap.goal_seed import GoalSeed
from football_ai.calibration.camera_projection_3d import CameraProjection3D


@dataclass(frozen=True, slots=True)
class CameraAnchor3D:
    anchor_id: str
    goal_id: str
    frame_number: int
    time_seconds: float
    camera_state: int
    view_position: float | None
    projection: CameraProjection3D
    rms_error_px: float
    maximum_error_px: float
    anchor_type: str = "primary"
    parent_anchor_id: str | None = None
    local_inliers: int | None = None
    local_inlier_ratio: float | None = None
    local_coverage: float | None = None

    def __post_init__(self) -> None:
        if self.goal_id not in ("A", "B"):
            raise ValueError("Een camera-anker moet bij Doel A of Doel B horen.")
        if self.view_position is not None and not 0.0 <= self.view_position <= 1.0:
            raise ValueError("Camerapositie moet tussen 0 en 1 liggen.")
        if self.rms_error_px < 0.0 or self.maximum_error_px < 0.0:
            raise ValueError("Ankerfouten mogen niet negatief zijn.")
        if self.anchor_type not in ("primary", "intermediate"):
            raise ValueError("Ankertype moet primary of intermediate zijn.")
        if self.anchor_type == "intermediate" and self.parent_anchor_id is None:
            raise ValueError("Een tussenanker moet zijn primaire ouder vermelden.")
        self.projection.ground_homography()

    def to_dict(self) -> dict:
        return {
            "anchor_id": self.anchor_id,
            "goal_id": self.goal_id,
            "frame_number": self.frame_number,
            "time_seconds": self.time_seconds,
            "camera_state": self.camera_state,
            "view_position": self.view_position,
            "projection_matrix": self.projection.matrix.tolist(),
            "ground_homography": self.projection.ground_homography().tolist(),
            "rms_error_px": self.rms_error_px,
            "maximum_error_px": self.maximum_error_px,
            "anchor_type": self.anchor_type,
            "parent_anchor_id": self.parent_anchor_id,
            "local_inliers": self.local_inliers,
            "local_inlier_ratio": self.local_inlier_ratio,
            "local_coverage": self.local_coverage,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraAnchor3D":
        return cls(
            anchor_id=str(data["anchor_id"]),
            goal_id=str(data["goal_id"]).upper(),
            frame_number=int(data["frame_number"]),
            time_seconds=float(data["time_seconds"]),
            camera_state=int(data["camera_state"]),
            view_position=(float(data["view_position"]) if data.get("view_position") is not None else None),
            projection=CameraProjection3D(np.asarray(data["projection_matrix"], dtype=np.float64)),
            rms_error_px=float(data["rms_error_px"]),
            maximum_error_px=float(data["maximum_error_px"]),
            anchor_type=str(data.get("anchor_type", "primary")),
            parent_anchor_id=data.get("parent_anchor_id"),
            local_inliers=(int(data["local_inliers"]) if data.get("local_inliers") is not None else None),
            local_inlier_ratio=(float(data["local_inlier_ratio"]) if data.get("local_inlier_ratio") is not None else None),
            local_coverage=(float(data["local_coverage"]) if data.get("local_coverage") is not None else None),
        )


@dataclass(frozen=True, slots=True)
class CameraAnchorBank3D:
    match_format: str
    video_name: str
    pitch_length_m: float
    pitch_width_m: float
    anchors: tuple[CameraAnchor3D, ...]

    def __post_init__(self) -> None:
        if len(self.anchors) < 2:
            raise ValueError("Een ankerbank vereist minimaal twee camerastanden.")
        identifiers = [item.anchor_id for item in self.anchors]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Camera-anker-ID's moeten uniek zijn.")

    def nearest_view(self, view_position: float) -> CameraAnchor3D:
        if not 0.0 <= view_position <= 1.0:
            raise ValueError("Camerapositie moet tussen 0 en 1 liggen.")
        positioned = tuple(item for item in self.anchors if item.view_position is not None)
        if not positioned:
            raise ValueError("Ankerbank bevat geen ankers met een bekende kijkpositie.")
        return min(positioned, key=lambda item: abs(float(item.view_position) - view_position))

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "match_format": self.match_format,
            "video_name": self.video_name,
            "pitch_length_m": self.pitch_length_m,
            "pitch_width_m": self.pitch_width_m,
            "anchors": [item.to_dict() for item in self.anchors],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraAnchorBank3D":
        return cls(
            match_format=str(data["match_format"]),
            video_name=str(data["video_name"]),
            pitch_length_m=float(data["pitch_length_m"]),
            pitch_width_m=float(data["pitch_width_m"]),
            anchors=tuple(CameraAnchor3D.from_dict(item) for item in data["anchors"]),
        )


def build_camera_anchor(
    seed: GoalSeed,
    result_path: Path,
) -> CameraAnchor3D:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    if not data.get("solved", False) or "projection" not in data:
        raise ValueError(f"3D-resultaat is niet geldig opgelost: {result_path}")
    view = data["view"]
    if int(view["frame_number"]) != seed.frame_number:
        raise ValueError("Doel-seed en 3D-resultaat verwijzen niet naar hetzelfde frame.")
    projection = data["projection"]
    return CameraAnchor3D(
        anchor_id=f"goal-{seed.goal_id.lower()}",
        goal_id=seed.goal_id.upper(),
        frame_number=seed.frame_number,
        time_seconds=seed.time_seconds,
        camera_state=seed.camera_state,
        view_position=seed.view_position,
        projection=CameraProjection3D(np.asarray(projection["matrix"], dtype=np.float64)),
        rms_error_px=float(projection["rms_error_px"]),
        maximum_error_px=float(projection["maximum_error_px"]),
    )


def save_camera_anchor_bank(bank: CameraAnchorBank3D, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_camera_anchor_bank(path: Path) -> CameraAnchorBank3D:
    return CameraAnchorBank3D.from_dict(json.loads(path.read_text(encoding="utf-8")))


def refine_camera_anchor_bank_ground(
    bank: CameraAnchorBank3D,
    contour_report: dict,
) -> CameraAnchorBank3D:
    """Apply confirmed field-contour ground planes to primaries and descendants.

    The confirmed end lines/corners live in the refined ground homography from
    static contour QA. Intermediate anchors retain their direct image motion,
    but inherit that corrected metric ground plane from their primary parent.
    """
    by_id = {item.anchor_id: item for item in bank.anchors}
    refined_primary: dict[str, CameraAnchor3D] = {}
    for anchor in bank.anchors:
        if anchor.anchor_type != "primary":
            continue
        homography_data = (
            contour_report.get("parallelism_quality", {})
            .get(anchor.anchor_id, {})
            .get("refined_ground_homography")
        )
        if homography_data is None:
            refined_primary[anchor.anchor_id] = anchor
            continue
        refined_h = np.asarray(homography_data, dtype=np.float64)
        if refined_h.shape != (3, 3) or not np.all(np.isfinite(refined_h)):
            raise ValueError(f"Ongeldige verfijnde grondhomography voor {anchor.anchor_id}.")
        original_h = anchor.projection.ground_homography()
        scale = float(np.sum(original_h * refined_h) / max(np.sum(refined_h * refined_h), 1e-12))
        scaled_h = refined_h * scale
        matrix = anchor.projection.matrix.copy()
        matrix[:, (0, 1, 3)] = scaled_h
        refined_primary[anchor.anchor_id] = _copy_anchor(
            anchor,
            CameraProjection3D(matrix),
        )

    anchors = []
    for anchor in bank.anchors:
        if anchor.anchor_type == "primary":
            anchors.append(refined_primary[anchor.anchor_id])
            continue
        if anchor.parent_anchor_id is None or anchor.parent_anchor_id not in refined_primary:
            raise ValueError(f"Tussenanker {anchor.anchor_id} heeft geen geldige primaire ouder.")
        old_parent = by_id[anchor.parent_anchor_id]
        new_parent = refined_primary[anchor.parent_anchor_id]
        image_motion = (
            anchor.projection.ground_homography()
            @ np.linalg.inv(old_parent.projection.ground_homography())
        )
        projection = CameraProjection3D(image_motion @ new_parent.projection.matrix)
        anchors.append(_copy_anchor(anchor, projection))
    return CameraAnchorBank3D(
        bank.match_format,
        bank.video_name,
        bank.pitch_length_m,
        bank.pitch_width_m,
        tuple(anchors),
    )


def _copy_anchor(anchor: CameraAnchor3D, projection: CameraProjection3D) -> CameraAnchor3D:
    return CameraAnchor3D(
        anchor.anchor_id,
        anchor.goal_id,
        anchor.frame_number,
        anchor.time_seconds,
        anchor.camera_state,
        anchor.view_position,
        projection,
        anchor.rms_error_px,
        anchor.maximum_error_px,
        anchor.anchor_type,
        anchor.parent_anchor_id,
        anchor.local_inliers,
        anchor.local_inlier_ratio,
        anchor.local_coverage,
    )
