from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from football_ai.calibration.camera_anchor_recognition import (
    AnchorRecognition,
    CameraAnchorRecognizer,
)
from football_ai.calibration.global_frame_graph import estimate_frame_edge, estimate_ground_frame_edge
from football_ai.calibration.local_field_atlas import LocalFieldAtlas, LocalFieldPatch


FIELD_CORNERS = ((0.0, 0.0), (64.0, 0.0), (64.0, 42.5), (0.0, 42.5))
MINIMUM_RUNTIME_INLIERS = 60
MINIMUM_RUNTIME_INLIER_RATIO = 0.55


@dataclass(frozen=True, slots=True)
class AtlasRuntimeProjection:
    valid: bool
    patch_id: str | None
    ground_to_frame: np.ndarray | None
    anchor_to_frame: np.ndarray | None
    polygon: tuple[tuple[float, float], ...]
    recognition: AnchorRecognition
    inliers: int
    inlier_ratio: float
    coverage: float
    predicted_vanishing_point: tuple[float, float] | None
    reason: str


class LocalFieldAtlasRuntime:
    """Move immutable local field patches directly into nearby video frames."""

    def __init__(
        self,
        atlas: LocalFieldAtlas,
        anchor_frames: dict[str, np.ndarray],
    ) -> None:
        missing = {patch.patch_id for patch in atlas.patches} - anchor_frames.keys()
        if missing:
            raise ValueError(f"Atlasankerbeelden ontbreken: {', '.join(sorted(missing))}")
        self.atlas = atlas
        self.anchor_frames = anchor_frames
        self.patch_by_id = {patch.patch_id: patch for patch in atlas.patches}
        self.recognizer = CameraAnchorRecognizer.from_frames(anchor_frames)

    def project(self, frame: np.ndarray) -> AtlasRuntimeProjection:
        recognition = self.recognizer.recognize(frame)
        ranked = sorted(recognition.scores, key=lambda item: item.score, reverse=True)
        failures = []
        for score in ranked[: min(3, len(ranked))]:
            result = self.project_with_patch(frame, score.anchor_id, recognition)
            if result.valid:
                return result
            failures.append(result.reason)
        reason = failures[0] if failures else recognition.reason
        return AtlasRuntimeProjection(
            False, None, None, None, (), recognition, 0, 0.0, 0.0, None, reason
        )

    def project_with_patch(
        self,
        frame: np.ndarray,
        patch_id: str,
        recognition: AnchorRecognition,
    ) -> AtlasRuntimeProjection:
        if patch_id not in self.patch_by_id:
            raise ValueError(f"Onbekend atlasvlak: {patch_id}")
        patch = self.patch_by_id[patch_id]
        anchor = self.anchor_frames[patch_id]
        try:
            try:
                edge = estimate_ground_frame_edge(patch_id, "frame", anchor, frame)
            except ValueError:
                edge = estimate_frame_edge(patch_id, "frame", anchor, frame)
        except ValueError as error:
            return self._failure(patch_id, recognition, str(error))
        coverage = min(edge.source_coverage, edge.target_coverage)
        if (
            edge.inliers < MINIMUM_RUNTIME_INLIERS
            or edge.inlier_ratio < MINIMUM_RUNTIME_INLIER_RATIO
        ):
            return self._failure(
                patch_id,
                recognition,
                (
                    "Atlasanker heeft onvoldoende betrouwbare beeldsteun: "
                    f"{edge.inliers} inliers, {edge.inlier_ratio:.0%} ratio, "
                    f"{coverage:.0%} dekking."
                ),
                edge.inliers,
                edge.inlier_ratio,
                coverage,
            )
        matrix = edge.source_to_target @ patch.ground_to_anchor
        matrix /= matrix[2, 2]
        corners = np.asarray(
            (
                (0.0, 0.0),
                (self.atlas.pitch_length_m, 0.0),
                (self.atlas.pitch_length_m, self.atlas.pitch_width_m),
                (0.0, self.atlas.pitch_width_m),
            ),
            dtype=np.float64,
        )
        polygon = _project(corners, matrix)
        support = _project(np.asarray(patch.support_polygon, dtype=np.float64), matrix)
        if (
            not np.all(np.isfinite(support))
            or not cv2.isContourConvex(support.astype(np.float32).reshape(-1, 1, 2))
            or abs(float(cv2.contourArea(support.astype(np.float32)))) < 0.002 * frame.shape[0] * frame.shape[1]
        ):
            return self._failure(
                patch_id, recognition,
                "Het zichtbare lokale atlasvlak vormt geen betrouwbare convexe projectie.",
                edge.inliers,
                edge.inlier_ratio, coverage,
            )
        # Slicing returns a view.  Normalising that view in place would also
        # rescale the first homography column and corrupt metric projection.
        vanishing = matrix[:, 0].copy()
        if abs(float(vanishing[2])) < 1e-9:
            return self._failure(
                patch_id, recognition, "Atlas-zijlijnen hebben geen eindig verdwijnpunt.",
                edge.inliers, edge.inlier_ratio,
                coverage,
            )
        vanishing /= vanishing[2]
        return AtlasRuntimeProjection(
            True,
            patch_id,
            matrix,
            edge.source_to_target,
            tuple(tuple(map(float, point)) for point in polygon),
            recognition,
            edge.inliers,
            edge.inlier_ratio,
            coverage,
            tuple(map(float, vanishing[:2])),
            "Lokaal atlasvlak rechtstreeks naar huidig frame gekoppeld.",
        )

    def propagate(
        self,
        previous_frame: np.ndarray,
        frame: np.ndarray,
        previous: AtlasRuntimeProjection,
    ) -> AtlasRuntimeProjection:
        """Move a previously accepted field plane one nearby frame forward."""
        if not previous.valid or previous.ground_to_frame is None or previous.patch_id is None:
            return self._failure(
                previous.patch_id or "midfield", previous.recognition,
                "Geen geldige vorige atlasprojectie om temporeel voort te zetten.",
            )
        edges = []
        errors = []
        for estimator in (estimate_ground_frame_edge, estimate_frame_edge):
            try:
                edges.append(estimator("previous", "frame", previous_frame, frame))
            except ValueError as error:
                errors.append(str(error))
        if not edges:
            return self._failure(
                previous.patch_id, previous.recognition,
                errors[0] if errors else "Temporele framekoppeling mislukte.",
            )
        edge = max(
            edges,
            key=lambda item: (
                item.inliers >= MINIMUM_RUNTIME_INLIERS
                and item.inlier_ratio >= MINIMUM_RUNTIME_INLIER_RATIO,
                item.inlier_ratio,
                item.inliers,
            ),
        )
        coverage = min(edge.source_coverage, edge.target_coverage)
        if edge.inliers < MINIMUM_RUNTIME_INLIERS or edge.inlier_ratio < MINIMUM_RUNTIME_INLIER_RATIO:
            return self._failure(
                previous.patch_id, previous.recognition,
                "Opeenvolgende videoframes hebben onvoldoende betrouwbare grondsteun.",
                edge.inliers, edge.inlier_ratio, coverage,
            )
        matrix = edge.source_to_target @ previous.ground_to_frame
        matrix /= matrix[2, 2]
        patch = self.patch_by_id[previous.patch_id]
        support = _project(np.asarray(patch.support_polygon, dtype=np.float64), matrix)
        if (
            not np.all(np.isfinite(support))
            or not cv2.isContourConvex(support.astype(np.float32).reshape(-1, 1, 2))
            or abs(float(cv2.contourArea(support.astype(np.float32))))
            < 0.002 * frame.shape[0] * frame.shape[1]
        ):
            return self._failure(
                previous.patch_id, previous.recognition,
                "De temporeel voortgezette grondprojectie is niet convex.",
                edge.inliers, edge.inlier_ratio, coverage,
            )
        corners = np.asarray(
            ((0.0, 0.0), (self.atlas.pitch_length_m, 0.0),
             (self.atlas.pitch_length_m, self.atlas.pitch_width_m),
             (0.0, self.atlas.pitch_width_m)), dtype=np.float64,
        )
        polygon = _project(corners, matrix)
        vanishing = matrix[:, 0].copy()
        if abs(float(vanishing[2])) < 1e-9:
            return self._failure(previous.patch_id, previous.recognition, "Verdwijnpunt ligt op oneindig.")
        vanishing /= vanishing[2]
        return AtlasRuntimeProjection(
            True, previous.patch_id, matrix, edge.source_to_target,
            tuple(tuple(map(float, point)) for point in polygon),
            previous.recognition, edge.inliers, edge.inlier_ratio, coverage,
            tuple(map(float, vanishing[:2])),
            "Veldvlak temporeel voortgezet vanaf het vorige videoframe.",
        )

    @staticmethod
    def _failure(
        patch_id: str,
        recognition: AnchorRecognition,
        reason: str,
        inliers: int = 0,
        ratio: float = 0.0,
        coverage: float = 0.0,
    ) -> AtlasRuntimeProjection:
        return AtlasRuntimeProjection(
            False, patch_id, None, None, (), recognition, inliers, ratio, coverage,
            None, reason,
        )


SwitchValidator = Callable[[np.ndarray, AtlasRuntimeProjection], bool]


class LocalFieldAtlasTracker:
    """Stateful camera tracking around one immutable metric field model.

    Visual registration may move the current projection, but it cannot change
    its semantic atlas patch. A patch switch requires explicit approval from a
    caller that can verify fixed field evidence (for example a goal/end line).
    """

    def __init__(
        self,
        runtime: LocalFieldAtlasRuntime,
        switch_validator: SwitchValidator | None = None,
    ) -> None:
        self.runtime = runtime
        self.switch_validator = switch_validator
        self.previous_frame: np.ndarray | None = None
        self.current: AtlasRuntimeProjection | None = None

    def reset(self) -> None:
        self.previous_frame = None
        self.current = None

    def update(self, frame: np.ndarray) -> AtlasRuntimeProjection:
        direct = self.runtime.project(frame)
        if self.current is None or self.previous_frame is None:
            return self._bootstrap(frame, direct)

        temporal = self.runtime.propagate(self.previous_frame, frame, self.current)
        if temporal.valid:
            if direct.valid and direct.patch_id == self.current.patch_id:
                return self._accept(frame, direct)
            if self._switch_allowed(frame, direct):
                return self._accept(frame, direct)
            return self._accept(frame, temporal)

        recognition = direct.recognition
        same_patch = self.runtime.project_with_patch(
            frame, self.current.patch_id, recognition
        )
        if same_patch.valid:
            return self._accept(frame, same_patch)
        alternatives = []
        for patch_id in self.runtime.patch_by_id:
            if patch_id == self.current.patch_id:
                continue
            candidate = self.runtime.project_with_patch(frame, patch_id, recognition)
            if self._switch_allowed(frame, candidate):
                alternatives.append(candidate)
        if alternatives:
            best = max(
                alternatives,
                key=lambda item: (item.coverage, item.inlier_ratio, item.inliers),
            )
            return self._accept(frame, best)
        return AtlasRuntimeProjection(
            False,
            self.current.patch_id,
            None,
            None,
            (),
            recognition,
            max(temporal.inliers, same_patch.inliers),
            max(temporal.inlier_ratio, same_patch.inlier_ratio),
            max(temporal.coverage, same_patch.coverage),
            None,
            "Camerapositie tijdelijk onbekend; geen onbewezen atlaswissel uitgevoerd.",
        )

    def _switch_allowed(
        self,
        frame: np.ndarray,
        candidate: AtlasRuntimeProjection,
    ) -> bool:
        if not candidate.valid or candidate.patch_id == self.current.patch_id:
            return False
        return self.switch_validator is not None and self.switch_validator(frame, candidate)

    def _bootstrap(
        self,
        frame: np.ndarray,
        direct: AtlasRuntimeProjection,
    ) -> AtlasRuntimeProjection:
        if self.switch_validator is None:
            return self._accept(frame, direct)
        if direct.valid and self.switch_validator(frame, direct):
            return self._accept(frame, direct)
        recognition = direct.recognition
        candidates = []
        for patch_id in self.runtime.patch_by_id:
            candidate = self.runtime.project_with_patch(frame, patch_id, recognition)
            if candidate.valid and self.switch_validator(frame, candidate):
                candidates.append(candidate)
        if candidates:
            best = max(
                candidates,
                key=lambda item: (item.coverage, item.inlier_ratio, item.inliers),
            )
            return self._accept(frame, best)
        return AtlasRuntimeProjection(
            False, None, None, None, (), recognition, 0, 0.0, 0.0, None,
            "Eerste camerapositie heeft nog geen semantisch bevestigd veldanker.",
        )

    def _accept(
        self,
        frame: np.ndarray,
        projection: AtlasRuntimeProjection,
    ) -> AtlasRuntimeProjection:
        if projection.valid:
            self.previous_frame = frame.copy()
            self.current = projection
        return projection


class FixedPatchTracker:
    """Track one semantic atlas patch without ever changing its identity.

    This is used for physical boundaries that have one explicit owner.  Visual
    registration may move that owner through time, but a failed registration
    can neither substitute another patch nor invent an inferred boundary.
    """

    def __init__(self, runtime: LocalFieldAtlasRuntime, patch_id: str) -> None:
        if patch_id not in runtime.patch_by_id:
            raise ValueError(f"Onbekend atlasvlak: {patch_id}")
        self.runtime = runtime
        self.patch_id = patch_id
        self.previous_frame: np.ndarray | None = None
        self.current: AtlasRuntimeProjection | None = None

    def reset(self) -> None:
        self.previous_frame = None
        self.current = None

    def update(self, frame: np.ndarray) -> AtlasRuntimeProjection:
        if self.current is not None and self.previous_frame is not None:
            temporal = self.runtime.propagate(self.previous_frame, frame, self.current)
            if temporal.valid:
                return self._accept(frame, temporal)
            # A direct image match can recover direction while landing on the
            # wrong parallel line.  Once a physical boundary is established,
            # never re-seed its position from appearance alone.  Recovery must
            # come from continuous motion or a separate semantic validator.
            return AtlasRuntimeProjection(
                False, self.patch_id, None, None, (), temporal.recognition,
                temporal.inliers, temporal.inlier_ratio, temporal.coverage, None,
                "Vaste veldgrens verloor temporele positie; geen visuele herstart toegestaan.",
            )
        recognition = self.runtime.recognizer.recognize(frame)
        direct = self.runtime.project_with_patch(frame, self.patch_id, recognition)
        if direct.valid:
            return self._accept(frame, direct)
        return AtlasRuntimeProjection(
            False, self.patch_id, None, None, (), recognition,
            direct.inliers, direct.inlier_ratio, direct.coverage, None,
            "Vaste veldgrens tijdelijk niet betrouwbaar gekoppeld.",
        )

    def _accept(
        self,
        frame: np.ndarray,
        projection: AtlasRuntimeProjection,
    ) -> AtlasRuntimeProjection:
        self.previous_frame = frame.copy()
        self.current = projection
        return projection


def sideline_vanishing_error_degrees(
    predicted: tuple[float, float],
    observed: tuple[float, float],
    principal_point: tuple[float, float],
) -> float:
    center = np.asarray(principal_point, dtype=np.float64)
    first = np.asarray(predicted, dtype=np.float64) - center
    second = np.asarray(observed, dtype=np.float64) - center
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-9:
        return float("inf")
    # Line directions are unoriented: 0 and 180 degrees describe the same
    # perspective axis in the image.
    cosine = np.clip(abs(float(first @ second)) / denominator, 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (matrix @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) < 1e-9):
        raise ValueError("Atlascontour projecteert naar oneindig.")
    return projected[:, :2] / projected[:, 2:3]
