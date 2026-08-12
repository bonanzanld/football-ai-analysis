from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import PitchDetectionProfile
from football_ai.calibration.bootstrap.white_line_detection import detect_white_field_lines
from football_ai.calibration.camera_anchor_runtime import CameraAnchorRuntime
from football_ai.calibration.camera_projection_3d import CameraProjection3D
from football_ai.calibration.ground_line_evidence import detect_metric_ground_lines
from football_ai.calibration.global_ground_registration import GlobalGroundRegistration
from football_ai.calibration.image_line_perspective import estimate_sideline_perspective
from football_ai.calibration.local_anchor_projection import estimate_local_anchor_projection
from football_ai.calibration.perspective_parallelism import align_playable_sidelines_to_vanishing_point


FIELD_IDS = ("corner_a_rear", "corner_b_rear", "corner_b_front", "corner_a_front")


@dataclass(frozen=True, slots=True)
class PlannedProjection:
    time_seconds: float
    frame_number: int
    status: str
    anchor_id: str | None
    projection_matrix: tuple[tuple[float, ...], ...] | None
    reason: str
    inliers: int = 0
    inlier_ratio: float = 0.0
    coverage: float = 0.0
    supporting_line_count: int = 0
    supporting_line_length_m: float = 0.0

    @property
    def projection(self) -> CameraProjection3D | None:
        if self.projection_matrix is None:
            return None
        return CameraProjection3D(np.asarray(self.projection_matrix, dtype=np.float64))

    def to_dict(self) -> dict:
        return {
            "time_seconds": self.time_seconds,
            "frame_number": self.frame_number,
            "status": self.status,
            "anchor_id": self.anchor_id,
            "projection_matrix": self.projection_matrix,
            "reason": self.reason,
            "inliers": self.inliers,
            "inlier_ratio": self.inlier_ratio,
            "coverage": self.coverage,
            "supporting_line_count": self.supporting_line_count,
            "supporting_line_length_m": self.supporting_line_length_m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlannedProjection":
        matrix = data.get("projection_matrix")
        return cls(
            time_seconds=float(data["time_seconds"]),
            frame_number=int(data["frame_number"]),
            status=str(data["status"]),
            anchor_id=data.get("anchor_id"),
            projection_matrix=None if matrix is None else tuple(tuple(float(v) for v in row) for row in matrix),
            reason=str(data["reason"]),
            inliers=int(data.get("inliers", 0)),
            inlier_ratio=float(data.get("inlier_ratio", 0.0)),
            coverage=float(data.get("coverage", 0.0)),
            supporting_line_count=int(data.get("supporting_line_count", 0)),
            supporting_line_length_m=float(data.get("supporting_line_length_m", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class VideoProjectionPlan:
    video: str
    match_format: str
    start_seconds: float
    duration_seconds: float
    interval_seconds: float
    records: tuple[PlannedProjection, ...]

    @property
    def resolved_ratio(self) -> float:
        if not self.records:
            return 0.0
        return sum(item.projection_matrix is not None for item in self.records) / len(self.records)

    @property
    def trusted_ratio(self) -> float:
        if not self.records:
            return 0.0
        return sum(item.status == "valid" for item in self.records) / len(self.records)

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "video": self.video,
            "match_format": self.match_format,
            "start_seconds": self.start_seconds,
            "duration_seconds": self.duration_seconds,
            "interval_seconds": self.interval_seconds,
            "resolved_ratio": self.resolved_ratio,
            "trusted_ratio": self.trusted_ratio,
            "records": [item.to_dict() for item in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoProjectionPlan":
        return cls(
            video=str(data["video"]),
            match_format=str(data["match_format"]),
            start_seconds=float(data["start_seconds"]),
            duration_seconds=float(data["duration_seconds"]),
            interval_seconds=float(data["interval_seconds"]),
            records=tuple(PlannedProjection.from_dict(item) for item in data["records"]),
        )


def save_video_projection_plan(plan: VideoProjectionPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_video_projection_plan(path: Path) -> VideoProjectionPlan:
    return VideoProjectionPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))


def gate_projection_plan_with_player_evidence(
    plan: VideoProjectionPlan,
    classifications_by_frame: dict[int, str],
) -> VideoProjectionPlan:
    """Reject resolved field geometry contradicted by player footpoints.

    Player evidence is deliberately one-way: it may veto geometry, but it may
    never create a projection or promote a candidate to valid.
    """
    allowed = {"unavailable", "insufficient_evidence", "ambiguous", "supportive", "rejected"}
    unknown = set(classifications_by_frame.values()) - allowed
    if unknown:
        raise ValueError(f"Onbekende spelersclassificaties: {sorted(unknown)}")
    records = []
    for item in plan.records:
        classification = classifications_by_frame.get(item.frame_number)
        if classification == "rejected" and item.projection_matrix is not None:
            records.append(
                PlannedProjection(
                    item.time_seconds,
                    item.frame_number,
                    "unknown",
                    None,
                    None,
                    f"Spelervoetpunten verwerpen veldprojectie. Eerder: {item.reason}",
                    item.inliers,
                    item.inlier_ratio,
                    item.coverage,
                    item.supporting_line_count,
                    item.supporting_line_length_m,
                )
            )
        else:
            records.append(item)
    return VideoProjectionPlan(
        plan.video,
        plan.match_format,
        plan.start_seconds,
        plan.duration_seconds,
        plan.interval_seconds,
        tuple(records),
    )


class OfflineVideoProjectionAnalyzer:
    """Resolve an entire video section before any QA frames are rendered."""

    def __init__(
        self,
        runtime: CameraAnchorRuntime,
        profile: PitchDetectionProfile,
        trusted_anchors: set[str],
        global_registration: GlobalGroundRegistration | None = None,
    ) -> None:
        self.runtime = runtime
        self.profile = profile
        self.trusted_anchors = trusted_anchors
        self.global_registration = global_registration

    def analyze(
        self,
        capture: cv2.VideoCapture,
        video: str,
        match_format: str,
        start_seconds: float,
        duration_seconds: float,
        interval_seconds: float,
    ) -> VideoProjectionPlan:
        if self.global_registration is not None and self.global_registration.solved_for_playable_field:
            return self._analyze_from_global_registration(
                capture,
                video,
                match_format,
                start_seconds,
                duration_seconds,
                interval_seconds,
            )
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(np.floor(duration_seconds / interval_seconds)) + 1
        records: list[PlannedProjection] = []
        previous_polygon: np.ndarray | None = None
        for index in range(count):
            time_seconds = start_seconds + index * interval_seconds
            frame_number = int(round(time_seconds * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = capture.read()
            if not success:
                break
            recognition = self.runtime.recognizer.recognize(frame)
            ranked = sorted(recognition.scores, key=lambda item: item.score, reverse=True)[:3]
            candidates = []
            visual_lines = detect_white_field_lines(frame, self.profile)
            for rank, score in enumerate(ranked):
                resolved = self.runtime.project_with_anchor(frame, score.anchor_id, recognition)
                if not resolved.valid or resolved.projection is None or resolved.local is None:
                    continue
                polygon = np.asarray(
                    [resolved.projection.project(self.runtime.reference.landmark(item).point) for item in FIELD_IDS],
                    dtype=np.float64,
                )
                perspective = estimate_sideline_perspective(
                    visual_lines.candidates,
                    polygon,
                    (frame.shape[1], frame.shape[0]),
                )
                if perspective.valid and perspective.vanishing_point is not None:
                    anchor = self.runtime.anchor_by_id[score.anchor_id]
                    anchored_end = anchor.goal_id
                    try:
                        corrected_polygon = align_playable_sidelines_to_vanishing_point(
                            polygon,
                            perspective.vanishing_point,
                            anchored_end,
                        )
                    except ValueError:
                        corrected_polygon = None
                    if corrected_polygon is not None:
                        correction = float(np.mean(np.linalg.norm(corrected_polygon - polygon, axis=1)))
                    else:
                        correction = float("inf")
                    if corrected_polygon is not None and correction <= 0.08 * float(np.hypot(frame.shape[1], frame.shape[0])):
                        corrected_h = cv2.getPerspectiveTransform(
                            np.asarray(
                                [
                                    self.runtime.reference.landmark(item).point.as_tuple()[:2]
                                    for item in FIELD_IDS
                                ],
                                dtype=np.float32,
                            ),
                            corrected_polygon.astype(np.float32),
                        )
                        projection_matrix = resolved.projection.matrix.copy()
                        original_h = resolved.projection.ground_homography()
                        scale = float(
                            np.sum(original_h * corrected_h)
                            / max(np.sum(corrected_h * corrected_h), 1e-12)
                        )
                        projection_matrix[:, (0, 1, 3)] = corrected_h * scale
                        resolved = type(resolved)(
                            resolved.valid,
                            resolved.anchor_id,
                            CameraProjection3D(projection_matrix),
                            resolved.recognition,
                            resolved.local,
                            "Zijlijnen actief uitgelijnd met witte 11v11-lijnen.",
                        )
                        polygon = corrected_polygon
                continuity = 0.0
                if previous_polygon is not None:
                    continuity = float(np.mean(np.linalg.norm(polygon - previous_polygon, axis=1)))
                local = resolved.local
                quality = (
                    local.inliers
                    * local.inlier_ratio
                    * np.sqrt(max(local.anchor_coverage * local.frame_coverage, 0.0))
                )
                line_detection = detect_metric_ground_lines(
                    frame,
                    self.profile,
                    resolved.projection.ground_homography(),
                    minimum_length_m=3.0,
                )
                supporting_length = float(sum(item.metric_length for item in line_detection.lines))
                trusted_anchor = resolved.anchor_id in self.trusted_anchors
                candidates.append(
                    (
                        not trusted_anchor,
                        not bool(line_detection.lines),
                        continuity,
                        -supporting_length,
                        -quality,
                        rank,
                        resolved,
                        polygon,
                        line_detection,
                    )
                )
            if not candidates:
                records.append(
                    PlannedProjection(time_seconds, frame_number, "unknown", None, None, recognition.reason)
                )
                continue
            candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2] if previous_polygon is not None else 0.0,
                    item[3],
                    item[4],
                    item[5],
                )
            )
            (
                _untrusted,
                _unsupported,
                _continuity,
                _support,
                _quality,
                _rank,
                resolved,
                polygon,
                line_detection,
            ) = candidates[0]
            previous_polygon = polygon
            local = resolved.local
            assert resolved.projection is not None and local is not None
            supporting_length = float(sum(item.metric_length for item in line_detection.lines))
            line_supported = bool(line_detection.lines)
            trusted = resolved.anchor_id in self.trusted_anchors and line_supported
            anchor_approved = resolved.anchor_id in self.trusted_anchors
            if not anchor_approved and not line_supported:
                records.append(
                    PlannedProjection(
                        time_seconds,
                        frame_number,
                        "unknown",
                        None,
                        None,
                        "Niet-goedgekeurd anker zonder witte 11v11-lijnsteun verworpen.",
                    )
                )
                continue
            if trusted:
                reason = "Offline opgelost en bevestigd door witte 11v11-lijnsteun."
            elif resolved.anchor_id not in self.trusted_anchors:
                reason = "Geometrisch opgelost via nog niet goedgekeurd anker."
            else:
                reason = "Geometrisch opgelost, maar zonder passende witte lijn van minimaal 3 meter."
            records.append(
                PlannedProjection(
                    time_seconds,
                    frame_number,
                    "valid" if trusted else "candidate",
                    resolved.anchor_id,
                    tuple(tuple(float(v) for v in row) for row in resolved.projection.matrix),
                    reason,
                    local.inliers,
                    local.inlier_ratio,
                    min(local.anchor_coverage, local.frame_coverage),
                    len(line_detection.lines),
                    supporting_length,
                )
            )
        records = self._resolve_from_temporal_neighbors(
            capture,
            records,
            fps,
            maximum_distance_seconds=3.0,
        )
        return VideoProjectionPlan(
            video,
            match_format,
            start_seconds,
            duration_seconds,
            interval_seconds,
            tuple(records),
        )

    def _analyze_from_global_registration(
        self,
        capture: cv2.VideoCapture,
        video: str,
        match_format: str,
        start_seconds: float,
        duration_seconds: float,
        interval_seconds: float,
    ) -> VideoProjectionPlan:
        assert self.global_registration is not None
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(np.floor(duration_seconds / interval_seconds)) + 1
        records = []
        for index in range(count):
            time_seconds = start_seconds + index * interval_seconds
            frame_number = int(round(time_seconds * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = capture.read()
            if not success:
                break
            homography = self.global_registration.ground_to_image_at(time_seconds)
            matrix = np.zeros((3, 4), dtype=np.float64)
            matrix[:, 0] = homography[:, 0]
            matrix[:, 1] = homography[:, 1]
            matrix[:, 2] = (0.0, 0.0, 1.0)
            matrix[:, 3] = homography[:, 2]
            projection = CameraProjection3D(matrix)
            detection = detect_metric_ground_lines(
                frame,
                self.profile,
                homography,
                minimum_length_m=3.0,
            )
            support_length = float(sum(item.metric_length for item in detection.lines))
            directly_confirmed = bool(detection.lines)
            records.append(
                PlannedProjection(
                    time_seconds,
                    frame_number,
                    "valid" if directly_confirmed else "candidate",
                    "global-ground",
                    tuple(tuple(float(v) for v in row) for row in projection.matrix),
                    (
                        "Globaal grondvlak; lokaal bevestigd door witte 11v11-lijn."
                        if directly_confirmed
                        else "Globaal grondvlak geïnterpoleerd; in dit frame geen extra witte-lijnbevestiging."
                    ),
                    supporting_line_count=len(detection.lines),
                    supporting_line_length_m=support_length,
                )
            )
        return VideoProjectionPlan(
            video,
            match_format,
            start_seconds,
            duration_seconds,
            interval_seconds,
            tuple(records),
        )

    def _resolve_from_temporal_neighbors(
        self,
        capture: cv2.VideoCapture,
        records: list[PlannedProjection],
        fps: float,
        maximum_distance_seconds: float,
    ) -> list[PlannedProjection]:
        """Fill short gaps directly from nearby confirmed frames in either direction."""
        confirmed = [item for item in records if item.status == "valid" and item.projection is not None]
        if not confirmed:
            return records
        result = list(records)
        frame_cache: dict[int, np.ndarray] = {}

        def read_frame(frame_number: int) -> np.ndarray | None:
            if frame_number not in frame_cache:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = capture.read()
                if not success:
                    return None
                frame_cache[frame_number] = frame
            return frame_cache[frame_number]

        for index, current in enumerate(records):
            if current.status == "valid":
                continue
            neighbors = sorted(
                (
                    item
                    for item in confirmed
                    if abs(item.time_seconds - current.time_seconds) <= maximum_distance_seconds
                ),
                key=lambda item: abs(item.time_seconds - current.time_seconds),
            )[:2]
            target_frame = read_frame(current.frame_number)
            if target_frame is None:
                continue
            candidates = []
            for source in neighbors:
                source_frame = read_frame(source.frame_number)
                source_projection = source.projection
                if source_frame is None or source_projection is None:
                    continue
                local = estimate_local_anchor_projection(
                    source_frame,
                    target_frame,
                    source_projection,
                    self.runtime.reference,
                )
                if not local.valid or local.projection is None:
                    continue
                temporal_projection = local.projection
                polygon = np.asarray(
                    [temporal_projection.project(self.runtime.reference.landmark(item).point) for item in FIELD_IDS],
                    dtype=np.float64,
                )
                visual_lines = detect_white_field_lines(target_frame, self.profile)
                perspective = estimate_sideline_perspective(
                    visual_lines.candidates,
                    polygon,
                    (target_frame.shape[1], target_frame.shape[0]),
                )
                if perspective.valid and perspective.vanishing_point is not None:
                    source_anchor_id = source.anchor_id
                    if source_anchor_id is not None and source_anchor_id in self.runtime.anchor_by_id:
                        anchored_end = self.runtime.anchor_by_id[source_anchor_id].goal_id
                        try:
                            corrected_polygon = align_playable_sidelines_to_vanishing_point(
                                polygon,
                                perspective.vanishing_point,
                                anchored_end,
                            )
                        except ValueError:
                            corrected_polygon = None
                        if corrected_polygon is not None:
                            correction = float(np.mean(np.linalg.norm(corrected_polygon - polygon, axis=1)))
                        else:
                            correction = float("inf")
                        if corrected_polygon is not None and correction <= 0.08 * float(np.hypot(target_frame.shape[1], target_frame.shape[0])):
                            corrected_h = cv2.getPerspectiveTransform(
                                np.asarray(
                                    [
                                        self.runtime.reference.landmark(item).point.as_tuple()[:2]
                                        for item in FIELD_IDS
                                    ],
                                    dtype=np.float32,
                                ),
                                corrected_polygon.astype(np.float32),
                            )
                            matrix = temporal_projection.matrix.copy()
                            original_h = temporal_projection.ground_homography()
                            scale = float(
                                np.sum(original_h * corrected_h)
                                / max(np.sum(corrected_h * corrected_h), 1e-12)
                            )
                            matrix[:, (0, 1, 3)] = corrected_h * scale
                            temporal_projection = CameraProjection3D(matrix)
                line_detection = detect_metric_ground_lines(
                    target_frame,
                    self.profile,
                    temporal_projection.ground_homography(),
                    minimum_length_m=3.0,
                )
                support_length = float(sum(item.metric_length for item in line_detection.lines))
                quality = local.inliers * local.inlier_ratio * max(local.frame_coverage, 1e-6)
                candidates.append(
                    (-bool(line_detection.lines), -support_length, -quality, source, local, line_detection, temporal_projection)
                )
            if not candidates:
                continue
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            _supported, _length, _quality, source, local, line_detection, temporal_projection = candidates[0]
            line_supported = bool(line_detection.lines)
            result[index] = PlannedProjection(
                current.time_seconds,
                current.frame_number,
                "valid" if line_supported else "candidate",
                f"temporal-{source.frame_number}",
                tuple(tuple(float(v) for v in row) for row in temporal_projection.matrix),
                (
                    "Bidirectioneel opgelost en bevestigd door witte 11v11-lijnsteun."
                    if line_supported
                    else "Bidirectioneel opgelost vanaf nabij bevestigd beeld; witte lijnsteun ontbreekt."
                ),
                local.inliers,
                local.inlier_ratio,
                min(local.anchor_coverage, local.frame_coverage),
                len(line_detection.lines),
                float(sum(item.metric_length for item in line_detection.lines)),
            )
        return result
