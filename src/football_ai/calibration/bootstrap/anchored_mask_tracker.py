from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from football_ai.calibration.bootstrap.local_mask_tracker import LocalMaskTracker
from football_ai.calibration.bootstrap.visible_field_mask import clip_polygon_to_frame


@dataclass(frozen=True, slots=True)
class MaskAnchor:
    anchor_id: str
    frame_number: int
    frame: np.ndarray
    polygon: np.ndarray
    trusted: bool = False


@dataclass(frozen=True, slots=True)
class AnchoredTrackingResult:
    polygon: np.ndarray
    reliable: bool
    mode: str
    anchor_id: str
    tracked_points: int
    inlier_ratio: float


class AnchoredMaskTracker:
    """Track a field mask and recover from known or learned camera anchors."""

    def __init__(
        self,
        frame: np.ndarray,
        polygon: np.ndarray,
        frame_number: int,
        anchor_id: str,
        *,
        anchor_interval_frames: int = 30,
        recognition_interval_frames: int = 15,
        maximum_learned_anchors: int = 24,
    ) -> None:
        if anchor_interval_frames < 1:
            raise ValueError("anchor_interval_frames moet minimaal 1 zijn.")
        self.anchor_interval_frames = anchor_interval_frames
        self.recognition_interval_frames = recognition_interval_frames
        self.maximum_learned_anchors = maximum_learned_anchors
        self.frame_size = (frame.shape[1], frame.shape[0])
        self.local_tracker = self._new_local_tracker(frame, polygon)
        self.anchors: list[MaskAnchor] = []
        self.active_anchor_id = anchor_id
        self.last_anchor_frame = frame_number
        self.unreliable_streak = 0
        self.add_anchor(anchor_id, frame_number, frame, polygon, trusted=True)

    def add_anchor(
        self,
        anchor_id: str,
        frame_number: int,
        frame: np.ndarray,
        polygon: np.ndarray,
        *,
        trusted: bool = True,
    ) -> None:
        anchor = MaskAnchor(
            anchor_id=anchor_id,
            frame_number=frame_number,
            frame=frame.copy(),
            polygon=np.asarray(polygon, dtype=np.float64).copy(),
            trusted=trusted,
        )
        self.anchors = [item for item in self.anchors if item.anchor_id != anchor_id]
        self.anchors.append(anchor)
        self._trim_learned_anchors()

    def update(self, frame: np.ndarray, frame_number: int) -> AnchoredTrackingResult:
        local = self.local_tracker.update(frame)
        if local.reliable:
            self.unreliable_streak = 0
            if frame_number % self.recognition_interval_frames == 0:
                recognized = self._recover_from_anchor(frame, trusted_only=True)
                if recognized is not None and recognized[0].anchor_id != self.active_anchor_id:
                    anchor, polygon, tracked_points, inlier_ratio = recognized
                    self.local_tracker = self._new_local_tracker(frame, polygon)
                    self.active_anchor_id = anchor.anchor_id
                    self.last_anchor_frame = frame_number
                    return AnchoredTrackingResult(
                        polygon=self._visible_polygon(polygon),
                        reliable=True,
                        mode="anchor",
                        anchor_id=anchor.anchor_id,
                        tracked_points=tracked_points,
                        inlier_ratio=inlier_ratio,
                    )
            if frame_number - self.last_anchor_frame >= self.anchor_interval_frames:
                self._learn_anchor(frame, local.polygon, frame_number)
            return AnchoredTrackingResult(
                polygon=self._visible_polygon(local.polygon),
                reliable=True,
                mode="local",
                anchor_id=self.active_anchor_id,
                tracked_points=local.tracked_points,
                inlier_ratio=local.inlier_ratio,
            )

        self.unreliable_streak += 1
        recovered = self._recover_from_anchor(frame)
        if recovered is not None:
            anchor, polygon, tracked_points, inlier_ratio = recovered
            self.local_tracker = self._new_local_tracker(frame, polygon)
            self.active_anchor_id = anchor.anchor_id
            self.last_anchor_frame = frame_number
            self.unreliable_streak = 0
            return AnchoredTrackingResult(
                polygon=self._visible_polygon(polygon),
                reliable=True,
                mode="anchor",
                anchor_id=anchor.anchor_id,
                tracked_points=tracked_points,
                inlier_ratio=inlier_ratio,
            )
        return AnchoredTrackingResult(
            polygon=self._visible_polygon(local.polygon),
            reliable=False,
            mode="hold",
            anchor_id=self.active_anchor_id,
            tracked_points=local.tracked_points,
            inlier_ratio=local.inlier_ratio,
        )

    def _learn_anchor(self, frame: np.ndarray, polygon: np.ndarray, frame_number: int) -> None:
        anchor_id = f"auto-{frame_number}"
        self.add_anchor(anchor_id, frame_number, frame, polygon, trusted=False)
        self.active_anchor_id = anchor_id
        self.last_anchor_frame = frame_number

    def _recover_from_anchor(
        self,
        frame: np.ndarray,
        *,
        trusted_only: bool = False,
    ) -> tuple[MaskAnchor, np.ndarray, int, float] | None:
        candidates = self.anchors
        if trusted_only:
            candidates = [item for item in candidates if item.trusted]
        candidates = sorted(
            candidates,
            key=lambda item: (camera_view_distance(item.frame, frame), not item.trusted),
        )
        # Kunstgras bevat veel herhalende details. Eerst de camerastand kiezen op
        # globale beeldinhoud, daarna pas de grondgeometrie van die stand oplossen.
        candidates = candidates[:1] if trusted_only else candidates[:6]
        best: tuple[MaskAnchor, np.ndarray, int, float] | None = None
        best_score = -1.0
        for anchor in candidates:
            match = _match_anchor(anchor, frame)
            if match is None:
                continue
            polygon, tracked_points, inlier_ratio, median_error, _matrix = match
            score = inlier_ratio * min(tracked_points, 150) / 150.0 - median_error / 100.0
            if score > best_score:
                best = anchor, polygon, tracked_points, inlier_ratio
                best_score = score
        return best

    def _trim_learned_anchors(self) -> None:
        trusted = [item for item in self.anchors if item.trusted]
        learned = [item for item in self.anchors if not item.trusted]
        learned = learned[-self.maximum_learned_anchors :]
        self.anchors = trusted + learned

    def _visible_polygon(self, polygon: np.ndarray) -> np.ndarray:
        visible = clip_polygon_to_frame(polygon, self.frame_size)
        return visible if len(visible) >= 3 else np.asarray(polygon, dtype=np.float64).copy()

    def _new_local_tracker(self, frame: np.ndarray, polygon: np.ndarray) -> LocalMaskTracker:
        return LocalMaskTracker(
            frame,
            polygon,
            feature_polygon=self._visible_polygon(polygon),
            transform_model="homography",
        )


def _match_anchor(
    anchor: MaskAnchor,
    frame: np.ndarray,
) -> tuple[np.ndarray, int, float, float, np.ndarray] | None:
    gray_anchor = cv2.cvtColor(anchor.frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=4000, fastThreshold=8)
    anchor_mask = np.zeros(gray_anchor.shape, dtype=np.uint8)
    visible_anchor = clip_polygon_to_frame(
        anchor.polygon,
        (anchor.frame.shape[1], anchor.frame.shape[0]),
    )
    if len(visible_anchor) < 3:
        return None
    cv2.fillPoly(anchor_mask, [np.round(visible_anchor).astype(np.int32)], 255)
    keypoints_anchor, descriptors_anchor = detector.detectAndCompute(gray_anchor, anchor_mask)
    keypoints_frame, descriptors_frame = detector.detectAndCompute(gray_frame, None)
    if descriptors_anchor is None or descriptors_frame is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors_anchor, descriptors_frame, k=2)
    matches = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    if len(matches) < 30:
        return None
    source = np.float32([keypoints_anchor[item.queryIdx].pt for item in matches])
    target = np.float32([keypoints_frame[item.trainIdx].pt for item in matches])
    source_span = np.ptp(source, axis=0)
    if source_span[0] < anchor.frame.shape[1] * 0.20 or source_span[1] < anchor.frame.shape[0] * 0.12:
        return None
    matrix, inliers = cv2.findHomography(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=4000,
        confidence=0.995,
    )
    if matrix is None or inliers is None:
        return None
    selected = inliers.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(selected))
    inlier_ratio = inlier_count / len(matches)
    if inlier_count < 24 or inlier_ratio < 0.45:
        return None
    predicted = cv2.perspectiveTransform(source[selected][None, :, :], matrix)[0]
    errors = np.linalg.norm(predicted - target[selected], axis=1)
    median_error = float(np.median(errors))
    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    perspective = float(np.hypot(matrix[2, 0], matrix[2, 1]))
    if median_error > 3.0 or not 0.70 <= scale <= 1.35 or perspective > 0.004:
        return None
    polygon = cv2.perspectiveTransform(
        anchor.polygon.astype(np.float32)[None, :, :], matrix
    )[0].astype(np.float64)
    return polygon, len(matches), inlier_ratio, median_error, matrix


def match_ground_anchor_transform(
    anchor_frame: np.ndarray,
    anchor_polygon: np.ndarray,
    frame: np.ndarray,
) -> tuple[np.ndarray, int, float] | None:
    """Match one trusted ground anchor directly to the current camera image."""
    anchor = MaskAnchor("direct", 0, anchor_frame, anchor_polygon, trusted=True)
    matched = _match_anchor(anchor, frame)
    if matched is None:
        return None
    _polygon, matched_points, inlier_ratio, _median_error, matrix = matched
    return matrix, matched_points, inlier_ratio


def camera_view_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_small = cv2.resize(first, (64, 36), interpolation=cv2.INTER_AREA)
    second_small = cv2.resize(second, (64, 36), interpolation=cv2.INTER_AREA)
    first_lab = cv2.cvtColor(first_small, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    second_lab = cv2.cvtColor(second_small, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    first_edges = cv2.Canny(cv2.cvtColor(first_small, cv2.COLOR_BGR2GRAY), 60, 140).astype(np.float32) / 255.0
    second_edges = cv2.Canny(cv2.cvtColor(second_small, cv2.COLOR_BGR2GRAY), 60, 140).astype(np.float32) / 255.0
    color_distance = float(np.mean(np.abs(first_lab - second_lab)))
    edge_distance = float(np.mean(np.abs(first_edges - second_edges)))
    return color_distance + 0.35 * edge_distance
