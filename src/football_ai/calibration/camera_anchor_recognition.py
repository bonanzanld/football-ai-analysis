from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class AnchorRecognitionStatus(str, Enum):
    MATCHED = "matched"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class AnchorAppearance:
    anchor_id: str
    frame_size: tuple[int, int]
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray


@dataclass(frozen=True, slots=True)
class AnchorMatchScore:
    anchor_id: str
    good_matches: int
    inliers: int
    inlier_ratio: float
    anchor_coverage: float
    frame_coverage: float
    score: float

    @property
    def reliable(self) -> bool:
        return (
            self.inliers >= 18
            and self.inlier_ratio >= 0.35
            and self.anchor_coverage >= 0.04
            and self.frame_coverage >= 0.04
        )


@dataclass(frozen=True, slots=True)
class AnchorRecognition:
    status: AnchorRecognitionStatus
    anchor_id: str | None
    scores: tuple[AnchorMatchScore, ...]
    reason: str


class CameraAnchorRecognizer:
    def __init__(self, appearances: tuple[AnchorAppearance, ...]) -> None:
        if not appearances:
            raise ValueError("Camerastandherkenning vereist minimaal een ankerbeeld.")
        self.appearances = appearances
        self.orb = cv2.ORB_create(nfeatures=3500, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    @classmethod
    def from_frames(cls, frames: dict[str, np.ndarray]) -> "CameraAnchorRecognizer":
        orb = cv2.ORB_create(nfeatures=3500, fastThreshold=12)
        appearances = []
        for anchor_id, frame in frames.items():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) < 30:
                raise ValueError(f"Ankerbeeld {anchor_id} bevat te weinig visuele kenmerken.")
            appearances.append(
                AnchorAppearance(
                    anchor_id,
                    (frame.shape[1], frame.shape[0]),
                    tuple(keypoints),
                    descriptors,
                )
            )
        return cls(tuple(appearances))

    def recognize(self, frame: np.ndarray) -> AnchorRecognition:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 30:
            return AnchorRecognition(AnchorRecognitionStatus.UNKNOWN, None, (), "Te weinig beeldkenmerken.")
        scores = tuple(
            self._score(appearance, tuple(keypoints), descriptors, (frame.shape[1], frame.shape[0]))
            for appearance in self.appearances
        )
        ranked = sorted(scores, key=lambda item: item.score, reverse=True)
        best = ranked[0]
        if not best.reliable:
            return AnchorRecognition(
                AnchorRecognitionStatus.UNKNOWN,
                None,
                scores,
                f"Beste kandidaat {best.anchor_id} is onvoldoende ondersteund.",
            )
        second = ranked[1] if len(ranked) > 1 else None
        if second is not None and second.reliable and best.score < second.score * 1.20:
            return AnchorRecognition(
                AnchorRecognitionStatus.AMBIGUOUS,
                None,
                scores,
                f"{best.anchor_id} en {second.anchor_id} lijken te veel op elkaar.",
            )
        return AnchorRecognition(
            AnchorRecognitionStatus.MATCHED,
            best.anchor_id,
            scores,
            f"Betrouwbaar gekoppeld aan {best.anchor_id}.",
        )

    def _score(
        self,
        anchor: AnchorAppearance,
        frame_keypoints: tuple[cv2.KeyPoint, ...],
        frame_descriptors: np.ndarray,
        frame_size: tuple[int, int],
    ) -> AnchorMatchScore:
        pairs = self.matcher.knnMatch(anchor.descriptors, frame_descriptors, k=2)
        good = [first for pair in pairs if len(pair) == 2 for first, second in [pair] if first.distance < 0.72 * second.distance]
        if len(good) < 8:
            return AnchorMatchScore(anchor.anchor_id, len(good), 0, 0.0, 0.0, 0.0, 0.0)
        source = np.float32([anchor.keypoints[item.queryIdx].pt for item in good])
        target = np.float32([frame_keypoints[item.trainIdx].pt for item in good])
        _homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
        if mask is None:
            return AnchorMatchScore(anchor.anchor_id, len(good), 0, 0.0, 0.0, 0.0, 0.0)
        selected = mask.ravel().astype(bool)
        inliers = int(np.count_nonzero(selected))
        ratio = inliers / len(good)
        anchor_coverage = _point_coverage(source[selected], anchor.frame_size)
        frame_coverage = _point_coverage(target[selected], frame_size)
        score = inliers * ratio * np.sqrt(anchor_coverage * frame_coverage)
        return AnchorMatchScore(
            anchor.anchor_id,
            len(good),
            inliers,
            float(ratio),
            anchor_coverage,
            frame_coverage,
            float(score),
        )


def _point_coverage(points: np.ndarray, frame_size: tuple[int, int]) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    area = abs(float(cv2.contourArea(hull)))
    return area / max(float(frame_size[0] * frame_size[1]), 1.0)
