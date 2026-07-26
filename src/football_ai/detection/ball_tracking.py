from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BallCandidate:
    box: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def size(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (max(0.0, x2 - x1), max(0.0, y2 - y1))


@dataclass(frozen=True)
class BallObservation:
    frame_number: int
    center: tuple[float, float]
    box: tuple[float, float, float, float]
    confidence: float
    source: str


class BallTracker:
    """Select one temporally plausible ball candidate per frame.

    The first version deliberately stays independent from pitch calibration. A
    detector hit is preferred; a short constant-velocity prediction bridges
    brief occlusions without pretending that the ball was truly detected.
    """

    def __init__(
        self,
        maximum_gap_frames: int = 5,
        maximum_jump_pixels: float = 140.0,
        confidence_weight: float = 0.65,
        strong_reacquisition_confidence: float = 0.55,
        acquisition_confidence: float = 0.50,
        supporting_confidence: float = 0.15,
        weak_support_radius_pixels: float = 35.0,
        reacquisition_confirmation_radius_pixels: float = 70.0,
        uncertainty_growth_per_missed_frame: float = 0.25,
        maximum_uncertainty_scale: float = 2.0,
        minimum_prediction_confidence: float = 0.15,
    ) -> None:
        if maximum_gap_frames < 0:
            raise ValueError("maximum_gap_frames must be non-negative")
        self.maximum_gap_frames = int(maximum_gap_frames)
        self.maximum_jump_pixels = float(maximum_jump_pixels)
        self.confidence_weight = float(confidence_weight)
        self.strong_reacquisition_confidence = float(strong_reacquisition_confidence)
        self.acquisition_confidence = float(acquisition_confidence)
        self.supporting_confidence = float(supporting_confidence)
        self.weak_support_radius_pixels = float(weak_support_radius_pixels)
        self.reacquisition_confirmation_radius_pixels = float(
            reacquisition_confirmation_radius_pixels
        )
        self.uncertainty_growth_per_missed_frame = float(
            uncertainty_growth_per_missed_frame
        )
        self.maximum_uncertainty_scale = float(maximum_uncertainty_scale)
        self.minimum_prediction_confidence = float(minimum_prediction_confidence)
        self._observations: list[BallObservation] = []
        self._detected_observations: list[BallObservation] = []
        self._missed_frames = 0
        self._pending_reacquisition: tuple[int, BallCandidate] | None = None

    @property
    def observations(self) -> tuple[BallObservation, ...]:
        return tuple(self._observations)

    def update(
        self,
        frame_number: int,
        candidates: Iterable[BallCandidate],
    ) -> BallObservation | None:
        valid = [candidate for candidate in candidates if self._is_valid(candidate)]
        if not self._observations:
            valid = [
                candidate
                for candidate in valid
                if candidate.confidence >= self.acquisition_confidence
            ]
        predicted_center = (
            None
            if self._missed_frames >= self.maximum_gap_frames
            else self._predict_center(frame_number)
        )
        selected = self._select(valid, predicted_center, frame_number)

        if selected is not None:
            self._pending_reacquisition = None
            observation = BallObservation(
                frame_number=int(frame_number),
                center=selected.center,
                box=selected.box,
                confidence=float(selected.confidence),
                source="detected",
            )
            self._observations.append(observation)
            if selected.confidence >= self.supporting_confidence:
                self._detected_observations.append(observation)
                self._missed_frames = 0
            else:
                self._missed_frames += 1
            return observation

        self._missed_frames += 1
        if self._pending_reacquisition is not None:
            return None
        if (
            predicted_center is None
            or self._missed_frames > self.maximum_gap_frames
            or not self._observations
        ):
            return None

        previous = self._observations[-1]
        last_detected = self._detected_observations[-1]
        predicted_confidence = max(
            0.0,
            last_detected.confidence * (0.72 ** self._missed_frames),
        )
        if predicted_confidence < self.minimum_prediction_confidence:
            return None
        width = previous.box[2] - previous.box[0]
        height = previous.box[3] - previous.box[1]
        x, y = predicted_center
        observation = BallObservation(
            frame_number=int(frame_number),
            center=(x, y),
            box=(x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0),
            confidence=predicted_confidence,
            source="predicted",
        )
        self._observations.append(observation)
        return observation

    def _select(
        self,
        candidates: list[BallCandidate],
        predicted_center: tuple[float, float] | None,
        frame_number: int,
    ) -> BallCandidate | None:
        if not candidates:
            return None
        if predicted_center is None:
            strongest = max(candidates, key=lambda candidate: candidate.confidence)
            if self._observations and strongest.confidence < self.acquisition_confidence:
                return None
            return strongest

        strongest = max(candidates, key=lambda candidate: candidate.confidence)
        previous_confidence = (
            self._observations[-1].confidence if self._observations else 0.0
        )
        if (
            strongest.confidence >= self.strong_reacquisition_confidence
            and previous_confidence < 0.50
            and strongest.confidence >= previous_confidence + 0.15
        ):
            pending = self._pending_reacquisition
            if pending is not None:
                pending_frame, pending_candidate = pending
                confirmation_distance = float(
                    np.linalg.norm(
                        np.subtract(strongest.center, pending_candidate.center)
                    )
                )
                if (
                    int(frame_number) == pending_frame + 1
                    and confirmation_distance
                    <= self.reacquisition_confirmation_radius_pixels
                ):
                    return strongest
            self._pending_reacquisition = (int(frame_number), strongest)
            return None

        self._pending_reacquisition = None

        scored: list[tuple[float, BallCandidate]] = []
        for candidate in candidates:
            distance = float(np.linalg.norm(np.subtract(candidate.center, predicted_center)))
            if (
                candidate.confidence < self.supporting_confidence
                and distance > self.weak_support_radius_pixels
            ):
                continue
            confidence_scale = max(
                0.25,
                min(1.0, candidate.confidence / max(self.acquisition_confidence, 1e-6)),
            )
            uncertainty_scale = min(
                self.maximum_uncertainty_scale,
                1.0
                + self.uncertainty_growth_per_missed_frame * self._missed_frames,
            )
            allowed_jump = (
                self.maximum_jump_pixels * confidence_scale * uncertainty_scale
            )
            if distance > allowed_jump:
                continue
            proximity = 1.0 - distance / allowed_jump
            score = self.confidence_weight * candidate.confidence + (1.0 - self.confidence_weight) * proximity
            scored.append((score, candidate))
        return max(scored, key=lambda item: item[0])[1] if scored else None

    def _predict_center(self, frame_number: int) -> tuple[float, float] | None:
        if not self._detected_observations:
            return None
        last = self._detected_observations[-1]
        if len(self._detected_observations) < 2:
            return last.center
        previous = self._detected_observations[-2]
        frame_delta = max(1, last.frame_number - previous.frame_number)
        velocity = np.subtract(last.center, previous.center) / frame_delta
        speed = float(np.linalg.norm(velocity))
        if speed > self.maximum_jump_pixels:
            velocity *= self.maximum_jump_pixels / speed
        steps = max(1, int(frame_number) - last.frame_number)
        return tuple(np.add(last.center, velocity * steps))

    @staticmethod
    def _is_valid(candidate: BallCandidate) -> bool:
        values = (*candidate.box, candidate.confidence)
        if not all(np.isfinite(value) for value in values):
            return False
        width, height = candidate.size
        if width < 2.0 or height < 2.0 or width > 160.0 or height > 160.0:
            return False
        ratio = width / height
        return 0.35 <= ratio <= 2.8 and 0.0 <= candidate.confidence <= 1.0


def candidates_from_detections(detections: object) -> list[BallCandidate]:
    xyxy = getattr(detections, "xyxy", ())
    confidence = getattr(detections, "confidence", ())
    return [
        BallCandidate(tuple(float(value) for value in box), float(score))
        for box, score in zip(xyxy, confidence, strict=True)
    ]


def exclude_candidates_inside_people(
    candidates: Iterable[BallCandidate],
    person_boxes: Iterable[Iterable[float]],
    lower_body_start: float = 0.45,
    lower_body_end: float = 0.88,
) -> list[BallCandidate]:
    """Remove shin/sock candidates while preserving the ball near the feet."""

    people = [tuple(float(value) for value in box) for box in person_boxes]
    accepted: list[BallCandidate] = []
    for candidate in candidates:
        x, y = candidate.center
        inside_lower_body = any(
            x1 <= x <= x2
            and y1 + (y2 - y1) * lower_body_start <= y <= y2
            and y <= y1 + (y2 - y1) * lower_body_end
            for x1, y1, x2, y2 in people
        )
        if not inside_lower_body:
            accepted.append(candidate)
    return accepted


def save_ball_observations(
    observations: Iterable[BallObservation],
    path: Path,
    source_video: str,
    fps: float,
) -> None:
    payload = {
        "schema_version": 1,
        "source_video": source_video,
        "fps": float(fps),
        "observations": [asdict(item) for item in observations],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
