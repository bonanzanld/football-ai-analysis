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
        confidence_weight: float = 0.25,
        strong_reacquisition_confidence: float = 0.55,
        acquisition_confidence: float = 0.50,
        supporting_confidence: float = 0.15,
        weak_support_radius_pixels: float = 35.0,
        maximum_trajectory_support_frames: int = 15,
        player_activity_radius_pixels: float = 90.0,
        minimum_activity_players: int = 2,
        maximum_player_activity_support_frames: int = 15,
        reacquisition_confirmation_radius_pixels: float = 70.0,
        uncertainty_growth_per_missed_frame: float = 0.25,
        maximum_uncertainty_scale: float = 2.0,
        minimum_prediction_confidence: float = 0.15,
        unrestricted_reacquisition_after_frames: int = 60,
        maximum_speed_pixels_per_frame: float = 45.0,
        player_contact_radius_pixels: float = 55.0,
        contact_speed_multiplier: float = 2.0,
        direction_change_tolerance_degrees: float = 70.0,
        stationary_history_size: int = 4,
        stationary_motion_pixels: float = 12.0,
        stationary_lock_radius_pixels: float = 35.0,
        reacquisition_confirmation_frames: int = 3,
        weak_reacquisition_confidence: float | None = None,
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
        self.maximum_trajectory_support_frames = max(
            0,
            int(maximum_trajectory_support_frames),
        )
        self.player_activity_radius_pixels = max(
            0.0,
            float(player_activity_radius_pixels),
        )
        self.minimum_activity_players = max(1, int(minimum_activity_players))
        self.maximum_player_activity_support_frames = max(
            0,
            int(maximum_player_activity_support_frames),
        )
        self.reacquisition_confirmation_radius_pixels = float(
            reacquisition_confirmation_radius_pixels
        )
        self.uncertainty_growth_per_missed_frame = float(
            uncertainty_growth_per_missed_frame
        )
        self.maximum_uncertainty_scale = float(maximum_uncertainty_scale)
        self.minimum_prediction_confidence = float(minimum_prediction_confidence)
        self.unrestricted_reacquisition_after_frames = int(
            unrestricted_reacquisition_after_frames
        )
        self.maximum_speed_pixels_per_frame = float(
            maximum_speed_pixels_per_frame
        )
        self.player_contact_radius_pixels = float(player_contact_radius_pixels)
        self.contact_speed_multiplier = float(contact_speed_multiplier)
        self.direction_change_tolerance_degrees = float(
            direction_change_tolerance_degrees
        )
        self.stationary_history_size = int(stationary_history_size)
        self.stationary_motion_pixels = float(stationary_motion_pixels)
        self.stationary_lock_radius_pixels = float(stationary_lock_radius_pixels)
        self.reacquisition_confirmation_frames = max(
            2,
            int(reacquisition_confirmation_frames),
        )
        self.weak_reacquisition_confidence = float(
            supporting_confidence
            if weak_reacquisition_confidence is None
            else weak_reacquisition_confidence
        )
        self._observations: list[BallObservation] = []
        self._detected_observations: list[BallObservation] = []
        self._missed_frames = 0
        self._trajectory_support_frames = 0
        self._player_activity_support_frames = 0
        self._pending_reacquisition: tuple[int, BallCandidate, int] | None = None
        self._confirmed_weak_reacquisition = False

    @property
    def observations(self) -> tuple[BallObservation, ...]:
        return tuple(self._observations)

    def update(
        self,
        frame_number: int,
        candidates: Iterable[BallCandidate],
        player_footpoints: Iterable[tuple[float, float]] = (),
    ) -> BallObservation | None:
        valid = [candidate for candidate in candidates if self._is_valid(candidate)]
        if not self._observations:
            valid = [
                candidate
                for candidate in valid
                if candidate.confidence >= self.acquisition_confidence
            ]
        footpoints = tuple(player_footpoints)
        trajectory_center = self._predict_center(frame_number)
        predicted_center = (
            None
            if self._missed_frames >= self.maximum_gap_frames
            else trajectory_center
        )
        selected = self._select(
            valid,
            predicted_center,
            frame_number,
            footpoints,
        )

        if selected is not None:
            self._player_activity_support_frames = 0
            self._pending_reacquisition = None
            confirmed_weak_reacquisition = self._confirmed_weak_reacquisition
            self._confirmed_weak_reacquisition = False
            if (
                selected.confidence < self.acquisition_confidence
                and not confirmed_weak_reacquisition
            ):
                # Een kandidaat onder de zelfstandige detectiedrempel mag een
                # bestaand traject alleen ondersteunen. Hij mag de fysieke
                # balpositie niet verplaatsen en wordt nooit een nieuw anker.
                # Zo kan een witte schoen het bewezen balspoor niet overnemen.
                extends_trajectory_support = (
                    selected.confidence >= self.supporting_confidence
                )
                if extends_trajectory_support:
                    self._trajectory_support_frames += 1
                else:
                    # Een uiterst zwakke kandidaat (< supporting_confidence)
                    # mag alleen binnen de normale korte voorspellingsruimte
                    # helpen. Zo houdt een schoen van 6-10% het spoor niet
                    # onbeperkt levend, ook niet wanneer die dichtbij staat.
                    self._trajectory_support_frames = 0
                    self._missed_frames += 1
                if (
                    predicted_center is None
                    or (
                        extends_trajectory_support
                        and self._trajectory_support_frames
                        > self.maximum_trajectory_support_frames
                    )
                    or (
                        not extends_trajectory_support
                        and self._missed_frames > self.maximum_gap_frames
                    )
                    or not self._detected_observations
                ):
                    if extends_trajectory_support:
                        self._missed_frames += 1
                    return None
                last_detected = self._detected_observations[-1]
                # Een kleine/verre bal levert vaak langere tijd slechts een
                # zwakke detectorscore. Zolang die kandidaat wel exact bij de
                # bestaande baan blijft, telt hij als visuele steun en begint
                # de korte blinde voorspeltijd opnieuw. De kandidaat zelf wordt
                # nog altijd geen fysiek anker en kan de positie dus niet naar
                # een schoen of ander wit object trekken.
                if extends_trajectory_support:
                    self._missed_frames = 0
                    predicted_confidence = min(
                        self.acquisition_confidence - 1e-6,
                        last_detected.confidence
                        * (0.96 ** self._trajectory_support_frames),
                    )
                else:
                    predicted_confidence = (
                        last_detected.confidence
                        * (0.72 ** self._missed_frames)
                    )
                if predicted_confidence < self.minimum_prediction_confidence:
                    return None
                width = last_detected.box[2] - last_detected.box[0]
                height = last_detected.box[3] - last_detected.box[1]
                x, y = predicted_center
                observation = BallObservation(
                    frame_number=int(frame_number),
                    center=(x, y),
                    box=(
                        x - width / 2.0,
                        y - height / 2.0,
                        x + width / 2.0,
                        y + height / 2.0,
                    ),
                    confidence=predicted_confidence,
                    source="predicted",
                )
                self._observations.append(observation)
                return observation
            observation = BallObservation(
                frame_number=int(frame_number),
                center=selected.center,
                box=selected.box,
                confidence=float(selected.confidence),
                source="detected",
            )
            self._observations.append(observation)
            self._detected_observations.append(observation)
            self._missed_frames = 0
            self._trajectory_support_frames = 0
            return observation

        self._trajectory_support_frames = 0
        self._missed_frames += 1
        activity_supported = (
            trajectory_center is not None
            and self._player_activity_support_frames
            < self.maximum_player_activity_support_frames
            and self._has_player_activity_support(trajectory_center, footpoints)
        )
        if activity_supported:
            self._player_activity_support_frames += 1
        if self._pending_reacquisition is not None:
            return None
        if not self._observations:
            return None
        if not activity_supported and (
            predicted_center is None or self._missed_frames > self.maximum_gap_frames
        ):
            return None

        previous = self._observations[-1]
        last_detected = self._detected_observations[-1]
        if activity_supported:
            predicted_confidence = min(
                self.acquisition_confidence - 1e-6,
                last_detected.confidence
                * (0.94 ** self._player_activity_support_frames),
            )
        else:
            predicted_confidence = max(
                0.0,
                last_detected.confidence * (0.72 ** self._missed_frames),
            )
        if predicted_confidence < self.minimum_prediction_confidence:
            return None
        width = previous.box[2] - previous.box[0]
        height = previous.box[3] - previous.box[1]
        prediction_center = trajectory_center if activity_supported else predicted_center
        if prediction_center is None:
            return None
        x, y = prediction_center
        observation = BallObservation(
            frame_number=int(frame_number),
            center=(x, y),
            box=(x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0),
            confidence=predicted_confidence,
            source="predicted",
        )
        self._observations.append(observation)
        return observation

    def _has_player_activity_support(
        self,
        predicted_center: tuple[float, float],
        player_footpoints: tuple[tuple[float, float], ...],
    ) -> bool:
        """Use existing player tracks as bounded support, never as a ball source."""
        nearby_players = 0
        for footpoint in player_footpoints:
            distance = float(
                np.hypot(
                    footpoint[0] - predicted_center[0],
                    footpoint[1] - predicted_center[1],
                )
            )
            if distance <= self.player_contact_radius_pixels:
                return True
            if distance <= self.player_activity_radius_pixels:
                nearby_players += 1
        return nearby_players >= self.minimum_activity_players

    def _select(
        self,
        candidates: list[BallCandidate],
        predicted_center: tuple[float, float] | None,
        frame_number: int,
        player_footpoints: tuple[tuple[float, float], ...],
    ) -> BallCandidate | None:
        if not candidates:
            return None

        # Alleen bevestigde detecties vormen een fysiek uitgangspunt. Zeer
        # zwakke kandidaten mogen de bewezen balbaan niet verplaatsen.
        candidates = self._physically_plausible_candidates(
            candidates,
            frame_number,
            player_footpoints,
        )
        if not candidates:
            return None

        if predicted_center is None:
            if not self._detected_observations:
                return max(candidates, key=lambda candidate: candidate.confidence)

            last = self._detected_observations[-1]
            elapsed = max(1, int(frame_number) - last.frame_number)
            if elapsed <= self.unrestricted_reacquisition_after_frames:
                # Ook nadat de korte live-voorspelling is gestopt, mag een
                # willekeurige sterke false positive niet ineens de bal worden.
                # De zoekzone groeit langzaam, maar blijft aan de laatst bewezen
                # positie gekoppeld.
                radius = self.maximum_jump_pixels * min(2.5, 1.0 + 0.04 * elapsed)
                nearby = [
                    candidate
                    for candidate in candidates
                    if candidate.confidence >= self.strong_reacquisition_confidence
                    and float(
                        np.linalg.norm(np.subtract(candidate.center, last.center))
                    )
                    <= radius
                ]
                if not nearby:
                    return self._confirm_weak_reacquisition(
                        candidates,
                        frame_number,
                        player_footpoints,
                        last.center,
                        radius,
                    )
                expected_center = self._predict_center(frame_number) or last.center
                return self._best_continuation(nearby, expected_center, radius)

            strongest = max(candidates, key=lambda candidate: candidate.confidence)
            if strongest.confidence < self.acquisition_confidence:
                return None
            pending = self._pending_reacquisition
            if pending is not None:
                pending_frame, pending_candidate, confirmation_count = pending
                distance = float(
                    np.linalg.norm(np.subtract(strongest.center, pending_candidate.center))
                )
                if (
                    int(frame_number) == pending_frame + 1
                    and distance <= self.reacquisition_confirmation_radius_pixels
                ):
                    confirmation_count += 1
                    if confirmation_count >= self.reacquisition_confirmation_frames:
                        return strongest
                    self._pending_reacquisition = (
                        int(frame_number),
                        strongest,
                        confirmation_count,
                    )
                    return None
            self._pending_reacquisition = (int(frame_number), strongest, 1)
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
            # Zodra een baltrack bestaat, is continuiteit belangrijker dan de
            # losse detectorscore van dit ene frame. Een witte schoen of lijn
            # kan immers een hogere score krijgen, maar hoort niet ineens de
            # bewezen baltrack over te nemen.
            score = (
                self.confidence_weight * candidate.confidence
                + (1.0 - self.confidence_weight) * proximity
            )
            scored.append((score, candidate))
        return max(scored, key=lambda item: item[0])[1] if scored else None

    def _confirm_weak_reacquisition(
        self,
        candidates: list[BallCandidate],
        frame_number: int,
        player_footpoints: tuple[tuple[float, float], ...],
        last_center: tuple[float, float],
        search_radius: float,
    ) -> BallCandidate | None:
        """Promote only persistent weak detections near active players."""

        eligible = [
            candidate
            for candidate in candidates
            if candidate.confidence >= self.weak_reacquisition_confidence
            and self._has_multi_player_activity(
                candidate.center,
                player_footpoints,
            )
            and float(np.linalg.norm(np.subtract(candidate.center, last_center)))
            <= search_radius
        ]
        if not eligible:
            self._pending_reacquisition = None
            return None

        pending = self._pending_reacquisition
        if pending is None:
            candidate = max(eligible, key=lambda item: item.confidence)
            self._pending_reacquisition = (int(frame_number), candidate, 1)
            return None

        pending_frame, pending_candidate, confirmation_count = pending
        continuations = [
            candidate
            for candidate in eligible
            if float(
                np.linalg.norm(
                    np.subtract(candidate.center, pending_candidate.center)
                )
            )
            <= self.reacquisition_confirmation_radius_pixels
        ]
        if int(frame_number) != pending_frame + 1 or not continuations:
            candidate = max(eligible, key=lambda item: item.confidence)
            self._pending_reacquisition = (int(frame_number), candidate, 1)
            return None

        candidate = self._best_continuation(
            continuations,
            pending_candidate.center,
            self.reacquisition_confirmation_radius_pixels,
        )
        confirmation_count += 1
        if confirmation_count < self.reacquisition_confirmation_frames:
            self._pending_reacquisition = (
                int(frame_number),
                candidate,
                confirmation_count,
            )
            return None

        self._confirmed_weak_reacquisition = True
        return candidate

    def _has_multi_player_activity(
        self,
        center: tuple[float, float],
        player_footpoints: tuple[tuple[float, float], ...],
    ) -> bool:
        """Require a genuine local play cluster for a very weak new anchor."""
        nearby_players = sum(
            float(np.linalg.norm(np.subtract(center, footpoint)))
            <= self.player_activity_radius_pixels
            for footpoint in player_footpoints
        )
        return nearby_players >= self.minimum_activity_players

    def _best_continuation(
        self,
        candidates: list[BallCandidate],
        expected_center: tuple[float, float],
        radius: float,
    ) -> BallCandidate:
        """Prefer continuity over an isolated detector confidence peak."""

        def score(candidate: BallCandidate) -> float:
            distance = float(
                np.linalg.norm(np.subtract(candidate.center, expected_center))
            )
            proximity = max(0.0, 1.0 - distance / max(radius, 1e-6))
            return (
                self.confidence_weight * candidate.confidence
                + (1.0 - self.confidence_weight) * proximity
            )

        return max(candidates, key=score)

    def _physically_plausible_candidates(
        self,
        candidates: list[BallCandidate],
        frame_number: int,
        player_footpoints: tuple[tuple[float, float], ...] = (),
    ) -> list[BallCandidate]:
        """Reject candidates that require an impossible image-space jump."""

        last_accepted = (
            self._detected_observations[-1]
            if self._detected_observations
            else None
        )
        if last_accepted is None:
            return candidates

        elapsed = max(1, int(frame_number) - last_accepted.frame_number)
        if elapsed > self.unrestricted_reacquisition_after_frames:
            return candidates

        stationary_center = self._stationary_center()
        velocity = self._robust_velocity()
        plausible: list[BallCandidate] = []
        for candidate in candidates:
            displacement = float(
                np.linalg.norm(np.subtract(candidate.center, last_accepted.center))
            )
            near_player = self._near_player_contact(
                candidate.center,
                player_footpoints,
            ) or self._near_player_contact(last_accepted.center, player_footpoints)
            speed_limit = self.maximum_speed_pixels_per_frame
            if near_player:
                speed_limit *= self.contact_speed_multiplier
            if displacement > speed_limit * elapsed:
                continue
            if stationary_center is not None:
                stationary_displacement = float(
                    np.linalg.norm(np.subtract(candidate.center, stationary_center))
                )
                stationary_radius = (
                    self.stationary_lock_radius_pixels + 3.0 * self._missed_frames
                )
                if stationary_displacement > stationary_radius:
                    continue
            elif velocity is not None and not near_player:
                movement = np.subtract(candidate.center, last_accepted.center)
                movement_length = float(np.linalg.norm(movement))
                velocity_length = float(np.linalg.norm(velocity))
                if movement_length > 8.0 and velocity_length > 2.0:
                    cosine = float(
                        np.dot(movement, velocity)
                        / max(1e-6, movement_length * velocity_length)
                    )
                    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
                    if angle > self.direction_change_tolerance_degrees:
                        continue
            plausible.append(candidate)
        return plausible

    def _near_player_contact(
        self,
        center: tuple[float, float],
        player_footpoints: tuple[tuple[float, float], ...],
    ) -> bool:
        return any(
            float(np.linalg.norm(np.subtract(center, footpoint)))
            <= self.player_contact_radius_pixels
            for footpoint in player_footpoints
        )

    def _robust_velocity(self) -> np.ndarray | None:
        """Estimate direction from several detections instead of one noisy pair."""

        recent = self._detected_observations[-6:]
        if len(recent) < 3:
            return None
        velocities: list[np.ndarray] = []
        for previous, current in zip(recent, recent[1:]):
            frame_delta = current.frame_number - previous.frame_number
            if frame_delta <= 0:
                continue
            velocities.append(
                np.subtract(current.center, previous.center) / frame_delta
            )
        if len(velocities) < 2:
            return None
        return np.median(np.asarray(velocities, dtype=float), axis=0)

    def _stationary_center(self) -> tuple[float, float] | None:
        """Return the stable recent ball position when it is effectively still."""

        count = self.stationary_history_size
        if count < 2 or len(self._detected_observations) < count:
            return None
        recent = self._detected_observations[-count:]
        centers = np.asarray([observation.center for observation in recent], dtype=float)
        center = np.median(centers, axis=0)
        maximum_deviation = float(
            np.max(np.linalg.norm(centers - center, axis=1))
        )
        if maximum_deviation > self.stationary_motion_pixels:
            return None
        return (float(center[0]), float(center[1]))

    def _predict_center(self, frame_number: int) -> tuple[float, float] | None:
        if not self._detected_observations:
            return None
        last = self._detected_observations[-1]
        if len(self._detected_observations) < 2:
            return last.center
        velocity = self._robust_velocity()
        if velocity is None:
            previous = self._detected_observations[-2]
            frame_delta = max(1, last.frame_number - previous.frame_number)
            velocity = np.subtract(last.center, previous.center) / frame_delta
        speed = float(np.linalg.norm(velocity))
        maximum_prediction_speed = min(
            self.maximum_jump_pixels,
            self.maximum_speed_pixels_per_frame,
        )
        if speed > maximum_prediction_speed:
            velocity *= maximum_prediction_speed / speed
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


def interpolate_detected_gaps(
    observations: Iterable[BallObservation],
    maximum_gap_frames: int = 12,
    maximum_speed_pixels_per_frame: float = 75.0,
) -> tuple[BallObservation, ...]:
    """Fill short, plausible gaps bounded by two actual ball detections.

    This is deliberately a retrospective operation: unlike a live prediction,
    interpolation knows where the ball was detected again. Existing predicted
    observations inside an accepted gap are replaced, while long gaps and
    implausible jumps remain unknown.
    """

    if maximum_gap_frames < 0:
        raise ValueError("maximum_gap_frames must be non-negative")
    if maximum_speed_pixels_per_frame <= 0.0:
        raise ValueError("maximum_speed_pixels_per_frame must be positive")

    ordered = sorted(observations, key=lambda item: item.frame_number)
    by_frame = {item.frame_number: item for item in ordered}
    detected = [item for item in ordered if item.source == "detected"]
    for start, end in zip(detected, detected[1:], strict=False):
        frame_delta = end.frame_number - start.frame_number
        missing_frames = frame_delta - 1
        if missing_frames <= 0 or missing_frames > maximum_gap_frames:
            continue
        displacement = float(np.linalg.norm(np.subtract(end.center, start.center)))
        if displacement / frame_delta > maximum_speed_pixels_per_frame:
            continue

        start_box = np.asarray(start.box, dtype=np.float64)
        end_box = np.asarray(end.box, dtype=np.float64)
        start_center = np.asarray(start.center, dtype=np.float64)
        end_center = np.asarray(end.center, dtype=np.float64)
        confidence = min(start.confidence, end.confidence) * 0.70
        for frame_number in range(start.frame_number + 1, end.frame_number):
            fraction = (frame_number - start.frame_number) / frame_delta
            center = start_center + fraction * (end_center - start_center)
            box = start_box + fraction * (end_box - start_box)
            by_frame[frame_number] = BallObservation(
                frame_number=frame_number,
                center=tuple(float(value) for value in center),
                box=tuple(float(value) for value in box),
                confidence=float(confidence),
                source="interpolated",
            )
    return tuple(by_frame[frame] for frame in sorted(by_frame))


def hold_stationary_detected_gaps(
    observations: Iterable[BallObservation],
    maximum_gap_frames: int,
    maximum_displacement_pixels: float = 35.0,
    minimum_endpoint_confidence: float = 0.50,
) -> tuple[BallObservation, ...]:
    """Keep a stationary ball visible between matching reliable detections.

    The fill is retrospective: both ends of the gap must independently show a
    reliable ball at nearly the same location. This prevents an early false
    positive from being held indefinitely while preserving a genuinely still
    ball that temporarily blends into the background.
    """

    if maximum_gap_frames < 0:
        raise ValueError("maximum_gap_frames must be non-negative")
    if maximum_displacement_pixels <= 0.0:
        raise ValueError("maximum_displacement_pixels must be positive")

    ordered = sorted(observations, key=lambda item: item.frame_number)
    by_frame = {item.frame_number: item for item in ordered}
    detected = [
        item
        for item in ordered
        if item.source == "detected"
        and item.confidence >= minimum_endpoint_confidence
    ]
    for start, end in zip(detected, detected[1:], strict=False):
        frame_delta = end.frame_number - start.frame_number
        missing_frames = frame_delta - 1
        if missing_frames <= 0 or missing_frames > maximum_gap_frames:
            continue
        displacement = float(np.linalg.norm(np.subtract(end.center, start.center)))
        start_size = max(start.box[2] - start.box[0], start.box[3] - start.box[1])
        end_size = max(end.box[2] - end.box[0], end.box[3] - end.box[1])
        allowed = max(
            maximum_displacement_pixels,
            2.5 * max(start_size, end_size),
        )
        if displacement > allowed:
            continue

        center = np.mean(np.asarray([start.center, end.center]), axis=0)
        box = np.mean(np.asarray([start.box, end.box]), axis=0)
        confidence = min(start.confidence, end.confidence) * 0.70
        for frame_number in range(start.frame_number + 1, end.frame_number):
            by_frame[frame_number] = BallObservation(
                frame_number=frame_number,
                center=tuple(float(value) for value in center),
                box=tuple(float(value) for value in box),
                confidence=float(confidence),
                source="stationary_hold",
            )
    return tuple(by_frame[frame] for frame in sorted(by_frame))


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
    maximum_person_overlap: float = 0.55,
) -> list[BallCandidate]:
    """Remove body, sock and shoe candidates while preserving nearby balls.

    A real ball may touch a player's bounding box, so merely testing the
    candidate centre is too aggressive.  Shoes and socks, however, normally
    lie almost completely *inside* the lower part of that box.  The overlap
    test therefore rejects those candidates without removing a ball beside or
    just below a player's feet.
    """

    people = [tuple(float(value) for value in box) for box in person_boxes]
    accepted: list[BallCandidate] = []
    for candidate in candidates:
        x, y = candidate.center
        candidate_x1, candidate_y1, candidate_x2, candidate_y2 = candidate.box
        candidate_area = max(
            1.0,
            (candidate_x2 - candidate_x1) * (candidate_y2 - candidate_y1),
        )
        overlaps_person_feet = False
        inside_lower_body = False
        for x1, y1, x2, y2 in people:
            height = y2 - y1
            lower_start_y = y1 + height * lower_body_start
            inside_lower_body = inside_lower_body or (
                x1 <= x <= x2
                and lower_start_y <= y <= y1 + height * lower_body_end
            )

            overlap_x = max(0.0, min(candidate_x2, x2) - max(candidate_x1, x1))
            overlap_y = max(
                0.0,
                min(candidate_y2, y2) - max(candidate_y1, lower_start_y),
            )
            overlap_fraction = (overlap_x * overlap_y) / candidate_area
            if overlap_fraction >= maximum_person_overlap:
                overlaps_person_feet = True
                break

        if not inside_lower_body and not overlaps_person_feet:
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
