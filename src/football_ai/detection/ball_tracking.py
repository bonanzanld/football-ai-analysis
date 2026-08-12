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
class PlayerContext:
    track_id: int | None
    team_id: int | None
    footpoint: tuple[float, float]
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class BallObservation:
    frame_number: int
    center: tuple[float, float]
    box: tuple[float, float, float, float]
    confidence: float
    source: str
    track_segment: int = 0


def offset_ball_candidate(
    candidate: BallCandidate,
    offset_x: int,
    offset_y: int,
) -> BallCandidate:
    """Translate a crop-space candidate back into full-frame coordinates."""

    x1, y1, x2, y2 = candidate.box
    return BallCandidate(
        box=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
        confidence=candidate.confidence,
    )


def best_local_search_anchor(
    candidates: Iterable[BallCandidate],
    previous_center: tuple[float, float],
    maximum_distance: float = 180.0,
    minimum_size: float = 12.0,
) -> BallCandidate | None:
    """Choose a detailed crop hit while rejecting tiny or distant clutter."""

    eligible = [
        candidate
        for candidate in candidates
        if min(candidate.size) >= minimum_size
        and float(np.linalg.norm(np.subtract(candidate.center, previous_center)))
        <= maximum_distance
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda candidate: candidate.confidence)


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
        player_proximity_weight: float = 0.25,
        early_motion_confirmation_frames: int = 3,
        early_motion_window_frames: int = 45,
        early_motion_radius_pixels: float = 35.0,
        early_motion_minimum_displacement_pixels: float = 8.0,
        maximum_player_occlusion_frames: int = 12,
        player_occlusion_horizontal_margin_pixels: float = 18.0,
        active_handoff_confidence: float = 0.75,
        active_handoff_confirmation_frames: int = 3,
        player_contact_memory_frames: int = 15,
        remote_weak_player_contact_lock_frames: int = 15,
        remote_weak_reacquisition_after_frames: int = 30,
        weak_reacquisition_minimum_players: int | None = None,
        weak_reacquisition_minimum_size: float = 0.0,
        remote_weak_footpoint_vertical_tolerance_pixels: float = 30.0,
        remote_weak_reacquisition_confidence: float | None = None,
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
        self.player_proximity_weight = float(
            np.clip(player_proximity_weight, 0.0, 1.0)
        )
        self.early_motion_confirmation_frames = max(
            2,
            int(early_motion_confirmation_frames),
        )
        self.early_motion_window_frames = max(0, int(early_motion_window_frames))
        self.early_motion_radius_pixels = max(0.0, float(early_motion_radius_pixels))
        self.early_motion_minimum_displacement_pixels = max(
            0.0,
            float(early_motion_minimum_displacement_pixels),
        )
        self.maximum_player_occlusion_frames = max(
            0,
            int(maximum_player_occlusion_frames),
        )
        self.player_occlusion_horizontal_margin_pixels = max(
            0.0,
            float(player_occlusion_horizontal_margin_pixels),
        )
        self.active_handoff_confidence = float(active_handoff_confidence)
        self.active_handoff_confirmation_frames = max(
            2,
            int(active_handoff_confirmation_frames),
        )
        self.player_contact_memory_frames = max(
            0,
            int(player_contact_memory_frames),
        )
        self.remote_weak_player_contact_lock_frames = max(
            0,
            int(remote_weak_player_contact_lock_frames),
        )
        self.remote_weak_reacquisition_after_frames = max(
            0,
            int(remote_weak_reacquisition_after_frames),
        )
        self.weak_reacquisition_minimum_players = max(
            1,
            self.minimum_activity_players
            if weak_reacquisition_minimum_players is None
            else int(weak_reacquisition_minimum_players),
        )
        self.weak_reacquisition_minimum_size = max(
            0.0,
            float(weak_reacquisition_minimum_size),
        )
        self.remote_weak_footpoint_vertical_tolerance_pixels = max(
            0.0,
            float(remote_weak_footpoint_vertical_tolerance_pixels),
        )
        self.remote_weak_reacquisition_confidence = float(
            self.weak_reacquisition_confidence
            if remote_weak_reacquisition_confidence is None
            else remote_weak_reacquisition_confidence
        )
        self._observations: list[BallObservation] = []
        self._detected_observations: list[BallObservation] = []
        self._missed_frames = 0
        self._trajectory_support_frames = 0
        self._player_activity_support_frames = 0
        self._player_occlusion_support_frames = 0
        self._pending_reacquisition: tuple[int, BallCandidate, int] | None = None
        self._confirmed_weak_reacquisition = False
        self._confirmed_remote_weak_reacquisition = False
        self._bootstrap_contact_confirmed = False
        self._pending_early_motion: (
            tuple[int, BallCandidate, BallCandidate, int] | None
        ) = None
        self._early_motion_active_until_frame: int | None = None
        self._pending_active_handoff: tuple[int, BallCandidate, int] | None = None
        self._confirmed_active_handoff = False
        self._track_segment = 0
        self._last_player_contact_frame: int | None = None
        self._last_owner_track_id: int | None = None
        self._last_owner_team_id: int | None = None
        self._last_owner_contact_frame: int | None = None

    @property
    def observations(self) -> tuple[BallObservation, ...]:
        return tuple(self._observations)

    @property
    def last_owner_track_id(self) -> int | None:
        return self._last_owner_track_id

    @property
    def last_owner_team_id(self) -> int | None:
        return self._last_owner_team_id

    def update(
        self,
        frame_number: int,
        candidates: Iterable[BallCandidate],
        player_footpoints: Iterable[tuple[float, float]] = (),
        player_boxes: Iterable[tuple[float, float, float, float]] = (),
        player_contexts: Iterable[PlayerContext] = (),
    ) -> BallObservation | None:
        valid = [candidate for candidate in candidates if self._is_valid(candidate)]
        if not self._observations:
            valid = [
                candidate
                for candidate in valid
                if candidate.confidence >= self.acquisition_confidence
            ]
        footpoints = tuple(player_footpoints)
        boxes = tuple(tuple(float(value) for value in box) for box in player_boxes)
        contexts = tuple(player_contexts)
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
            contexts,
        )

        if selected is not None:
            self._player_activity_support_frames = 0
            self._player_occlusion_support_frames = 0
            self._pending_reacquisition = None
            confirmed_weak_reacquisition = self._confirmed_weak_reacquisition
            self._confirmed_weak_reacquisition = False
            confirmed_remote_weak_reacquisition = (
                self._confirmed_remote_weak_reacquisition
            )
            self._confirmed_remote_weak_reacquisition = False
            confirmed_active_handoff = self._confirmed_active_handoff
            self._confirmed_active_handoff = False
            confirmed_early_motion = (
                self._early_motion_active_until_frame is not None
                and int(frame_number) <= self._early_motion_active_until_frame
                and selected.confidence >= self.weak_reacquisition_confidence
            )
            if (
                selected.confidence
                < min(
                    self.acquisition_confidence,
                    self.strong_reacquisition_confidence,
                )
                and not confirmed_weak_reacquisition
                and not confirmed_early_motion
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
                    track_segment=self._track_segment,
                )
                self._observations.append(observation)
                return observation
            if confirmed_active_handoff or confirmed_remote_weak_reacquisition:
                # Dit is een bewust bevestigde wissel naar de actieve
                # spelsituatie, of een verre zwakke herstart na langdurig
                # verlies. Oude snelheids- en stilstandshistorie hoort niet
                # bij het nieuwe fysieke spoor en interpolatie mag beide
                # situaties niet met een kunstmatige balbaan verbinden.
                self._detected_observations.clear()
                self._bootstrap_contact_confirmed = False
                self._pending_early_motion = None
                self._early_motion_active_until_frame = None
                self._track_segment += 1
            observation = BallObservation(
                frame_number=int(frame_number),
                center=selected.center,
                box=selected.box,
                confidence=float(selected.confidence),
                source="detected",
                track_segment=self._track_segment,
            )
            self._observations.append(observation)
            self._detected_observations.append(observation)
            if self._near_player_contact(observation.center, footpoints):
                self._last_player_contact_frame = int(frame_number)
            owner = self._nearest_contact_context(observation.center, contexts)
            if owner is not None:
                self._last_owner_track_id = owner.track_id
                self._last_owner_team_id = owner.team_id
                self._last_owner_contact_frame = int(frame_number)
            if len(self._detected_observations) == 1:
                # Detector boxes do not always extend to the planted foot. A
                # slightly wider activity radius still proves that the first
                # ball anchor belongs to the local play, without changing the
                # stricter contact radius used by normal trajectory physics.
                self._bootstrap_contact_confirmed = any(
                    float(np.linalg.norm(np.subtract(observation.center, footpoint)))
                    <= self.player_activity_radius_pixels
                    for footpoint in footpoints
                )
            else:
                self._pending_early_motion = None
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
        player_overlaps_prediction = (
            trajectory_center is not None
            and self._is_occluded_by_player(trajectory_center, boxes)
        )
        occlusion_supported = (
            player_overlaps_prediction
            and self._player_occlusion_support_frames
            < self.maximum_player_occlusion_frames
        )
        if occlusion_supported:
            self._player_occlusion_support_frames += 1
        elif not player_overlaps_prediction:
            self._player_occlusion_support_frames = 0
        if not self._observations:
            return None
        if not activity_supported and not occlusion_supported and (
            predicted_center is None or self._missed_frames > self.maximum_gap_frames
        ):
            return None

        previous = self._observations[-1]
        last_detected = self._detected_observations[-1]
        if activity_supported or occlusion_supported:
            support_frames = max(
                self._player_activity_support_frames,
                self._player_occlusion_support_frames,
            )
            predicted_confidence = max(
                self.minimum_prediction_confidence,
                min(
                    self.acquisition_confidence - 1e-6,
                    last_detected.confidence * (0.94 ** support_frames),
                ),
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
        prediction_center = (
            trajectory_center
            if activity_supported or occlusion_supported
            else predicted_center
        )
        if prediction_center is None:
            return None
        x, y = prediction_center
        observation = BallObservation(
            frame_number=int(frame_number),
            center=(x, y),
            box=(x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0),
            confidence=predicted_confidence,
            source="predicted",
            track_segment=self._track_segment,
        )
        self._observations.append(observation)
        return observation

    def _is_occluded_by_player(
        self,
        predicted_center: tuple[float, float],
        player_boxes: tuple[tuple[float, float, float, float], ...],
    ) -> bool:
        """Return whether a proven ball path passes behind a player's legs.

        Only the lower portion of a person box counts. This cannot start or
        move a track; it merely extends a pre-existing velocity prediction for
        a tightly bounded number of frames.
        """
        x, y = predicted_center
        margin = self.player_occlusion_horizontal_margin_pixels
        for x1, y1, x2, y2 in player_boxes:
            height = max(0.0, y2 - y1)
            lower_body_y = y1 + height * 0.45
            if (
                x1 - margin <= x <= x2 + margin
                and lower_body_y <= y <= y2 + margin
            ):
                return True
        return False

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
        player_contexts: tuple[PlayerContext, ...],
    ) -> BallCandidate | None:
        if not candidates:
            self._pending_early_motion = None
            self._pending_active_handoff = None
            return None

        active_handoff = self._confirm_active_handoff(
            candidates,
            predicted_center,
            frame_number,
            player_footpoints,
        )
        if active_handoff is not None:
            self._confirmed_active_handoff = True
            return active_handoff

        # Alleen bevestigde detecties vormen een fysiek uitgangspunt. Zeer
        # zwakke kandidaten mogen de bewezen balbaan niet verplaatsen.
        candidates = self._physically_plausible_candidates(
            candidates,
            frame_number,
            player_footpoints,
        )
        if not candidates:
            self._pending_early_motion = None
            return None

        grounded_candidate = self._best_grounded_candidate(
            candidates,
            predicted_center,
            player_footpoints,
        )
        if grounded_candidate is not None:
            return grounded_candidate

        early_motion_active = (
            self._early_motion_active_until_frame is not None
            and int(frame_number) <= self._early_motion_active_until_frame
        )
        if len(self._detected_observations) == 1 or (
            early_motion_active and predicted_center is None
        ):
            early_motion = self._confirm_early_motion(
                candidates,
                frame_number,
            )
            if early_motion is not None:
                return early_motion

        if predicted_center is None:
            if not self._detected_observations:
                return self._best_active_candidate(candidates, player_footpoints)

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

            strongest = self._best_active_candidate(candidates, player_footpoints)
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

    def _best_grounded_candidate(
        self,
        candidates: list[BallCandidate],
        predicted_center: tuple[float, float] | None,
        player_footpoints: tuple[tuple[float, float], ...],
    ) -> BallCandidate | None:
        """Prefer a strong ball at a player's feet over a weak continuation.

        A credible existing continuation still wins, because a ball in flight
        can legitimately be away from every player. The foot preference only
        resolves frames where the old trajectory is supported by weak clutter.
        """

        if predicted_center is None or not player_footpoints:
            return None

        credible_continuation = any(
            candidate.confidence >= self.acquisition_confidence
            and float(np.linalg.norm(np.subtract(candidate.center, predicted_center)))
            <= self.maximum_jump_pixels
            for candidate in candidates
        )
        if credible_continuation:
            return None

        grounded = [
            candidate
            for candidate in candidates
            if candidate.confidence >= self.acquisition_confidence
            and self._near_player_contact(candidate.center, player_footpoints)
        ]
        return max(grounded, key=lambda item: item.confidence) if grounded else None

    def _confirm_active_handoff(
        self,
        candidates: list[BallCandidate],
        predicted_center: tuple[float, float] | None,
        frame_number: int,
        player_footpoints: tuple[tuple[float, float], ...],
    ) -> BallCandidate | None:
        """Confirm a strong distant ball inside the active player cluster.

        Temporal continuity remains authoritative while a credible candidate
        exists near the predicted track. A distant handoff is considered only
        when that continuation is absent, and then requires a strong candidate
        near multiple players for consecutive frames. This lets the tracker
        recover from an incorrect weak track without jumping to one isolated
        white object.
        """

        if predicted_center is None or not self._detected_observations:
            self._pending_active_handoff = None
            return None

        credible_continuation = any(
            candidate.confidence >= self.acquisition_confidence
            and float(np.linalg.norm(np.subtract(candidate.center, predicted_center)))
            <= self.maximum_jump_pixels
            for candidate in candidates
        )
        if credible_continuation:
            self._pending_active_handoff = None
            return None

        eligible = [
            candidate
            for candidate in candidates
            if candidate.confidence >= self.active_handoff_confidence
            and float(np.linalg.norm(np.subtract(candidate.center, predicted_center)))
            > self.maximum_jump_pixels
            and self._has_multi_player_activity(
                candidate.center,
                player_footpoints,
            )
        ]
        if not eligible:
            self._pending_active_handoff = None
            return None

        strongest = max(eligible, key=lambda candidate: candidate.confidence)
        pending = self._pending_active_handoff
        if pending is not None:
            pending_frame, pending_candidate, count = pending
            distance = float(
                np.linalg.norm(
                    np.subtract(strongest.center, pending_candidate.center)
                )
            )
            if (
                int(frame_number) == pending_frame + 1
                and distance <= self.reacquisition_confirmation_radius_pixels
            ):
                count += 1
                if count >= self.active_handoff_confirmation_frames:
                    self._pending_active_handoff = None
                    return strongest
                self._pending_active_handoff = (
                    int(frame_number),
                    strongest,
                    count,
                )
                return None

        self._pending_active_handoff = (int(frame_number), strongest, 1)
        return None

    def _confirm_early_motion(
        self,
        candidates: list[BallCandidate],
        frame_number: int,
    ) -> BallCandidate | None:
        """Turn coherent weak post-contact motion into a second ball anchor.

        A single strong contact frame cannot provide velocity. During a short
        bootstrap window, weak detections may therefore form a pending motion
        chain. The chain never moves the public trajectory until consecutive,
        physically plausible motion has been confirmed.
        """

        if (
            not self._bootstrap_contact_confirmed
            or not self._detected_observations
            or self.early_motion_window_frames == 0
        ):
            self._pending_early_motion = None
            return None

        initial_anchor = self._detected_observations[0]
        if (
            self._early_motion_active_until_frame is not None
            and int(frame_number) > self._early_motion_active_until_frame
        ):
            self._pending_early_motion = None
            return None
        anchor = self._detected_observations[-1]
        elapsed = int(frame_number) - anchor.frame_number
        initial_elapsed = int(frame_number) - initial_anchor.frame_number
        if (
            elapsed <= 0
            or initial_elapsed > self.early_motion_window_frames
        ):
            self._pending_early_motion = None
            return None

        eligible = [
            candidate
            for candidate in candidates
            if candidate.confidence >= self.weak_reacquisition_confidence
            and float(np.linalg.norm(np.subtract(candidate.center, anchor.center)))
            <= self.maximum_speed_pixels_per_frame * elapsed
        ]
        if not eligible:
            self._pending_early_motion = None
            return None

        pending = self._pending_early_motion
        if pending is None:
            candidate = min(
                eligible,
                key=lambda item: float(
                    np.linalg.norm(np.subtract(item.center, anchor.center))
                ),
            )
            self._pending_early_motion = (
                int(frame_number),
                candidate,
                candidate,
                1,
            )
            return None

        pending_frame, first_candidate, previous_candidate, count = pending
        if int(frame_number) != pending_frame + 1:
            self._pending_early_motion = None
            return None

        first_delta = max(1, pending_frame - anchor.frame_number)
        velocity = np.subtract(first_candidate.center, anchor.center) / first_delta
        expected_center = np.add(previous_candidate.center, velocity)
        continuations = [
            candidate
            for candidate in eligible
            if float(np.linalg.norm(np.subtract(candidate.center, expected_center)))
            <= self.early_motion_radius_pixels
        ]
        if not continuations:
            self._pending_early_motion = None
            return None

        candidate = min(
            continuations,
            key=lambda item: float(
                np.linalg.norm(np.subtract(item.center, expected_center))
            ),
        )
        count += 1
        displacement = float(
            np.linalg.norm(np.subtract(candidate.center, anchor.center))
        )
        if count < self.early_motion_confirmation_frames:
            self._pending_early_motion = (
                int(frame_number),
                first_candidate,
                candidate,
                count,
            )
            return None
        if displacement < self.early_motion_minimum_displacement_pixels:
            self._pending_early_motion = None
            return None

        self._pending_early_motion = None
        if self._early_motion_active_until_frame is None:
            self._early_motion_active_until_frame = (
                initial_anchor.frame_number + self.early_motion_window_frames
            )
        self._confirmed_weak_reacquisition = True
        return candidate

    def _best_active_candidate(
        self,
        candidates: list[BallCandidate],
        player_footpoints: tuple[tuple[float, float], ...],
    ) -> BallCandidate:
        """Prefer a similarly credible candidate close to the current play.

        Player proximity is deliberately only a bounded tie-breaker. A ball in
        flight can be far from every player, so a clearly stronger isolated
        detection must still win. Temporal continuity remains authoritative
        after the first physical anchor has been selected.
        """

        weight = self.player_proximity_weight
        if weight <= 0.0 or not player_footpoints:
            return max(candidates, key=lambda candidate: candidate.confidence)

        def score(candidate: BallCandidate) -> float:
            nearest_player = min(
                float(np.linalg.norm(np.subtract(candidate.center, footpoint)))
                for footpoint in player_footpoints
            )
            proximity = max(
                0.0,
                1.0 - nearest_player / max(self.player_activity_radius_pixels, 1e-6),
            )
            return (1.0 - weight) * candidate.confidence + weight * proximity

        return max(candidates, key=score)

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
            and min(candidate.size) >= self.weak_reacquisition_minimum_size
            and (
                self._has_weak_reacquisition_activity(
                    candidate.center,
                    player_footpoints,
                )
                or (
                    candidate.confidence >= self.acquisition_confidence
                    and self._near_player_contact(
                        candidate.center,
                        player_footpoints,
                    )
                )
            )
            and (
                float(np.linalg.norm(np.subtract(candidate.center, last_center)))
                <= search_radius
                or (
                    self._missed_frames
                    >= self.remote_weak_reacquisition_after_frames
                    and candidate.confidence
                    >= self.remote_weak_reacquisition_confidence
                    and self._has_remote_weak_foot_support(
                        candidate.center,
                        player_footpoints,
                    )
                )
            )
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
        self._confirmed_remote_weak_reacquisition = (
            float(np.linalg.norm(np.subtract(candidate.center, last_center)))
            > search_radius
        )
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

    def _has_weak_reacquisition_activity(
        self,
        center: tuple[float, float],
        player_footpoints: tuple[tuple[float, float], ...],
    ) -> bool:
        nearby_players = sum(
            float(np.linalg.norm(np.subtract(center, footpoint)))
            <= self.player_activity_radius_pixels
            for footpoint in player_footpoints
        )
        return nearby_players >= self.weak_reacquisition_minimum_players

    def _has_remote_weak_foot_support(
        self,
        center: tuple[float, float],
        player_footpoints: tuple[tuple[float, float], ...],
    ) -> bool:
        """Reject weak remote restarts above the players' foot line.

        A low-confidence ball can restart a lost track only at ground-level
        player contact. This intentionally excludes persistent head and torso
        detections when the person box is missing or imprecise. Strong ball
        detections and local trajectory continuations remain unaffected.
        """

        x, y = center
        tolerance = self.remote_weak_footpoint_vertical_tolerance_pixels
        return any(
            abs(y - foot_y) <= tolerance
            and abs(x - foot_x) <= self.player_activity_radius_pixels
            for foot_x, foot_y in player_footpoints
        )

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
        recent_player_contact = (
            self._last_player_contact_frame is not None
            and int(frame_number) - self._last_player_contact_frame
            <= self.player_contact_memory_frames
        )
        plausible: list[BallCandidate] = []
        for candidate in candidates:
            candidate_at_player = self._near_player_contact(
                candidate.center,
                player_footpoints,
            )
            remote_weak_candidate = (
                self._missed_frames >= self.remote_weak_reacquisition_after_frames
                and candidate.confidence
                >= self.remote_weak_reacquisition_confidence
                and min(candidate.size) >= self.weak_reacquisition_minimum_size
                and not self._current_track_has_player_support(
                    frame_number,
                    player_footpoints,
                )
                and self._has_weak_reacquisition_activity(
                    candidate.center,
                    player_footpoints,
                )
                and self._has_remote_weak_foot_support(
                    candidate.center,
                    player_footpoints,
                )
            )
            strong_grounded_restart = (
                self._missed_frames >= self.remote_weak_reacquisition_after_frames
                and candidate.confidence >= self.acquisition_confidence
                and min(candidate.size) >= self.weak_reacquisition_minimum_size
                and not self._current_track_has_player_support(
                    frame_number,
                    player_footpoints,
                )
                and candidate_at_player
                and self._has_remote_weak_foot_support(
                    candidate.center,
                    player_footpoints,
                )
            )
            if remote_weak_candidate or strong_grounded_restart:
                plausible.append(candidate)
                continue
            displacement = float(
                np.linalg.norm(np.subtract(candidate.center, last_accepted.center))
            )
            near_player = candidate_at_player or self._near_player_contact(
                last_accepted.center,
                player_footpoints,
            ) or recent_player_contact
            speed_limit = self.maximum_speed_pixels_per_frame
            if near_player:
                speed_limit *= self.contact_speed_multiplier
            if displacement > speed_limit * elapsed:
                continue
            if stationary_center is not None and not (
                candidate_at_player
                and candidate.confidence >= self.acquisition_confidence
            ):
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

    def _nearest_contact_context(
        self,
        center: tuple[float, float],
        player_contexts: tuple[PlayerContext, ...],
    ) -> PlayerContext | None:
        eligible = [
            player
            for player in player_contexts
            if player.track_id is not None
            and float(np.linalg.norm(np.subtract(center, player.footpoint)))
            <= self.player_contact_radius_pixels
        ]
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda player: float(
                np.linalg.norm(np.subtract(center, player.footpoint))
            ),
        )

    def _current_track_has_player_support(
        self,
        frame_number: int,
        player_footpoints: tuple[tuple[float, float], ...],
    ) -> bool:
        """Keep a proven player-supported ball ahead of remote weak clutter."""

        predicted_center = self._predict_center(frame_number)
        recently_at_player = (
            self._last_player_contact_frame is not None
            and int(frame_number) - self._last_player_contact_frame
            <= self.remote_weak_player_contact_lock_frames
        )
        return recently_at_player or (
            predicted_center is not None
            and self._has_player_activity_support(
                predicted_center,
                player_footpoints,
            )
        )

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
        if start.track_segment != end.track_segment:
            continue
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
                track_segment=start.track_segment,
            )
    return tuple(by_frame[frame] for frame in sorted(by_frame))


def hold_stationary_detected_gaps(
    observations: Iterable[BallObservation],
    maximum_gap_frames: int,
    maximum_displacement_pixels: float = 35.0,
    minimum_endpoint_confidence: float = 0.50,
    stationary_evidence_detections: int = 3,
    stationary_evidence_window_frames: int = 5,
    stationary_evidence_radius_pixels: float = 12.0,
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
    all_detected = [item for item in ordered if item.source == "detected"]
    detected = [
        item
        for item in ordered
        if item.source == "detected"
        and item.confidence >= minimum_endpoint_confidence
    ]
    evidence_count = max(1, int(stationary_evidence_detections))
    evidence_window = max(0, int(stationary_evidence_window_frames))
    evidence_radius = max(0.0, float(stationary_evidence_radius_pixels))
    for endpoint_index, (start, end) in enumerate(
        zip(detected, detected[1:], strict=False)
    ):
        if start.track_segment != end.track_segment:
            continue
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

        before = detected[max(0, endpoint_index - evidence_count + 1) : endpoint_index + 1]
        after_start = endpoint_index + 1
        after = detected[after_start : after_start + evidence_count]
        if len(before) < evidence_count or len(after) < evidence_count:
            continue
        if start.frame_number - before[0].frame_number > evidence_window:
            continue
        if after[-1].frame_number - end.frame_number > evidence_window:
            continue
        start_evidence_center = np.mean(
            np.asarray([item.center for item in before]),
            axis=0,
        )
        end_evidence_center = np.mean(
            np.asarray([item.center for item in after]),
            axis=0,
        )
        if any(
            float(np.linalg.norm(np.subtract(item.center, start_evidence_center)))
            > evidence_radius
            for item in before
        ) or any(
            float(np.linalg.norm(np.subtract(item.center, end_evidence_center)))
            > evidence_radius
            for item in after
        ):
            continue

        center = np.mean(np.asarray([start.center, end.center]), axis=0)
        intervening_detections = [
            item
            for item in all_detected
            if start.frame_number < item.frame_number < end.frame_number
        ]
        if any(
            float(np.linalg.norm(np.subtract(item.center, center))) > allowed
            for item in intervening_detections
        ):
            # The ball left this location and later returned. Treating that as
            # one long stationary interval would erase the observed flight.
            continue
        box = np.mean(np.asarray([start.box, end.box]), axis=0)
        confidence = min(start.confidence, end.confidence) * 0.70
        for frame_number in range(start.frame_number + 1, end.frame_number):
            existing = by_frame.get(frame_number)
            if existing is not None and existing.source == "detected":
                continue
            by_frame[frame_number] = BallObservation(
                frame_number=frame_number,
                center=tuple(float(value) for value in center),
                box=tuple(float(value) for value in box),
                confidence=float(confidence),
                source="stationary_hold",
                track_segment=start.track_segment,
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
    foot_priority_confidence: float = 0.50,
    foot_priority_radius_pixels: float = 25.0,
    foot_priority_minimum_size: float = 14.0,
    weak_leg_minimum_confidence: float = 0.05,
    weak_leg_minimum_size: float = 8.0,
    weak_leg_maximum_size: float = 24.0,
    crowded_play_radius_pixels: float = 90.0,
    crowded_play_minimum_players: int = 2,
) -> list[BallCandidate]:
    """Preserve detector evidence even when it overlaps a person.

    This function used to remove weak head/torso hits and candidates largely
    inside a player's lower body.  That is an unsafe place to make a hard
    decision: real balls overlap person boxes during control, tackles,
    headers, throw-ins, and goalkeeper possession.  Temporal selection and
    active-ball classification may still reject these candidates later, but
    the detector evidence must remain available to them.

    The parameters remain for call-site compatibility while cached detector
    artifacts transition away from the former person-filtering policy.
    """

    del (
        person_boxes,
        lower_body_start,
        lower_body_end,
        maximum_person_overlap,
        foot_priority_confidence,
        foot_priority_radius_pixels,
        foot_priority_minimum_size,
        weak_leg_minimum_confidence,
        weak_leg_minimum_size,
        weak_leg_maximum_size,
        crowded_play_radius_pixels,
        crowded_play_minimum_players,
    )
    return list(candidates)


def save_ball_observations(
    observations: Iterable[BallObservation],
    path: Path,
    source_video: str,
    fps: float,
) -> None:
    payload = {
        "schema_version": 2,
        "source_video": source_video,
        "fps": float(fps),
        "observations": [asdict(item) for item in observations],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
