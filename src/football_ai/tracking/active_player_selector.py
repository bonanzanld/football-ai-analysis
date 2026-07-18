from __future__ import annotations

from dataclasses import dataclass

from football_ai.tracking.track_evaluator import TrackEvaluation
from football_ai.tracking.track_state import TrackState


@dataclass(frozen=True)
class ActivePlayerEvaluation:
    """
    Beoordeling of een track waarschijnlijk bij een actieve speler hoort.

    Deze eerste versie gebruikt alleen technische trackinformatie en
    beweging in beeldcoördinaten. Veldpositie, teamkleur en homografie
    worden later toegevoegd.
    """

    track_id: int

    active_player_score: float
    track_quality_score: float
    movement_score: float
    displacement_score: float
    duration_score: float
    size_score: float

    classification: str

    is_candidate: bool
    is_likely_active_player: bool

    rejection_reasons: tuple[str, ...]


class ActivePlayerSelector:
    """
    Selecteert tracks die waarschijnlijk actieve voetballers voorstellen.

    Dit onderdeel werkt na TrackEvaluator:

        ByteTrack
            ↓
        TrackManager
            ↓
        TrackEvaluator
            ↓
        ActivePlayerSelector

    Deze eerste versie maakt nog geen onderscheid tussen:

    - veldspeler;
    - keeper;
    - scheidsrechter;
    - coach die veel beweegt.

    Daarvoor worden later veldprojectie, teamclassificatie en ruimtelijke
    kenmerken toegevoegd.
    """

    def __init__(
        self,
        minimum_candidate_quality_score: float = 55.0,
        minimum_active_player_score: float = 60.0,
        minimum_frames_seen: int = 31,
        minimum_average_confidence: float = 0.60,
        minimum_average_box_height: float = 24.0,
        maximum_average_box_height: float = 350.0,
        minimum_total_distance_pixels: float = 100.0,
        minimum_displacement_pixels: float = 25.0,
        reference_distance_pixels: float = 1200.0,
        reference_displacement_pixels: float = 500.0,
        reference_duration_frames: int = 180,
        preferred_minimum_box_height: float = 35.0,
        preferred_maximum_box_height: float = 220.0,
    ) -> None:
        if not 0.0 <= minimum_candidate_quality_score <= 100.0:
            raise ValueError(
                "minimum_candidate_quality_score moet tussen "
                "0 en 100 liggen."
            )

        if not 0.0 <= minimum_active_player_score <= 100.0:
            raise ValueError(
                "minimum_active_player_score moet tussen "
                "0 en 100 liggen."
            )

        if minimum_frames_seen < 1:
            raise ValueError(
                "minimum_frames_seen moet minimaal 1 zijn."
            )

        if not 0.0 <= minimum_average_confidence <= 1.0:
            raise ValueError(
                "minimum_average_confidence moet tussen "
                "0 en 1 liggen."
            )

        if minimum_average_box_height <= 0:
            raise ValueError(
                "minimum_average_box_height moet groter zijn dan 0."
            )

        if maximum_average_box_height <= minimum_average_box_height:
            raise ValueError(
                "maximum_average_box_height moet groter zijn dan "
                "minimum_average_box_height."
            )

        if reference_distance_pixels <= 0:
            raise ValueError(
                "reference_distance_pixels moet groter zijn dan 0."
            )

        if reference_displacement_pixels <= 0:
            raise ValueError(
                "reference_displacement_pixels moet groter zijn dan 0."
            )

        if reference_duration_frames <= 0:
            raise ValueError(
                "reference_duration_frames moet groter zijn dan 0."
            )

        if (
            preferred_maximum_box_height
            <= preferred_minimum_box_height
        ):
            raise ValueError(
                "preferred_maximum_box_height moet groter zijn dan "
                "preferred_minimum_box_height."
            )

        self.minimum_candidate_quality_score = (
            minimum_candidate_quality_score
        )
        self.minimum_active_player_score = (
            minimum_active_player_score
        )

        self.minimum_frames_seen = minimum_frames_seen
        self.minimum_average_confidence = (
            minimum_average_confidence
        )

        self.minimum_average_box_height = (
            minimum_average_box_height
        )
        self.maximum_average_box_height = (
            maximum_average_box_height
        )

        self.minimum_total_distance_pixels = (
            minimum_total_distance_pixels
        )
        self.minimum_displacement_pixels = (
            minimum_displacement_pixels
        )

        self.reference_distance_pixels = (
            reference_distance_pixels
        )
        self.reference_displacement_pixels = (
            reference_displacement_pixels
        )
        self.reference_duration_frames = (
            reference_duration_frames
        )

        self.preferred_minimum_box_height = (
            preferred_minimum_box_height
        )
        self.preferred_maximum_box_height = (
            preferred_maximum_box_height
        )

    def evaluate(
        self,
        track: TrackState,
        track_evaluation: TrackEvaluation,
    ) -> ActivePlayerEvaluation:
        """
        Beoordeelt één track als mogelijke actieve speler.
        """

        rejection_reasons = self._rejection_reasons(
            track=track,
            track_evaluation=track_evaluation,
        )

        track_quality_score = (
            track_evaluation.quality_score
        )

        movement_score = self._movement_score(track)
        displacement_score = self._displacement_score(track)
        duration_score = self._duration_score(track)
        size_score = self._size_score(track)

        active_player_score = self._weighted_score(
            values=[
                track_quality_score,
                movement_score,
                displacement_score,
                duration_score,
                size_score,
            ],
            weights=[
                0.35,
                0.25,
                0.15,
                0.15,
                0.10,
            ],
        )

        is_candidate = len(rejection_reasons) == 0

        is_likely_active_player = (
            is_candidate
            and active_player_score
            >= self.minimum_active_player_score
        )

        classification = self._classification(
            is_candidate=is_candidate,
            is_likely_active_player=is_likely_active_player,
            active_player_score=active_player_score,
        )

        return ActivePlayerEvaluation(
            track_id=track.track_id,
            active_player_score=active_player_score,
            track_quality_score=track_quality_score,
            movement_score=movement_score,
            displacement_score=displacement_score,
            duration_score=duration_score,
            size_score=size_score,
            classification=classification,
            is_candidate=is_candidate,
            is_likely_active_player=is_likely_active_player,
            rejection_reasons=tuple(rejection_reasons),
        )

    def evaluate_all(
        self,
        tracks: list[TrackState],
        track_evaluations: dict[int, TrackEvaluation],
    ) -> dict[int, ActivePlayerEvaluation]:
        """
        Beoordeelt alle tracks waarvoor een TrackEvaluation bestaat.
        """

        results: dict[int, ActivePlayerEvaluation] = {}

        for track in tracks:
            track_evaluation = track_evaluations.get(
                track.track_id
            )

            if track_evaluation is None:
                continue

            results[track.track_id] = self.evaluate(
                track=track,
                track_evaluation=track_evaluation,
            )

        return results

    def select_likely_active_players(
        self,
        tracks: list[TrackState],
        track_evaluations: dict[int, TrackEvaluation],
    ) -> list[TrackState]:
        """
        Retourneert alleen tracks die waarschijnlijk actieve spelers zijn.
        """

        evaluations = self.evaluate_all(
            tracks=tracks,
            track_evaluations=track_evaluations,
        )

        return [
            track
            for track in tracks
            if (
                track.track_id in evaluations
                and evaluations[
                    track.track_id
                ].is_likely_active_player
            )
        ]

    def _rejection_reasons(
        self,
        track: TrackState,
        track_evaluation: TrackEvaluation,
    ) -> list[str]:
        reasons: list[str] = []

        if track_evaluation.is_noise:
            reasons.append("noise_track")

        if track_evaluation.is_short:
            reasons.append("short_track")

        if (
            track_evaluation.quality_score
            < self.minimum_candidate_quality_score
        ):
            reasons.append("quality_too_low")

        if track.frames_seen < self.minimum_frames_seen:
            reasons.append("too_few_frames")

        if (
            track.average_confidence
            < self.minimum_average_confidence
        ):
            reasons.append("confidence_too_low")

        if (
            track.average_box_height
            < self.minimum_average_box_height
        ):
            reasons.append("box_too_small")

        if (
            track.average_box_height
            > self.maximum_average_box_height
        ):
            reasons.append("box_too_large")

        movement_too_low = (
            track.total_distance_pixels
            < self.minimum_total_distance_pixels
        )

        displacement_too_low = (
            track.displacement_pixels
            < self.minimum_displacement_pixels
        )

        if movement_too_low and displacement_too_low:
            reasons.append("movement_too_low")

        return reasons

    def _movement_score(
        self,
        track: TrackState,
    ) -> float:
        """
        Score voor de totale beweging van een track.

        Dit is nog beweging in beeldpixels en dus nog geen echte
        loopafstand in meters.
        """

        ratio = (
            track.total_distance_pixels
            / self.reference_distance_pixels
        )

        return self._clamp_score(ratio * 100.0)

    def _displacement_score(
        self,
        track: TrackState,
    ) -> float:
        """
        Score voor afstand tussen begin- en eindpositie.

        Een speler kan veel bewegen en toch eindigen rond de startpositie.
        Daarom krijgt deze score minder gewicht dan totale beweging.
        """

        ratio = (
            track.displacement_pixels
            / self.reference_displacement_pixels
        )

        return self._clamp_score(ratio * 100.0)

    def _duration_score(
        self,
        track: TrackState,
    ) -> float:
        ratio = (
            track.frames_seen
            / self.reference_duration_frames
        )

        return self._clamp_score(ratio * 100.0)

    def _size_score(
        self,
        track: TrackState,
    ) -> float:
        """
        Geeft een hoge score aan boxhoogtes binnen een brede,
        aannemelijke band voor spelers.

        Dit is geen harde perspectiefcorrectie. Een speler ver weg is
        immers kleiner dan een speler dichtbij de camera.
        """

        box_height = track.average_box_height

        if (
            self.preferred_minimum_box_height
            <= box_height
            <= self.preferred_maximum_box_height
        ):
            return 100.0

        if box_height < self.preferred_minimum_box_height:
            ratio = (
                box_height
                / self.preferred_minimum_box_height
            )

            return self._clamp_score(ratio * 100.0)

        oversize_range = (
            self.maximum_average_box_height
            - self.preferred_maximum_box_height
        )

        if oversize_range <= 0:
            return 0.0

        oversize_amount = (
            box_height
            - self.preferred_maximum_box_height
        )

        score = (
            1.0
            - oversize_amount / oversize_range
        ) * 100.0

        return self._clamp_score(score)

    @staticmethod
    def _classification(
        is_candidate: bool,
        is_likely_active_player: bool,
        active_player_score: float,
    ) -> str:
        if is_likely_active_player:
            return "likely_active_player"

        if is_candidate:
            if active_player_score >= 45.0:
                return "possible_active_player"

            return "weak_active_player_candidate"

        return "rejected"

    @staticmethod
    def _weighted_score(
        values: list[float],
        weights: list[float],
    ) -> float:
        if len(values) != len(weights):
            raise ValueError(
                "values en weights moeten even lang zijn."
            )

        total_weight = sum(weights)

        if total_weight <= 0:
            return 0.0

        score = sum(
            value * weight
            for value, weight in zip(values, weights)
        ) / total_weight

        return round(
            ActivePlayerSelector._clamp_score(score),
            1,
        )

    @staticmethod
    def _clamp_score(
        value: float,
    ) -> float:
        return max(
            0.0,
            min(100.0, float(value)),
        )