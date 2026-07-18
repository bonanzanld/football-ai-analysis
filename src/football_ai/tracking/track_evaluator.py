from __future__ import annotations

from dataclasses import dataclass

from .track_state import TrackState


@dataclass(frozen=True)
class TrackEvaluation:
    """
    Beoordeling van één track.

    De evaluator verwijdert geen tracks. Hij kent alleen eigenschappen
    en scores toe die later gebruikt kunnen worden door bijvoorbeeld:

    - ActivePlayerSelector
    - TeamClassifier
    - TrackFilter
    - statistieken
    - tactische analyse
    """

    track_id: int

    quality_score: float
    stability_score: float
    confidence_score: float
    duration_score: float
    visibility_score: float

    classification: str

    is_noise: bool
    is_short: bool
    is_usable: bool
    is_stable: bool


class TrackEvaluator:
    """
    Beoordeelt de technische kwaliteit van tracks.

    Deze beoordeling zegt nog niet of een track daadwerkelijk een speler,
    coach, scheidsrechter of toeschouwer is. Daarvoor zijn later ook
    veldpositie, beweging en teaminformatie nodig.
    """

    def __init__(
        self,
        minimum_usable_frames: int = 31,
        minimum_stable_frames: int = 90,
        minimum_usable_confidence: float = 0.60,
        minimum_stable_confidence: float = 0.70,
        minimum_stable_visibility_ratio: float = 0.80,
        noise_maximum_frames: int = 5,
        short_maximum_frames: int = 30,
    ) -> None:
        if minimum_usable_frames < 1:
            raise ValueError(
                "minimum_usable_frames moet minimaal 1 zijn."
            )

        if minimum_stable_frames < minimum_usable_frames:
            raise ValueError(
                "minimum_stable_frames moet groter zijn dan of gelijk "
                "zijn aan minimum_usable_frames."
            )

        if noise_maximum_frames < 0:
            raise ValueError(
                "noise_maximum_frames mag niet negatief zijn."
            )

        if short_maximum_frames < noise_maximum_frames:
            raise ValueError(
                "short_maximum_frames moet groter zijn dan of gelijk "
                "zijn aan noise_maximum_frames."
            )

        self.minimum_usable_frames = minimum_usable_frames
        self.minimum_stable_frames = minimum_stable_frames

        self.minimum_usable_confidence = (
            minimum_usable_confidence
        )
        self.minimum_stable_confidence = (
            minimum_stable_confidence
        )

        self.minimum_stable_visibility_ratio = (
            minimum_stable_visibility_ratio
        )

        self.noise_maximum_frames = noise_maximum_frames
        self.short_maximum_frames = short_maximum_frames

    def evaluate(
        self,
        track: TrackState,
    ) -> TrackEvaluation:
        """
        Beoordeelt één TrackState en retourneert een TrackEvaluation.
        """

        visibility_ratio = self._visibility_ratio(track)

        duration_score = self._duration_score(track)
        confidence_score = self._confidence_score(track)
        visibility_score = self._visibility_score(track)

        stability_score = self._weighted_score(
            values=[
                duration_score,
                confidence_score,
                visibility_score,
            ],
            weights=[
                0.35,
                0.25,
                0.40,
            ],
        )

        quality_score = self._weighted_score(
            values=[
                duration_score,
                confidence_score,
                visibility_score,
                stability_score,
            ],
            weights=[
                0.25,
                0.25,
                0.25,
                0.25,
            ],
        )

        is_noise = (
            track.frames_seen <= self.noise_maximum_frames
        )

        is_short = (
            self.noise_maximum_frames
            < track.frames_seen
            <= self.short_maximum_frames
        )

        is_usable = (
            track.frames_seen >= self.minimum_usable_frames
            and track.average_confidence
            >= self.minimum_usable_confidence
        )

        is_stable = (
            track.frames_seen >= self.minimum_stable_frames
            and track.average_confidence
            >= self.minimum_stable_confidence
            and visibility_ratio
            >= self.minimum_stable_visibility_ratio
        )

        classification = self._classification(
            is_noise=is_noise,
            is_short=is_short,
            is_usable=is_usable,
            is_stable=is_stable,
        )

        return TrackEvaluation(
            track_id=track.track_id,
            quality_score=quality_score,
            stability_score=stability_score,
            confidence_score=confidence_score,
            duration_score=duration_score,
            visibility_score=visibility_score,
            classification=classification,
            is_noise=is_noise,
            is_short=is_short,
            is_usable=is_usable,
            is_stable=is_stable,
        )

    def evaluate_all(
        self,
        tracks: list[TrackState],
    ) -> dict[int, TrackEvaluation]:
        """
        Beoordeelt meerdere tracks.

        De sleutel van het resultaat is het ByteTrack-ID.
        """

        return {
            track.track_id: self.evaluate(track)
            for track in tracks
        }

    @staticmethod
    def _visibility_ratio(
        track: TrackState,
    ) -> float:
        if track.lifespan_frames <= 0:
            return 0.0

        return min(
            1.0,
            track.frames_seen / track.lifespan_frames,
        )

    def _duration_score(
        self,
        track: TrackState,
    ) -> float:
        """
        Zet trackduur om naar een score van 0 tot 100.

        Een track met minimum_stable_frames of meer krijgt 100 punten.
        """

        if self.minimum_stable_frames <= 0:
            return 100.0

        ratio = (
            track.frames_seen
            / self.minimum_stable_frames
        )

        return self._clamp_score(ratio * 100.0)

    @staticmethod
    def _confidence_score(
        track: TrackState,
    ) -> float:
        """
        Zet gemiddelde detectieconfidence om naar 0 tot 100.
        """

        return TrackEvaluator._clamp_score(
            track.average_confidence * 100.0
        )

    @staticmethod
    def _visibility_score(
        track: TrackState,
    ) -> float:
        """
        Zet zichtbaarheidsratio om naar 0 tot 100.
        """

        if track.lifespan_frames <= 0:
            return 0.0

        ratio = (
            track.frames_seen
            / track.lifespan_frames
        )

        return TrackEvaluator._clamp_score(
            ratio * 100.0
        )

    @staticmethod
    def _classification(
        is_noise: bool,
        is_short: bool,
        is_usable: bool,
        is_stable: bool,
    ) -> str:
        if is_stable:
            return "stable"

        if is_usable:
            return "usable"

        if is_short:
            return "short"

        if is_noise:
            return "noise"

        return "weak"

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
            TrackEvaluator._clamp_score(score),
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