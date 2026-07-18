from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from football_ai.tracking.track_state import TrackState


@dataclass(frozen=True)
class ProjectedTrackEvaluation:
    """
    Beoordeling van de kwaliteit van de veldprojectie van één track.
    """

    track_id: int

    projection_quality_score: float

    coverage_score: float
    inside_pitch_score: float
    continuity_score: float
    jump_score: float
    maturity_score: float

    projection_coverage: float
    inside_pitch_ratio: float
    acceptable_pitch_ratio: float

    track_frames: int
    projected_frames: int

    average_step_meters: float
    maximum_step_meters: float

    valid_step_count: int
    unrealistic_jump_count: int
    extreme_jump_count: int

    outside_position_count: int
    tolerated_outside_count: int
    severe_outside_count: int

    average_outside_distance_meters: float
    maximum_outside_distance_meters: float

    classification: str

    is_projection_available: bool
    is_projection_usable: bool
    is_projection_reliable: bool

    rejection_reasons: tuple[str, ...]


class ProjectedTrackEvaluator:
    """
    Beoordeelt of de veldprojectie van een track betrouwbaar genoeg is.

    Een geprojecteerde positie buiten de veldlijnen wordt niet direct
    als volledig fout beschouwd.

    Een kleine afwijking kan ontstaan door:

    - een onnauwkeurig voetpunt;
    - de onderkant van een bounding box;
    - een speler die deels buiten de lijn staat;
    - een scheidsrechter of andere persoon naast het veld;
    - kleine afwijkingen in de homografie.

    Daarom wordt naast de binnen-veldratio ook de daadwerkelijke afstand
    tot het speelveld berekend.
    """

    def __init__(
        self,
        field_length_meters: float = 64.0,
        field_width_meters: float = 42.0,
        tolerated_outside_distance_meters: float = 1.0,
        severe_outside_distance_meters: float = 5.0,
        minimum_track_frames: int = 30,
        minimum_projected_frames: int = 15,
        minimum_valid_steps: int = 10,
        minimum_projection_coverage: float = 0.70,
        minimum_acceptable_pitch_ratio: float = 0.75,
        minimum_usable_quality_score: float = 60.0,
        minimum_reliable_quality_score: float = 80.0,
        unrealistic_step_meters: float = 3.0,
        extreme_step_meters: float = 8.0,
        reference_average_step_meters: float = 1.0,
        maximum_unrealistic_jump_ratio: float = 0.10,
        maximum_reliable_jump_ratio: float = 0.03,
        maximum_severe_outside_ratio: float = 0.10,
        maximum_reliable_severe_outside_ratio: float = 0.03,
    ) -> None:
        if field_length_meters <= 0.0:
            raise ValueError(
                "field_length_meters moet groter dan 0 zijn."
            )

        if field_width_meters <= 0.0:
            raise ValueError(
                "field_width_meters moet groter dan 0 zijn."
            )

        if tolerated_outside_distance_meters < 0.0:
            raise ValueError(
                "tolerated_outside_distance_meters mag niet negatief zijn."
            )

        if (
            severe_outside_distance_meters
            <= tolerated_outside_distance_meters
        ):
            raise ValueError(
                "severe_outside_distance_meters moet groter zijn dan "
                "tolerated_outside_distance_meters."
            )

        if minimum_track_frames < 1:
            raise ValueError(
                "minimum_track_frames moet minimaal 1 zijn."
            )

        if minimum_projected_frames < 1:
            raise ValueError(
                "minimum_projected_frames moet minimaal 1 zijn."
            )

        if minimum_valid_steps < 1:
            raise ValueError(
                "minimum_valid_steps moet minimaal 1 zijn."
            )

        self._validate_ratio(
            value=minimum_projection_coverage,
            name="minimum_projection_coverage",
        )

        self._validate_ratio(
            value=minimum_acceptable_pitch_ratio,
            name="minimum_acceptable_pitch_ratio",
        )

        self._validate_ratio(
            value=maximum_unrealistic_jump_ratio,
            name="maximum_unrealistic_jump_ratio",
        )

        self._validate_ratio(
            value=maximum_reliable_jump_ratio,
            name="maximum_reliable_jump_ratio",
        )

        self._validate_ratio(
            value=maximum_severe_outside_ratio,
            name="maximum_severe_outside_ratio",
        )

        self._validate_ratio(
            value=maximum_reliable_severe_outside_ratio,
            name="maximum_reliable_severe_outside_ratio",
        )

        if minimum_usable_quality_score < 0.0:
            raise ValueError(
                "minimum_usable_quality_score mag niet negatief zijn."
            )

        if minimum_reliable_quality_score < minimum_usable_quality_score:
            raise ValueError(
                "minimum_reliable_quality_score moet minimaal gelijk zijn "
                "aan minimum_usable_quality_score."
            )

        if unrealistic_step_meters <= 0.0:
            raise ValueError(
                "unrealistic_step_meters moet groter dan 0 zijn."
            )

        if extreme_step_meters <= unrealistic_step_meters:
            raise ValueError(
                "extreme_step_meters moet groter zijn dan "
                "unrealistic_step_meters."
            )

        if reference_average_step_meters <= 0.0:
            raise ValueError(
                "reference_average_step_meters moet groter dan 0 zijn."
            )

        self.field_length_meters = float(
            field_length_meters
        )

        self.field_width_meters = float(
            field_width_meters
        )

        self.tolerated_outside_distance_meters = float(
            tolerated_outside_distance_meters
        )

        self.severe_outside_distance_meters = float(
            severe_outside_distance_meters
        )

        self.minimum_track_frames = int(
            minimum_track_frames
        )

        self.minimum_projected_frames = int(
            minimum_projected_frames
        )

        self.minimum_valid_steps = int(
            minimum_valid_steps
        )

        self.minimum_projection_coverage = float(
            minimum_projection_coverage
        )

        self.minimum_acceptable_pitch_ratio = float(
            minimum_acceptable_pitch_ratio
        )

        self.minimum_usable_quality_score = float(
            minimum_usable_quality_score
        )

        self.minimum_reliable_quality_score = float(
            minimum_reliable_quality_score
        )

        self.unrealistic_step_meters = float(
            unrealistic_step_meters
        )

        self.extreme_step_meters = float(
            extreme_step_meters
        )

        self.reference_average_step_meters = float(
            reference_average_step_meters
        )

        self.maximum_unrealistic_jump_ratio = float(
            maximum_unrealistic_jump_ratio
        )

        self.maximum_reliable_jump_ratio = float(
            maximum_reliable_jump_ratio
        )

        self.maximum_severe_outside_ratio = float(
            maximum_severe_outside_ratio
        )

        self.maximum_reliable_severe_outside_ratio = float(
            maximum_reliable_severe_outside_ratio
        )

    def evaluate(
        self,
        track: TrackState,
    ) -> ProjectedTrackEvaluation:
        """
        Beoordeel de veldprojectie van één track.
        """

        projection_coverage = (
            self._calculate_projection_coverage(
                track=track,
            )
        )

        inside_pitch_ratio = (
            track.inside_pitch_ratio
        )

        outside_distances = (
            self._calculate_outside_distances(
                track=track,
            )
        )

        outside_position_count = len(
            outside_distances
        )

        tolerated_outside_count = sum(
            distance
            <= self.tolerated_outside_distance_meters
            for distance in outside_distances
        )

        severe_outside_count = sum(
            distance
            > self.severe_outside_distance_meters
            for distance in outside_distances
        )

        average_outside_distance_meters = (
            sum(outside_distances)
            / outside_position_count
            if outside_position_count > 0
            else 0.0
        )

        maximum_outside_distance_meters = (
            max(outside_distances)
            if outside_distances
            else 0.0
        )

        acceptable_position_count = (
            track.inside_pitch_frames
            + tolerated_outside_count
        )

        acceptable_pitch_ratio = (
            acceptable_position_count
            / track.projected_frames
            if track.projected_frames > 0
            else 0.0
        )

        severe_outside_ratio = (
            severe_outside_count
            / track.projected_frames
            if track.projected_frames > 0
            else 0.0
        )

        step_distances = (
            self._calculate_valid_step_distances(
                track=track,
            )
        )

        valid_step_count = len(
            step_distances
        )

        average_step_meters = (
            sum(step_distances)
            / valid_step_count
            if valid_step_count > 0
            else 0.0
        )

        maximum_step_meters = (
            max(step_distances)
            if step_distances
            else 0.0
        )

        unrealistic_jump_count = sum(
            distance > self.unrealistic_step_meters
            for distance in step_distances
        )

        extreme_jump_count = sum(
            distance > self.extreme_step_meters
            for distance in step_distances
        )

        unrealistic_jump_ratio = (
            unrealistic_jump_count
            / valid_step_count
            if valid_step_count > 0
            else 0.0
        )

        coverage_score = (
            self._score_coverage(
                projection_coverage=projection_coverage,
            )
        )

        inside_pitch_score = (
            self._score_pitch_location(
                acceptable_pitch_ratio=acceptable_pitch_ratio,
                severe_outside_ratio=severe_outside_ratio,
                average_outside_distance_meters=(
                    average_outside_distance_meters
                ),
            )
        )

        continuity_score = (
            self._score_continuity(
                average_step_meters=average_step_meters,
                valid_step_count=valid_step_count,
            )
        )

        jump_score = (
            self._score_jumps(
                unrealistic_jump_ratio=unrealistic_jump_ratio,
                extreme_jump_count=extreme_jump_count,
            )
        )

        maturity_score = (
            self._score_maturity(
                track_frames=track.frames_seen,
                projected_frames=track.projected_frames,
                valid_step_count=valid_step_count,
            )
        )

        projection_quality_score = (
            self._weighted_score(
                coverage_score=coverage_score,
                inside_pitch_score=inside_pitch_score,
                continuity_score=continuity_score,
                jump_score=jump_score,
                maturity_score=maturity_score,
            )
        )

        is_projection_available = (
            track.projected_frames > 0
        )

        has_minimum_evidence = (
            track.frames_seen
            >= self.minimum_track_frames
            and track.projected_frames
            >= self.minimum_projected_frames
            and valid_step_count
            >= self.minimum_valid_steps
        )

        is_projection_usable = (
            is_projection_available
            and has_minimum_evidence
            and projection_quality_score
            >= self.minimum_usable_quality_score
            and projection_coverage
            >= self.minimum_projection_coverage
            and acceptable_pitch_ratio
            >= self.minimum_acceptable_pitch_ratio
            and severe_outside_ratio
            <= self.maximum_severe_outside_ratio
            and unrealistic_jump_ratio
            <= self.maximum_unrealistic_jump_ratio
            and extreme_jump_count == 0
        )

        is_projection_reliable = (
            is_projection_usable
            and projection_quality_score
            >= self.minimum_reliable_quality_score
            and severe_outside_ratio
            <= self.maximum_reliable_severe_outside_ratio
            and unrealistic_jump_ratio
            <= self.maximum_reliable_jump_ratio
        )

        rejection_reasons = (
            self._determine_rejection_reasons(
                track=track,
                projection_coverage=projection_coverage,
                acceptable_pitch_ratio=acceptable_pitch_ratio,
                severe_outside_ratio=severe_outside_ratio,
                maximum_outside_distance_meters=(
                    maximum_outside_distance_meters
                ),
                valid_step_count=valid_step_count,
                unrealistic_jump_ratio=unrealistic_jump_ratio,
                extreme_jump_count=extreme_jump_count,
                projection_quality_score=projection_quality_score,
            )
        )

        classification = (
            self._classify(
                is_projection_available=is_projection_available,
                is_projection_usable=is_projection_usable,
                is_projection_reliable=is_projection_reliable,
                projection_quality_score=projection_quality_score,
            )
        )

        return ProjectedTrackEvaluation(
            track_id=track.track_id,
            projection_quality_score=projection_quality_score,
            coverage_score=coverage_score,
            inside_pitch_score=inside_pitch_score,
            continuity_score=continuity_score,
            jump_score=jump_score,
            maturity_score=maturity_score,
            projection_coverage=projection_coverage,
            inside_pitch_ratio=inside_pitch_ratio,
            acceptable_pitch_ratio=acceptable_pitch_ratio,
            track_frames=track.frames_seen,
            projected_frames=track.projected_frames,
            average_step_meters=average_step_meters,
            maximum_step_meters=maximum_step_meters,
            valid_step_count=valid_step_count,
            unrealistic_jump_count=unrealistic_jump_count,
            extreme_jump_count=extreme_jump_count,
            outside_position_count=outside_position_count,
            tolerated_outside_count=tolerated_outside_count,
            severe_outside_count=severe_outside_count,
            average_outside_distance_meters=(
                average_outside_distance_meters
            ),
            maximum_outside_distance_meters=(
                maximum_outside_distance_meters
            ),
            classification=classification,
            is_projection_available=is_projection_available,
            is_projection_usable=is_projection_usable,
            is_projection_reliable=is_projection_reliable,
            rejection_reasons=tuple(
                rejection_reasons
            ),
        )

    def evaluate_all(
        self,
        tracks: list[TrackState],
    ) -> dict[int, ProjectedTrackEvaluation]:
        """
        Beoordeel de veldprojectie van alle tracks.
        """

        return {
            track.track_id: self.evaluate(track)
            for track in tracks
        }

    @staticmethod
    def _validate_ratio(
        value: float,
        name: str,
    ) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} moet tussen 0 en 1 liggen."
            )

    @staticmethod
    def _calculate_projection_coverage(
        track: TrackState,
    ) -> float:
        if track.frames_seen <= 0:
            return 0.0

        return min(
            1.0,
            track.projected_frames
            / track.frames_seen,
        )

    def _calculate_outside_distances(
        self,
        track: TrackState,
    ) -> list[float]:
        """
        Bereken de afstand tot het veld voor alle buitenprojecties.

        De inside_pitch_flags en field_positions worden per waarneming
        naast elkaar verwerkt.

        True:
            positie ligt binnen het veld.

        False:
            positie ligt buiten het veld.

        None:
            geen projectie beschikbaar.
        """

        distances: list[float] = []

        for position, inside_flag in zip(
            track.field_positions,
            track.inside_pitch_flags,
        ):
            if position is None:
                continue

            if inside_flag is not False:
                continue

            outside_distance = (
                self._distance_to_pitch(
                    position=position,
                )
            )

            distances.append(
                outside_distance
            )

        return distances

    def _calculate_valid_step_distances(
        self,
        track: TrackState,
    ) -> list[float]:
        """
        Bereken afstanden tussen opeenvolgende bruikbare projecties.

        Alleen opeenvolgende waarnemingen worden gebruikt.

        Posities binnen het veld en posities tot maximaal de ingestelde
        tolerantie buiten het veld worden als bruikbaar beschouwd.
        """

        distances: list[float] = []

        number_of_positions = len(
            track.field_positions
        )

        if number_of_positions < 2:
            return distances

        for index in range(
            1,
            number_of_positions,
        ):
            previous_position = (
                track.field_positions[index - 1]
            )

            current_position = (
                track.field_positions[index]
            )

            if (
                previous_position is None
                or current_position is None
            ):
                continue

            previous_outside_distance = (
                self._distance_to_pitch(
                    position=previous_position,
                )
            )

            current_outside_distance = (
                self._distance_to_pitch(
                    position=current_position,
                )
            )

            if (
                previous_outside_distance
                > self.tolerated_outside_distance_meters
                or current_outside_distance
                > self.tolerated_outside_distance_meters
            ):
                continue

            previous_x, previous_y = (
                previous_position
            )

            current_x, current_y = (
                current_position
            )

            distances.append(
                hypot(
                    current_x - previous_x,
                    current_y - previous_y,
                )
            )

        return distances

    def _distance_to_pitch(
        self,
        position: tuple[float, float],
    ) -> float:
        """
        Bereken de kortste Euclidische afstand tot de veldrechthoek.

        Binnen het veld is de afstand 0 meter.
        """

        field_x, field_y = position

        distance_x = max(
            0.0,
            -field_x,
            field_x - self.field_length_meters,
        )

        distance_y = max(
            0.0,
            -field_y,
            field_y - self.field_width_meters,
        )

        return hypot(
            distance_x,
            distance_y,
        )

    def _score_coverage(
        self,
        projection_coverage: float,
    ) -> float:
        if projection_coverage <= 0.0:
            return 0.0

        return min(
            100.0,
            (
                100.0
                * projection_coverage
                / self.minimum_projection_coverage
            ),
        )

    def _score_pitch_location(
        self,
        acceptable_pitch_ratio: float,
        severe_outside_ratio: float,
        average_outside_distance_meters: float,
    ) -> float:
        """
        Beoordeel de ligging van projecties ten opzichte van het veld.

        Kleine afwijkingen krijgen weinig straf.

        Grote of structurele afwijkingen krijgen een zware straf.
        """

        base_score = min(
            100.0,
            (
                100.0
                * acceptable_pitch_ratio
                / self.minimum_acceptable_pitch_ratio
            ),
        )

        severe_penalty = min(
            70.0,
            severe_outside_ratio * 200.0,
        )

        distance_penalty = min(
            30.0,
            average_outside_distance_meters * 2.0,
        )

        return max(
            0.0,
            base_score
            - severe_penalty
            - distance_penalty,
        )

    def _score_continuity(
        self,
        average_step_meters: float,
        valid_step_count: int,
    ) -> float:
        if valid_step_count == 0:
            return 0.0

        if (
            average_step_meters
            <= self.reference_average_step_meters
        ):
            return 100.0

        excess_ratio = (
            average_step_meters
            / self.reference_average_step_meters
        )

        return max(
            0.0,
            100.0
            - ((excess_ratio - 1.0) * 30.0),
        )

    @staticmethod
    def _score_jumps(
        unrealistic_jump_ratio: float,
        extreme_jump_count: int,
    ) -> float:
        score = 100.0

        score -= min(
            80.0,
            unrealistic_jump_ratio * 400.0,
        )

        score -= min(
            100.0,
            extreme_jump_count * 25.0,
        )

        return max(
            0.0,
            score,
        )

    def _score_maturity(
        self,
        track_frames: int,
        projected_frames: int,
        valid_step_count: int,
    ) -> float:
        track_frame_score = min(
            100.0,
            (
                100.0
                * track_frames
                / self.minimum_track_frames
            ),
        )

        projected_frame_score = min(
            100.0,
            (
                100.0
                * projected_frames
                / self.minimum_projected_frames
            ),
        )

        valid_step_score = min(
            100.0,
            (
                100.0
                * valid_step_count
                / self.minimum_valid_steps
            ),
        )

        return (
            track_frame_score * 0.30
            + projected_frame_score * 0.30
            + valid_step_score * 0.40
        )

    @staticmethod
    def _weighted_score(
        coverage_score: float,
        inside_pitch_score: float,
        continuity_score: float,
        jump_score: float,
        maturity_score: float,
    ) -> float:
        score = (
            coverage_score * 0.20
            + inside_pitch_score * 0.30
            + continuity_score * 0.20
            + jump_score * 0.20
            + maturity_score * 0.10
        )

        return round(
            max(
                0.0,
                min(
                    100.0,
                    score,
                ),
            ),
            1,
        )

    def _determine_rejection_reasons(
        self,
        track: TrackState,
        projection_coverage: float,
        acceptable_pitch_ratio: float,
        severe_outside_ratio: float,
        maximum_outside_distance_meters: float,
        valid_step_count: int,
        unrealistic_jump_ratio: float,
        extreme_jump_count: int,
        projection_quality_score: float,
    ) -> list[str]:
        reasons: list[str] = []

        if track.projected_frames == 0:
            reasons.append(
                "geen veldprojecties"
            )

        if (
            track.frames_seen
            < self.minimum_track_frames
        ):
            reasons.append(
                "track te kort"
            )

        if (
            track.projected_frames
            < self.minimum_projected_frames
        ):
            reasons.append(
                "te weinig geprojecteerde frames"
            )

        if (
            valid_step_count
            < self.minimum_valid_steps
        ):
            reasons.append(
                "te weinig geldige veldstappen"
            )

        if (
            projection_coverage
            < self.minimum_projection_coverage
        ):
            reasons.append(
                "te lage projectiedekking"
            )

        if (
            acceptable_pitch_ratio
            < self.minimum_acceptable_pitch_ratio
        ):
            reasons.append(
                "te veel posities buiten de veldtolerantie"
            )

        if (
            severe_outside_ratio
            > self.maximum_severe_outside_ratio
        ):
            reasons.append(
                "te veel ernstige buitenprojecties"
            )

        if (
            maximum_outside_distance_meters
            > self.severe_outside_distance_meters
        ):
            reasons.append(
                "projectie ligt soms meer dan "
                f"{self.severe_outside_distance_meters:.1f} meter "
                "buiten het veld"
            )

        if (
            unrealistic_jump_ratio
            > self.maximum_unrealistic_jump_ratio
        ):
            reasons.append(
                "te veel onrealistische sprongen"
            )

        if extreme_jump_count > 0:
            reasons.append(
                "extreme projectiesprongen"
            )

        if (
            projection_quality_score
            < self.minimum_usable_quality_score
        ):
            reasons.append(
                "projectiekwaliteit te laag"
            )

        return reasons

    @staticmethod
    def _classify(
        is_projection_available: bool,
        is_projection_usable: bool,
        is_projection_reliable: bool,
        projection_quality_score: float,
    ) -> str:
        if not is_projection_available:
            return "no_projection"

        if is_projection_reliable:
            return "reliable_projection"

        if is_projection_usable:
            return "usable_projection"

        if projection_quality_score >= 40.0:
            return "weak_projection"

        return "bad_projection"