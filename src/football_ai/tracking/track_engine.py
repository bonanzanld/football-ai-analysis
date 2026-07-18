from __future__ import annotations

from typing import Any

from football_ai.pitch.field_projector import FieldProjector
from football_ai.tracking.active_player_selector import (
    ActivePlayerEvaluation,
    ActivePlayerSelector,
)
from football_ai.tracking.projected_track_evaluator import (
    ProjectedTrackEvaluation,
    ProjectedTrackEvaluator,
)
from football_ai.tracking.track_evaluator import (
    TrackEvaluation,
    TrackEvaluator,
)
from football_ai.tracking.track_manager import TrackManager
from football_ai.tracking.track_state import TrackState


class TrackEngine:
    """
    Centrale orchestrator voor de trackingpipeline.

    Pipeline:

        ByteTrack
            ↓
        TrackManager
            ↓
        ┌────────────────────────────┐
        │                            │
        ▼                            ▼
        TrackEvaluator     ProjectedTrackEvaluator
        │                            │
        └──────────────┬─────────────┘
                       ▼
             ActivePlayerSelector

    Trackkwaliteit en projectiekwaliteit worden onafhankelijk
    van elkaar beoordeeld.

    Wanneer een FieldProjector beschikbaar is, worden veldposities,
    afstanden in meters en projectieafwijkingen bijgehouden.
    """

    def __init__(
        self,
        field_projector: FieldProjector | None = None,
    ) -> None:
        self.track_manager = TrackManager(
            field_projector=field_projector,
        )

        self.track_evaluator = TrackEvaluator(
            minimum_usable_frames=31,
            minimum_stable_frames=90,
            minimum_usable_confidence=0.60,
            minimum_stable_confidence=0.70,
            minimum_stable_visibility_ratio=0.80,
            noise_maximum_frames=5,
            short_maximum_frames=30,
        )

        (
            field_length_meters,
            field_width_meters,
        ) = self._get_field_dimensions(
            field_projector=field_projector,
        )

        self.projected_track_evaluator = (
            ProjectedTrackEvaluator(
                field_length_meters=field_length_meters,
                field_width_meters=field_width_meters,
                tolerated_outside_distance_meters=1.0,
                severe_outside_distance_meters=5.0,
                minimum_track_frames=30,
                minimum_projected_frames=15,
                minimum_valid_steps=10,
                minimum_projection_coverage=0.70,
                minimum_acceptable_pitch_ratio=0.75,
                minimum_usable_quality_score=60.0,
                minimum_reliable_quality_score=80.0,
                unrealistic_step_meters=3.0,
                extreme_step_meters=8.0,
                reference_average_step_meters=1.0,
                maximum_unrealistic_jump_ratio=0.10,
                maximum_reliable_jump_ratio=0.03,
                maximum_severe_outside_ratio=0.10,
                maximum_reliable_severe_outside_ratio=0.03,
            )
        )

        self.active_player_selector = ActivePlayerSelector(
            minimum_candidate_quality_score=55.0,
            minimum_active_player_score=60.0,
            minimum_frames_seen=31,
            minimum_average_confidence=0.60,
            minimum_average_box_height=24.0,
            maximum_average_box_height=350.0,
            minimum_total_distance_pixels=100.0,
            minimum_displacement_pixels=25.0,
            reference_distance_pixels=1200.0,
            reference_displacement_pixels=500.0,
            reference_duration_frames=180,
            preferred_minimum_box_height=35.0,
            preferred_maximum_box_height=220.0,
        )

        self.track_evaluations: dict[
            int,
            TrackEvaluation,
        ] = {}

        self.projected_track_evaluations: dict[
            int,
            ProjectedTrackEvaluation,
        ] = {}

        self.active_player_evaluations: dict[
            int,
            ActivePlayerEvaluation,
        ] = {}

        self._is_finalized = False

    def update(
        self,
        tracked_players: Any,
        frame_number: int,
    ) -> None:
        """
        Verwerk de tracks van één videoframe.
        """

        self.track_manager.update(
            tracked_players=tracked_players,
            frame_number=frame_number,
        )

        self._is_finalized = False

    def finalize(self) -> None:
        """
        Bereken alle track-, projectie- en actieve-spelerevaluaties.
        """

        if self._is_finalized:
            return

        tracks = self.tracks

        self.track_evaluations = (
            self.track_evaluator.evaluate_all(
                tracks=tracks,
            )
        )

        self.projected_track_evaluations = (
            self.projected_track_evaluator.evaluate_all(
                tracks=tracks,
            )
        )

        self.active_player_evaluations = (
            self.active_player_selector.evaluate_all(
                tracks=tracks,
                track_evaluations=self.track_evaluations,
            )
        )

        self._is_finalized = True

    @property
    def tracks(self) -> list[TrackState]:
        return self.track_manager.get_all_tracks()

    @property
    def number_of_tracks(self) -> int:
        return self.track_manager.number_of_tracks

    @property
    def has_field_projection(self) -> bool:
        """
        Geeft aan of de TrackEngine een FieldProjector gebruikt.
        """

        return self.track_manager.field_projector is not None

    @property
    def likely_active_players(self) -> list[TrackState]:
        self.finalize()

        return [
            track
            for track in self.tracks
            if (
                track.track_id
                in self.active_player_evaluations
                and self.active_player_evaluations[
                    track.track_id
                ].is_likely_active_player
            )
        ]

    @property
    def candidate_active_players(self) -> list[TrackState]:
        self.finalize()

        return [
            track
            for track in self.tracks
            if (
                track.track_id
                in self.active_player_evaluations
                and self.active_player_evaluations[
                    track.track_id
                ].is_candidate
            )
        ]

    @property
    def usable_projected_tracks(self) -> list[TrackState]:
        """
        Tracks waarvan de veldprojectie bruikbaar is.
        """

        self.finalize()

        return [
            track
            for track in self.tracks
            if (
                track.track_id
                in self.projected_track_evaluations
                and self.projected_track_evaluations[
                    track.track_id
                ].is_projection_usable
            )
        ]

    @property
    def reliable_projected_tracks(self) -> list[TrackState]:
        """
        Tracks waarvan de veldprojectie betrouwbaar is.
        """

        self.finalize()

        return [
            track
            for track in self.tracks
            if (
                track.track_id
                in self.projected_track_evaluations
                and self.projected_track_evaluations[
                    track.track_id
                ].is_projection_reliable
            )
        ]

    def get_track_evaluation(
        self,
        track_id: int,
    ) -> TrackEvaluation | None:
        """
        Geef de trackkwaliteitsevaluatie van één track terug.
        """

        self.finalize()

        return self.track_evaluations.get(track_id)

    def get_projected_track_evaluation(
        self,
        track_id: int,
    ) -> ProjectedTrackEvaluation | None:
        """
        Geef de projectiekwaliteitsevaluatie van één track terug.
        """

        self.finalize()

        return self.projected_track_evaluations.get(track_id)

    def get_active_player_evaluation(
        self,
        track_id: int,
    ) -> ActivePlayerEvaluation | None:
        """
        Geef de actieve-spelerevaluatie van één track terug.
        """

        self.finalize()

        return self.active_player_evaluations.get(track_id)

    def print_summary(self) -> None:
        """
        Toon een samenvatting van tracks, kwaliteit, actieve spelers
        en beschikbare veldprojecties.
        """

        self.finalize()

        tracks = self.tracks

        print()
        print("=" * 64)
        print("Track Engine-overzicht")
        print("=" * 64)
        print(f"Aantal unieke tracks: {self.number_of_tracks}")
        print(
            "Veldprojectie actief: "
            f"{'ja' if self.has_field_projection else 'nee'}"
        )

        if not tracks:
            print("Geen tracks geregistreerd.")
            return

        tracks_by_frames_seen = sorted(
            tracks,
            key=lambda track: track.frames_seen,
            reverse=True,
        )

        print()
        print("Top 10 langst zichtbare tracks")
        print("-" * 64)

        for track in tracks_by_frames_seen[:10]:
            track_evaluation = self.track_evaluations[
                track.track_id
            ]

            projected_evaluation = (
                self.projected_track_evaluations[
                    track.track_id
                ]
            )

            active_evaluation = (
                self.active_player_evaluations[
                    track.track_id
                ]
            )

            print()
            print(f"Track {track.track_id}")
            print(
                "  Trackclassificatie : "
                f"{track_evaluation.classification}"
            )
            print(
                "  Quality score      : "
                f"{track_evaluation.quality_score:.1f}"
            )
            print(
                "  Stability score    : "
                f"{track_evaluation.stability_score:.1f}"
            )
            print(
                "  Actieve-spelerscore: "
                f"{active_evaluation.active_player_score:.1f}"
            )
            print(
                "  Spelerclassificatie: "
                f"{active_evaluation.classification}"
            )
            print(
                "  Movement score     : "
                f"{active_evaluation.movement_score:.1f}"
            )
            print(
                "  Displacement score : "
                f"{active_evaluation.displacement_score:.1f}"
            )
            print(
                "  Size score         : "
                f"{active_evaluation.size_score:.1f}"
            )
            print(
                "  Frames gezien      : "
                f"{track.frames_seen}"
            )
            print(
                "  Eerste–laatste     : "
                f"{track.first_frame}–{track.last_frame}"
            )
            print(
                "  Lifespan           : "
                f"{track.lifespan_frames} frames"
            )
            print(
                "  Zichtbaarheidsratio: "
                f"{track.visibility_ratio:.1%}"
            )
            print(
                "  Totale afstand     : "
                f"{track.total_distance_pixels:.1f} px"
            )
            print(
                "  Verplaatsing       : "
                f"{track.displacement_pixels:.1f} px"
            )
            print(
                "  Gem. boxhoogte     : "
                f"{track.average_box_height:.1f} px"
            )
            print(
                "  Gem. confidence    : "
                f"{track.average_confidence:.3f}"
            )

            if track.start_position is not None:
                start_x, start_y = track.start_position

                print(
                    "  Startpositie beeld : "
                    f"({start_x:.1f}, {start_y:.1f}) px"
                )

            if track.end_position is not None:
                end_x, end_y = track.end_position

                print(
                    "  Eindpositie beeld  : "
                    f"({end_x:.1f}, {end_y:.1f}) px"
                )

            if self.has_field_projection:
                self._print_track_field_summary(
                    track=track,
                    evaluation=projected_evaluation,
                )

            if active_evaluation.rejection_reasons:
                rejection_text = ", ".join(
                    active_evaluation.rejection_reasons
                )

                print(
                    "  Afwijsredenen      : "
                    f"{rejection_text}"
                )

        self._print_track_quality_summary()
        self._print_active_player_summary()
        self._print_field_projection_summary()
        self._print_projected_track_summary()

    def _print_track_field_summary(
        self,
        track: TrackState,
        evaluation: ProjectedTrackEvaluation,
    ) -> None:
        """
        Toon de veldinformatie en projectiekwaliteit van één track.
        """

        print(
            "  Geprojecteerde frames: "
            f"{track.projected_frames}"
        )
        print(
            "  Binnen veld         : "
            f"{track.inside_pitch_frames}"
        )
        print(
            "  Buiten veld         : "
            f"{track.outside_pitch_frames}"
        )
        print(
            "  Binnen-veldratio    : "
            f"{track.inside_pitch_ratio:.1%}"
        )
        print(
            "  Acceptabele ratio   : "
            f"{evaluation.acceptable_pitch_ratio:.1%}"
        )
        print(
            "  Buitenposities      : "
            f"{evaluation.outside_position_count}"
        )
        print(
            "  Buiten ≤1 meter     : "
            f"{evaluation.tolerated_outside_count}"
        )
        print(
            "  Ernstig buiten      : "
            f"{evaluation.severe_outside_count}"
        )
        print(
            "  Gem. afstand buiten : "
            f"{evaluation.average_outside_distance_meters:.2f} m"
        )
        print(
            "  Max. afstand buiten : "
            f"{evaluation.maximum_outside_distance_meters:.2f} m"
        )
        print(
            "  Afstand op veld     : "
            f"{track.total_distance_meters:.2f} m"
        )
        print(
            "  Verplaatsing veld   : "
            f"{track.displacement_meters:.2f} m"
        )
        print(
            "  Projectiescore      : "
            f"{evaluation.projection_quality_score:.1f}"
        )
        print(
            "  Projectieklasse     : "
            f"{evaluation.classification}"
        )
        print(
            "  Projectiedekking    : "
            f"{evaluation.projection_coverage:.1%}"
        )
        print(
            "  Gem. veldstap       : "
            f"{evaluation.average_step_meters:.2f} m"
        )
        print(
            "  Grootste veldstap   : "
            f"{evaluation.maximum_step_meters:.2f} m"
        )
        print(
            "  Geldige veldstappen : "
            f"{evaluation.valid_step_count}"
        )
        print(
            "  Onrealistische stappen: "
            f"{evaluation.unrealistic_jump_count}"
        )
        print(
            "  Extreme stappen     : "
            f"{evaluation.extreme_jump_count}"
        )
        print(
            "  Projectie bruikbaar : "
            f"{'ja' if evaluation.is_projection_usable else 'nee'}"
        )
        print(
            "  Projectie betrouwbaar: "
            f"{'ja' if evaluation.is_projection_reliable else 'nee'}"
        )

        if track.start_field_position is not None:
            start_x, start_y = track.start_field_position

            print(
                "  Startpositie veld   : "
                f"({start_x:.2f}, {start_y:.2f}) m"
            )

        if track.end_field_position is not None:
            end_x, end_y = track.end_field_position

            print(
                "  Eindpositie veld    : "
                f"({end_x:.2f}, {end_y:.2f}) m"
            )

        if evaluation.rejection_reasons:
            rejection_text = ", ".join(
                evaluation.rejection_reasons
            )

            print(
                "  Projectie-afwijzing : "
                f"{rejection_text}"
            )

    def _print_track_quality_summary(self) -> None:
        classifications = {
            "noise": 0,
            "short": 0,
            "weak": 0,
            "usable": 0,
            "stable": 0,
        }

        for evaluation in self.track_evaluations.values():
            classification = evaluation.classification

            if classification in classifications:
                classifications[classification] += 1

        usable_tracks = [
            evaluation
            for evaluation in self.track_evaluations.values()
            if evaluation.is_usable
        ]

        stable_tracks = [
            evaluation
            for evaluation in self.track_evaluations.values()
            if evaluation.is_stable
        ]

        print()
        print("-" * 64)
        print("Trackkwaliteit")
        print("-" * 64)
        print(
            f"Noise tracks   : {classifications['noise']}"
        )
        print(
            f"Short tracks   : {classifications['short']}"
        )
        print(
            f"Weak tracks    : {classifications['weak']}"
        )
        print(
            f"Usable tracks  : {classifications['usable']}"
        )
        print(
            f"Stable tracks  : {classifications['stable']}"
        )

        print()
        print(
            "Totaal bruikbare tracks : "
            f"{len(usable_tracks)}"
        )
        print(
            "Totaal stabiele tracks  : "
            f"{len(stable_tracks)}"
        )

    def _print_active_player_summary(self) -> None:
        classifications = {
            "rejected": 0,
            "weak_active_player_candidate": 0,
            "possible_active_player": 0,
            "likely_active_player": 0,
        }

        for evaluation in (
            self.active_player_evaluations.values()
        ):
            classification = evaluation.classification

            if classification in classifications:
                classifications[classification] += 1

        candidate_tracks = [
            evaluation
            for evaluation in (
                self.active_player_evaluations.values()
            )
            if evaluation.is_candidate
        ]

        likely_active_tracks = [
            evaluation
            for evaluation in (
                self.active_player_evaluations.values()
            )
            if evaluation.is_likely_active_player
        ]

        print()
        print("-" * 64)
        print("Actieve-spelerselectie")
        print("-" * 64)
        print(
            "Afgewezen tracks         : "
            f"{classifications['rejected']}"
        )
        print(
            "Zwakke kandidaten        : "
            f"{classifications['weak_active_player_candidate']}"
        )
        print(
            "Mogelijke actieve spelers: "
            f"{classifications['possible_active_player']}"
        )
        print(
            "Waarschijnlijke spelers  : "
            f"{classifications['likely_active_player']}"
        )

        print()
        print(
            "Totaal kandidaten        : "
            f"{len(candidate_tracks)}"
        )
        print(
            "Totaal waarschijnlijk actief: "
            f"{len(likely_active_tracks)}"
        )

    def _print_field_projection_summary(self) -> None:
        """
        Toon een algemeen overzicht van de veldprojectie.
        """

        print()
        print("-" * 64)
        print("Veldprojectie")
        print("-" * 64)

        if not self.has_field_projection:
            print("Geen FieldProjector ingesteld.")
            return

        total_projected_frames = sum(
            track.projected_frames
            for track in self.tracks
        )

        total_inside_frames = sum(
            track.inside_pitch_frames
            for track in self.tracks
        )

        total_outside_frames = sum(
            track.outside_pitch_frames
            for track in self.tracks
        )

        classified_frames = (
            total_inside_frames
            + total_outside_frames
        )

        inside_ratio = (
            total_inside_frames / classified_frames
            if classified_frames > 0
            else 0.0
        )

        tracks_with_projection = sum(
            track.projected_frames > 0
            for track in self.tracks
        )

        tracks_mostly_inside = sum(
            track.projected_frames > 0
            and track.inside_pitch_ratio >= 0.80
            for track in self.tracks
        )

        print(
            "Tracks met veldposities : "
            f"{tracks_with_projection}"
        )
        print(
            "Geprojecteerde metingen : "
            f"{total_projected_frames}"
        )
        print(
            "Metingen binnen veld    : "
            f"{total_inside_frames}"
        )
        print(
            "Metingen buiten veld    : "
            f"{total_outside_frames}"
        )
        print(
            "Totale binnen-veldratio : "
            f"{inside_ratio:.1%}"
        )
        print(
            "Tracks ≥80% binnen veld : "
            f"{tracks_mostly_inside}"
        )

    def _print_projected_track_summary(self) -> None:
        """
        Toon een samenvatting van de projectiekwaliteit.
        """

        print()
        print("-" * 64)
        print("Projected Track Evaluation")
        print("-" * 64)

        if not self.has_field_projection:
            print("Geen FieldProjector ingesteld.")
            return

        evaluations = list(
            self.projected_track_evaluations.values()
        )

        if not evaluations:
            print("Geen projectie-evaluaties beschikbaar.")
            return

        classifications = {
            "no_projection": 0,
            "bad_projection": 0,
            "weak_projection": 0,
            "usable_projection": 0,
            "reliable_projection": 0,
        }

        for evaluation in evaluations:
            classification = evaluation.classification

            if classification in classifications:
                classifications[classification] += 1

        available_evaluations = [
            evaluation
            for evaluation in evaluations
            if evaluation.is_projection_available
        ]

        usable_evaluations = [
            evaluation
            for evaluation in evaluations
            if evaluation.is_projection_usable
        ]

        reliable_evaluations = [
            evaluation
            for evaluation in evaluations
            if evaluation.is_projection_reliable
        ]

        average_quality_score = self._average(
            values=[
                evaluation.projection_quality_score
                for evaluation in available_evaluations
            ]
        )

        average_inside_pitch_ratio = self._average(
            values=[
                evaluation.inside_pitch_ratio
                for evaluation in available_evaluations
            ]
        )

        average_acceptable_pitch_ratio = self._average(
            values=[
                evaluation.acceptable_pitch_ratio
                for evaluation in available_evaluations
            ]
        )

        average_projection_coverage = self._average(
            values=[
                evaluation.projection_coverage
                for evaluation in available_evaluations
            ]
        )

        average_outside_distance = self._average(
            values=[
                evaluation.average_outside_distance_meters
                for evaluation in available_evaluations
                if evaluation.outside_position_count > 0
            ]
        )

        maximum_outside_distance = max(
            (
                evaluation.maximum_outside_distance_meters
                for evaluation in available_evaluations
            ),
            default=0.0,
        )

        total_valid_steps = sum(
            evaluation.valid_step_count
            for evaluation in evaluations
        )

        total_unrealistic_jumps = sum(
            evaluation.unrealistic_jump_count
            for evaluation in evaluations
        )

        total_extreme_jumps = sum(
            evaluation.extreme_jump_count
            for evaluation in evaluations
        )

        total_outside_positions = sum(
            evaluation.outside_position_count
            for evaluation in evaluations
        )

        total_tolerated_outside = sum(
            evaluation.tolerated_outside_count
            for evaluation in evaluations
        )

        total_severe_outside = sum(
            evaluation.severe_outside_count
            for evaluation in evaluations
        )

        print(
            "Geen projectie          : "
            f"{classifications['no_projection']}"
        )
        print(
            "Slechte projectie       : "
            f"{classifications['bad_projection']}"
        )
        print(
            "Zwakke projectie        : "
            f"{classifications['weak_projection']}"
        )
        print(
            "Bruikbare projectie     : "
            f"{classifications['usable_projection']}"
        )
        print(
            "Betrouwbare projectie   : "
            f"{classifications['reliable_projection']}"
        )

        print()
        print(
            "Tracks met projectie    : "
            f"{len(available_evaluations)}"
        )
        print(
            "Totaal bruikbaar        : "
            f"{len(usable_evaluations)}"
        )
        print(
            "Totaal betrouwbaar      : "
            f"{len(reliable_evaluations)}"
        )
        print(
            "Gemiddelde kwaliteit    : "
            f"{average_quality_score:.1f}"
        )
        print(
            "Gemiddelde dekking      : "
            f"{average_projection_coverage:.1%}"
        )
        print(
            "Gemiddelde binnenratio  : "
            f"{average_inside_pitch_ratio:.1%}"
        )
        print(
            "Gem. acceptabele ratio  : "
            f"{average_acceptable_pitch_ratio:.1%}"
        )
        print(
            "Buitenprojecties        : "
            f"{total_outside_positions}"
        )
        print(
            "Buiten ≤1 meter         : "
            f"{total_tolerated_outside}"
        )
        print(
            "Ernstig buiten >5 meter : "
            f"{total_severe_outside}"
        )
        print(
            "Gem. afstand buiten     : "
            f"{average_outside_distance:.2f} m"
        )
        print(
            "Max. afstand buiten     : "
            f"{maximum_outside_distance:.2f} m"
        )
        print(
            "Geldige veldstappen     : "
            f"{total_valid_steps}"
        )
        print(
            "Onrealistische stappen  : "
            f"{total_unrealistic_jumps}"
        )
        print(
            "Extreme stappen         : "
            f"{total_extreme_jumps}"
        )

        ranked_evaluations = sorted(
            available_evaluations,
            key=lambda evaluation: (
                evaluation.projection_quality_score
            ),
            reverse=True,
        )

        if not ranked_evaluations:
            return

        print()
        print("Beste projecties")
        print("-" * 64)

        for evaluation in ranked_evaluations[:5]:
            print(
                f"Track {evaluation.track_id}: "
                f"{evaluation.projection_quality_score:.1f} "
                f"({evaluation.classification}), "
                f"binnen {evaluation.inside_pitch_ratio:.1%}, "
                f"acceptabel "
                f"{evaluation.acceptable_pitch_ratio:.1%}, "
                f"max buiten "
                f"{evaluation.maximum_outside_distance_meters:.2f} m, "
                f"max stap "
                f"{evaluation.maximum_step_meters:.2f} m"
            )

        print()
        print("Zwakste projecties")
        print("-" * 64)

        for evaluation in ranked_evaluations[-5:]:
            print(
                f"Track {evaluation.track_id}: "
                f"{evaluation.projection_quality_score:.1f} "
                f"({evaluation.classification}), "
                f"binnen {evaluation.inside_pitch_ratio:.1%}, "
                f"acceptabel "
                f"{evaluation.acceptable_pitch_ratio:.1%}, "
                f"max buiten "
                f"{evaluation.maximum_outside_distance_meters:.2f} m, "
                f"max stap "
                f"{evaluation.maximum_step_meters:.2f} m"
            )

    @classmethod
    def _get_field_dimensions(
        cls,
        field_projector: FieldProjector | None,
    ) -> tuple[float, float]:
        """
        Lees de veldafmetingen defensief uit de FieldProjector.

        Wanneer geen geschikte attributen worden gevonden, worden de
        standaardafmetingen van het huidige 8-tegen-8-veld gebruikt.
        """

        default_length = 64.0
        default_width = 42.0

        if field_projector is None:
            return default_length, default_width

        possible_sources = [
            field_projector,
            getattr(field_projector, "calibration", None),
            getattr(field_projector, "pitch_model", None),
            getattr(field_projector, "field_model", None),
        ]

        field_length = cls._extract_positive_number(
            sources=possible_sources,
            attribute_names=(
                "field_length_meters",
                "pitch_length_meters",
                "field_length",
                "pitch_length",
                "length_meters",
                "length",
            ),
            default=default_length,
        )

        field_width = cls._extract_positive_number(
            sources=possible_sources,
            attribute_names=(
                "field_width_meters",
                "pitch_width_meters",
                "field_width",
                "pitch_width",
                "width_meters",
                "width",
            ),
            default=default_width,
        )

        return field_length, field_width

    @staticmethod
    def _extract_positive_number(
        sources: list[Any],
        attribute_names: tuple[str, ...],
        default: float,
    ) -> float:
        """
        Zoek een positieve numerieke attribuutwaarde.
        """

        for source in sources:
            if source is None:
                continue

            for attribute_name in attribute_names:
                value = getattr(
                    source,
                    attribute_name,
                    None,
                )

                if value is None:
                    continue

                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue

                if numeric_value > 0.0:
                    return numeric_value

        return float(default)

    @staticmethod
    def _average(
        values: list[float],
    ) -> float:
        if not values:
            return 0.0

        return sum(values) / len(values)