from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot


@dataclass
class TrackState:
    """
    Historische informatie van één ByteTrack-track.

    De track bevat zowel beeldcoördinaten in pixels als, wanneer een
    FieldProjector beschikbaar is, veldcoördinaten in meters.
    """

    track_id: int

    first_frame: int
    last_frame: int

    frames_seen: int = 0

    # Middenpunten van bounding boxes in beeldcoördinaten.
    positions: list[tuple[float, float]] = field(
        default_factory=list
    )

    box_heights: list[float] = field(
        default_factory=list
    )

    confidences: list[float] = field(
        default_factory=list
    )

    # Per waarneming één veldpositie of None wanneer geen projectie
    # beschikbaar was.
    field_positions: list[
        tuple[float, float] | None
    ] = field(
        default_factory=list
    )

    # Per waarneming:
    # - True: positie ligt binnen het veld
    # - False: positie ligt buiten het veld
    # - None: geen veldprojectie beschikbaar
    inside_pitch_flags: list[
        bool | None
    ] = field(
        default_factory=list
    )

    total_distance_pixels: float = 0.0
    total_distance_meters: float = 0.0

    @property
    def lifespan_frames(self) -> int:
        """
        Aantal frames tussen het eerste en laatste waarnemingsmoment.
        """

        return self.last_frame - self.first_frame + 1

    @property
    def visibility_ratio(self) -> float:
        """
        Percentage van de levensduur waarin de track zichtbaar was.
        """

        if self.lifespan_frames <= 0:
            return 0.0

        return self.frames_seen / self.lifespan_frames

    @property
    def average_box_height(self) -> float:
        """
        Gemiddelde hoogte van de detectiebox.
        """

        if not self.box_heights:
            return 0.0

        return sum(self.box_heights) / len(
            self.box_heights
        )

    @property
    def average_confidence(self) -> float:
        """
        Gemiddelde confidence van alle detecties van deze track.
        """

        if not self.confidences:
            return 0.0

        return sum(self.confidences) / len(
            self.confidences
        )

    @property
    def start_position(
        self,
    ) -> tuple[float, float] | None:
        """
        Eerste geregistreerde beeldpositie.
        """

        if not self.positions:
            return None

        return self.positions[0]

    @property
    def end_position(
        self,
    ) -> tuple[float, float] | None:
        """
        Meest recente geregistreerde beeldpositie.
        """

        if not self.positions:
            return None

        return self.positions[-1]

    @property
    def displacement_pixels(self) -> float:
        """
        Hemelsbrede afstand tussen de eerste en laatste beeldpositie.
        """

        if len(self.positions) < 2:
            return 0.0

        start_x, start_y = self.positions[0]
        end_x, end_y = self.positions[-1]

        return hypot(
            end_x - start_x,
            end_y - start_y,
        )

    @property
    def valid_field_positions(
        self,
    ) -> list[tuple[float, float]]:
        """
        Alle beschikbare veldposities, zonder ontbrekende waarden.
        """

        return [
            position
            for position in self.field_positions
            if position is not None
        ]

    @property
    def projected_frames(self) -> int:
        """
        Aantal waarnemingen waarvoor een veldprojectie beschikbaar is.
        """

        return len(self.valid_field_positions)

    @property
    def inside_pitch_frames(self) -> int:
        """
        Aantal geprojecteerde waarnemingen binnen het speelveld.
        """

        return sum(
            flag is True
            for flag in self.inside_pitch_flags
        )

    @property
    def outside_pitch_frames(self) -> int:
        """
        Aantal geprojecteerde waarnemingen buiten het speelveld.
        """

        return sum(
            flag is False
            for flag in self.inside_pitch_flags
        )

    @property
    def inside_pitch_ratio(self) -> float:
        """
        Percentage geprojecteerde posities dat binnen het veld ligt.

        Waarnemingen zonder beschikbare projectie tellen niet mee.
        """

        classified_frames = (
            self.inside_pitch_frames
            + self.outside_pitch_frames
        )

        if classified_frames == 0:
            return 0.0

        return (
            self.inside_pitch_frames
            / classified_frames
        )

    @property
    def start_field_position(
        self,
    ) -> tuple[float, float] | None:
        """
        Eerste beschikbare positie op het veld.
        """

        for position in self.field_positions:
            if position is not None:
                return position

        return None

    @property
    def end_field_position(
        self,
    ) -> tuple[float, float] | None:
        """
        Laatste beschikbare positie op het veld.
        """

        for position in reversed(
            self.field_positions
        ):
            if position is not None:
                return position

        return None

    @property
    def displacement_meters(self) -> float:
        """
        Hemelsbrede afstand tussen de eerste en laatste veldpositie.
        """

        start_position = self.start_field_position
        end_position = self.end_field_position

        if (
            start_position is None
            or end_position is None
        ):
            return 0.0

        start_x, start_y = start_position
        end_x, end_y = end_position

        return hypot(
            end_x - start_x,
            end_y - start_y,
        )