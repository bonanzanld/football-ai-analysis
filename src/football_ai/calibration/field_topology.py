from __future__ import annotations


CORNER_CYCLE = (
    "linksachter",
    "rechtsachter",
    "rechtsvoor",
    "linksvoor",
)

# Directed in the canonical clockwise cycle. Reading either pair backwards is
# the same physical boundary.
BOUNDARY_CORNERS = {
    "sideline_rear": ("linksachter", "rechtsachter"),
    "end_line_b": ("rechtsachter", "rechtsvoor"),
    "sideline_front": ("rechtsvoor", "linksvoor"),
    "end_line_a": ("linksvoor", "linksachter"),
}

DUTCH_BOUNDARY_NAMES = {
    "sideline_rear": "zijlijn_achter",
    "end_line_b": "achterlijn_rechts",
    "sideline_front": "zijlijn_voor",
    "end_line_a": "achterlijn_links",
}


def ground_corners(
    pitch_length_m: float, pitch_width_m: float
) -> dict[str, tuple[float, float]]:
    return {
        "linksachter": (0.0, 0.0),
        "rechtsachter": (pitch_length_m, 0.0),
        "rechtsvoor": (pitch_length_m, pitch_width_m),
        "linksvoor": (0.0, pitch_width_m),
    }


def boundary_corner_pairs() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (name, *BOUNDARY_CORNERS[name])
        for name in ("sideline_rear", "end_line_b", "sideline_front", "end_line_a")
    )


def boundary_between(first: str, second: str) -> str:
    for name, pair in BOUNDARY_CORNERS.items():
        if (first, second) == pair or (second, first) == pair:
            return name
    raise ValueError(f"Niet-aangrenzende veldhoeken mogen niet worden verbonden: {first}, {second}.")
