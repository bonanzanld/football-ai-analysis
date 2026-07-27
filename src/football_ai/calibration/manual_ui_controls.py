from __future__ import annotations


ZOOM_IN_KEYS = (ord("+"), ord("="))
ZOOM_OUT_KEYS = (ord("-"), ord("_"))
RESET_VIEW_KEYS = (ord("0"),)
UNDO_KEYS = (ord("u"), ord("U"), 8, 127)
FINISH_KEYS = (10, 13)
CANCEL_KEYS = (27,)

PAN_KEY_DIRECTIONS = {
    2424832: (-1, 0),
    65361: (-1, 0),
    ord("a"): (-1, 0),
    ord("A"): (-1, 0),
    2555904: (1, 0),
    65363: (1, 0),
    ord("d"): (1, 0),
    ord("D"): (1, 0),
    2490368: (0, -1),
    65362: (0, -1),
    ord("w"): (0, -1),
    ord("W"): (0, -1),
    2621440: (0, 1),
    65364: (0, 1),
    ord("s"): (0, 1),
    ord("S"): (0, 1),
}


def mouse_wheel_direction(flags: int) -> int:
    """Lees de OpenCV-muiswielrichting ook op builds zonder helperfunctie."""
    delta = (int(flags) >> 16) & 0xFFFF
    if delta & 0x8000:
        delta -= 0x10000
    return 1 if delta > 0 else -1
