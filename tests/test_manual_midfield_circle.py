import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "collect_manual_midfield_circle.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("collect_manual_midfield_circle", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_payload_preserves_raw_points_without_claiming_ellipse():
    collector = object.__new__(MODULE.MidfieldCircleCollector)
    collector.video = Path("clip.mov")
    collector.fps = 30.0
    collector.time = 10.0
    collector.required_points = 7
    collector.points = [(float(index), float(index * 2)) for index in range(7)]

    payload = collector._payload(True)

    assert payload["points"] == [[float(index), float(index * 2)] for index in range(7)]
    assert payload["fit_status"] == "pending_camera_constrained_fit"
    assert "ellipse" not in payload
