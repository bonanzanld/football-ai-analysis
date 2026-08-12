import subprocess
import sys


def test_yolo_train_help_exposes_low_light_profile():
    result = subprocess.run(
        [sys.executable, "tools/train_ball_detector_yolo.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "low-light-tiny-ball" in result.stdout
