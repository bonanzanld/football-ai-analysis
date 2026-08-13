from __future__ import annotations

import numpy as np

from football_ai.video_processor import VideoProcessor


class _Detector:
    pass


def test_video_processor_anonymizes_people_by_default():
    processor = VideoProcessor(detector=_Detector(), debug_homography=False)

    assert processor.anonymize_people is True


def test_video_processor_allows_explicit_internal_opt_out():
    processor = VideoProcessor(
        detector=_Detector(),
        debug_homography=False,
        anonymize_people=False,
    )

    assert processor.anonymize_people is False
