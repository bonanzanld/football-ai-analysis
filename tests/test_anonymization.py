import numpy as np

from football_ai.privacy import anonymize_people_heads


def test_anonymization_changes_head_but_not_lower_body_or_input():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pattern = np.indices((60, 30))[1][..., None] * 8
    frame[10:70, 30:60] = np.repeat(pattern, 3, axis=2)
    original = frame.copy()

    result = anonymize_people_heads(frame, np.asarray([[30, 10, 60, 70]], dtype=float))

    assert not np.array_equal(result[10:28, 26:64], original[10:28, 26:64])
    assert np.array_equal(result[35:70, 30:60], original[35:70, 30:60])
    assert np.array_equal(frame, original)


def test_anonymization_clips_boxes_at_frame_edges():
    frame = np.random.default_rng(0).integers(0, 255, (30, 30, 3), dtype=np.uint8)
    result = anonymize_people_heads(frame, np.asarray([[-5, -5, 10, 20]], dtype=float))
    assert result.shape == frame.shape
