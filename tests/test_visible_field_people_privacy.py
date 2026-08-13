from __future__ import annotations

import numpy as np

from tools.qa_visible_field_people import _anonymize_detected_people


def test_visible_field_people_anonymizes_raw_detected_person():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[10:70, 30:60] = np.arange(60, dtype=np.uint8)[:, None, None]

    result = _anonymize_detected_people(
        frame,
        np.asarray([[30.0, 10.0, 60.0, 70.0]]),
    )

    assert not np.array_equal(result[10:28, 30:60], frame[10:28, 30:60])
    np.testing.assert_array_equal(result[40:70, 30:60], frame[40:70, 30:60])
