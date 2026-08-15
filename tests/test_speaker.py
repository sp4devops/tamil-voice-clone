import numpy as np
import pytest

from tamil_voice_clone.speaker import SpeakerEncoderError, _resample_linear, l2_normalize


def test_l2_normalize_returns_unit_vector() -> None:
    vector = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert np.linalg.norm(vector) == pytest.approx(1.0, rel=1e-6)
    assert vector.tolist() == pytest.approx([0.6, 0.8], rel=1e-6)


def test_l2_normalize_rejects_zero_vector() -> None:
    with pytest.raises(SpeakerEncoderError):
        l2_normalize(np.zeros(4, dtype=np.float32))


def test_linear_resample_changes_length() -> None:
    samples = np.linspace(-1.0, 1.0, num=8000, dtype=np.float32)
    result = _resample_linear(samples, source_rate=8000, target_rate=16000)
    assert result.dtype == np.float32
    assert len(result) == 16000
