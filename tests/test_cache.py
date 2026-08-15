from pathlib import Path

import numpy as np
import pytest

from tamil_voice_clone.cache import load_voice_cache, save_voice_cache
from tamil_voice_clone.model import SpeakerCondition


def test_voice_cache_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "voice.npz"
    condition = SpeakerCondition(
        embedding=np.array([3.0, 4.0], dtype=np.float32),
        source_seconds=30.5,
    )

    saved = save_voice_cache(output, "tester", condition)
    loaded_info, loaded_condition = load_voice_cache(output)

    assert saved.name == "tester"
    assert loaded_info.name == "tester"
    assert loaded_info.source_seconds == pytest.approx(30.5)
    assert loaded_condition.source_seconds == pytest.approx(30.5)
    assert loaded_condition.embedding.tolist() == pytest.approx([0.6, 0.8], rel=1e-6)
