from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SpeakerCondition:
    embedding: np.ndarray
    source_seconds: float


class ZeroShotVoiceModel(Protocol):
    """Contract every real zero-shot backend must satisfy."""

    def encode_speaker(self, reference_audio: Path) -> SpeakerCondition:
        """Create a reusable speaker condition from arbitrary reference speech."""
        ...

    def synthesize(self, text: str, speaker: SpeakerCondition) -> np.ndarray:
        """Generate mono floating-point waveform samples."""
        ...


class ModelNotConfiguredError(RuntimeError):
    pass
