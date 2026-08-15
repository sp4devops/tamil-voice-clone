from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .tokenizer import TokenizedPhonemes


@dataclass(frozen=True)
class SpeakerCondition:
    embedding: np.ndarray
    source_seconds: float


class SpeakerEncoder(Protocol):
    def encode_speaker(self, reference_audio: Path) -> SpeakerCondition:
        """Create a reusable speaker condition from arbitrary reference speech."""
        ...


class SynthesizerBackend(Protocol):
    sample_rate: int

    def synthesize_tokens(
        self,
        tokens: TokenizedPhonemes,
        speaker: SpeakerCondition,
    ) -> np.ndarray:
        """Generate mono floating-point waveform samples from model-ready inputs."""
        ...


class ZeroShotVoiceModel(Protocol):
    """High-level contract for a complete arbitrary-speaker TTS pipeline."""

    def encode_speaker(self, reference_audio: Path) -> SpeakerCondition:
        ...

    def synthesize(self, text: str, speaker: SpeakerCondition) -> np.ndarray:
        ...


class ModelNotConfiguredError(RuntimeError):
    pass
