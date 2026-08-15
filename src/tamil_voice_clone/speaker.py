from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import RuntimeConfig
from .model import SpeakerCondition

SPEAKER_SAMPLE_RATE = 16000


class SpeakerEncoderError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeakerEncoderConfig:
    model_path: Path
    num_threads: int = 2
    provider: str = "cpu"


def _mono(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise SpeakerEncoderError("Expected mono or stereo audio.")
    return audio


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Small dependency-free resampler used only for speaker conditioning.

    The final TTS audio path can use a higher-quality resampler. Speaker embedding
    extraction only needs stable 16 kHz speech features, so this avoids pulling
    scipy/librosa into the low-memory runtime.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise SpeakerEncoderError("Sample rates must be positive.")
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    if len(samples) < 2:
        return np.asarray(samples, dtype=np.float32)

    output_length = max(1, round(len(samples) * target_rate / source_rate))
    source_positions = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=output_length, endpoint=False)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def load_reference_16k(path: Path) -> tuple[np.ndarray, float]:
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    mono = _mono(samples)
    duration = len(mono) / sample_rate if sample_rate else 0.0
    return _resample_linear(mono, sample_rate, SPEAKER_SAMPLE_RATE), duration


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    value = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise SpeakerEncoderError("Speaker encoder returned an invalid embedding.")
    return value / norm


class SherpaOnnxSpeakerEncoder:
    """CPU-only arbitrary-speaker encoder backed by sherpa-onnx.

    sherpa-onnx is imported lazily so the base package remains tiny and tests do
    not require the native runtime. Install with `pip install -e '.[speaker]'`.
    """

    def __init__(
        self,
        encoder_config: SpeakerEncoderConfig,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        self.encoder_config = encoder_config
        self.runtime_config = runtime_config or RuntimeConfig()
        if not encoder_config.model_path.is_file():
            raise SpeakerEncoderError(
                f"Speaker encoder model not found: {encoder_config.model_path}"
            )

        try:
            sherpa_onnx = importlib.import_module("sherpa_onnx")
        except ImportError as exc:
            raise SpeakerEncoderError(
                "sherpa-onnx is not installed; install the 'speaker' extra."
            ) from exc

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(encoder_config.model_path),
            num_threads=encoder_config.num_threads,
            provider=encoder_config.provider,
        )
        if hasattr(config, "validate") and not config.validate():
            raise SpeakerEncoderError("Invalid sherpa-onnx speaker encoder configuration.")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def encode_speaker(self, reference_audio: Path) -> SpeakerCondition:
        samples, duration = load_reference_16k(reference_audio)
        if duration < self.runtime_config.min_reference_seconds:
            raise SpeakerEncoderError(
                f"Reference contains {duration:.1f}s; provide at least "
                f"{self.runtime_config.min_reference_seconds:.0f}s."
            )

        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=SPEAKER_SAMPLE_RATE, waveform=samples.tolist())
        stream.input_finished()
        embedding = self._extractor.compute(stream)
        return SpeakerCondition(
            embedding=l2_normalize(np.asarray(embedding, dtype=np.float32)),
            source_seconds=duration,
        )
