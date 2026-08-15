from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .cache import load_voice_cache
from .onnx_synth import OnnxSynthesizer
from .phonemes import EspeakPhonemizer
from .tokenizer import PhonemeVocabulary


class VoiceCloningPipeline:
    """End-to-end low-memory inference using precomputed speaker conditions."""

    def __init__(
        self,
        synthesizer: OnnxSynthesizer,
        vocabulary: PhonemeVocabulary,
        phonemizer: EspeakPhonemizer,
    ) -> None:
        self.synthesizer = synthesizer
        self.vocabulary = vocabulary
        self.phonemizer = phonemizer

    def synthesize_from_cache(self, text: str, voice_cache: Path) -> np.ndarray:
        _, speaker = load_voice_cache(voice_cache)
        spans = self.phonemizer.phonemize(text)
        tokens = self.vocabulary.encode(spans)
        return self.synthesizer.synthesize_tokens(tokens, speaker)

    def synthesize_to_file(self, text: str, voice_cache: Path, output: Path) -> Path:
        waveform = self.synthesize_from_cache(text, voice_cache)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output), waveform, self.synthesizer.sample_rate, subtype="PCM_16")
        return output
