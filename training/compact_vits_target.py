"""Training-time target configuration for the compact bilingual zero-shot model.

This file deliberately contains no runtime dependency on Coqui TTS. It records
our model budget and maps cleanly onto a YourTTS/VITS-style implementation that
uses external d-vectors and language embeddings.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompactVitsTarget:
    # Frontend / conditioning
    speaker_embedding_dim: int = 512
    num_languages: int = 2
    language_embedding_dim: int = 4

    # Audio
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    fft_size: int = 1024

    # Deliberately smaller than common VITS/YourTTS defaults.
    hidden_channels: int = 128
    text_encoder_ffn_channels: int = 384
    text_encoder_heads: int = 4
    text_encoder_layers: int = 6
    posterior_encoder_layers: int = 10
    flow_layers: int = 3
    decoder_initial_channels: int = 256
    decoder_resblock_type: str = "2"

    # Runtime acceptance gates. These are requirements, not measured claims.
    max_fp32_model_mb: int = 250
    target_quantized_model_mb: int = 120
    target_rss_mb: int = 1536
    hard_rss_mb: int = 2048
    max_cpu_threads: int = 2

    def validate(self) -> None:
        if self.speaker_embedding_dim != 512:
            raise ValueError("CAMPPlus boundary must remain 512-D for this experiment.")
        if self.num_languages != 2:
            raise ValueError("Initial experiment is intentionally Tamil + English only.")
        if self.hidden_channels % self.text_encoder_heads != 0:
            raise ValueError("hidden_channels must be divisible by text_encoder_heads.")
        if self.target_rss_mb > self.hard_rss_mb:
            raise ValueError("Target RSS cannot exceed the hard RSS ceiling.")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)


def write_target(path: Path) -> None:
    target = CompactVitsTarget()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(target.to_dict(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_target(Path("artifacts/compact_vits_target.json"))
