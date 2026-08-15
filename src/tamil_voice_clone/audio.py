from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from .config import RuntimeConfig


@dataclass(frozen=True)
class ReferenceAudioInfo:
    path: Path
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float


class ReferenceAudioError(ValueError):
    pass


def inspect_reference(path: str | Path, config: RuntimeConfig | None = None) -> ReferenceAudioInfo:
    cfg = config or RuntimeConfig()
    audio_path = Path(path)
    if not audio_path.is_file():
        raise ReferenceAudioError(f"Reference audio not found: {audio_path}")

    info = sf.info(str(audio_path))
    duration = info.frames / info.samplerate if info.samplerate else 0.0

    if duration < cfg.min_reference_seconds:
        raise ReferenceAudioError(
            f"Reference audio is {duration:.1f}s; provide at least "
            f"{cfg.min_reference_seconds:.0f}s of clean speech."
        )
    if duration > cfg.max_reference_seconds:
        raise ReferenceAudioError(
            f"Reference audio is {duration:.1f}s; trim it below "
            f"{cfg.max_reference_seconds:.0f}s for predictable memory use."
        )
    if info.channels not in (1, 2):
        raise ReferenceAudioError("Reference audio must be mono or stereo.")

    return ReferenceAudioInfo(
        path=audio_path,
        sample_rate=info.samplerate,
        channels=info.channels,
        frames=info.frames,
        duration_seconds=duration,
    )
