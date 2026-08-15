from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import SpeakerCondition
from .speaker import l2_normalize

CACHE_VERSION = 1


@dataclass(frozen=True)
class VoiceCacheInfo:
    name: str
    source_seconds: float
    embedding_size: int
    version: int = CACHE_VERSION


def save_voice_cache(path: Path, name: str, condition: SpeakerCondition) -> VoiceCacheInfo:
    path.parent.mkdir(parents=True, exist_ok=True)
    embedding = l2_normalize(condition.embedding)
    info = VoiceCacheInfo(
        name=name,
        source_seconds=float(condition.source_seconds),
        embedding_size=int(embedding.size),
    )
    metadata = json.dumps(
        {
            "version": info.version,
            "name": info.name,
            "source_seconds": info.source_seconds,
            "embedding_size": info.embedding_size,
        },
        separators=(",", ":"),
    )
    np.savez_compressed(path, embedding=embedding, metadata=np.array(metadata))
    return info


def load_voice_cache(path: Path) -> tuple[VoiceCacheInfo, SpeakerCondition]:
    with np.load(path, allow_pickle=False) as archive:
        embedding = l2_normalize(np.asarray(archive["embedding"], dtype=np.float32))
        metadata = json.loads(str(archive["metadata"].item()))

    version = int(metadata.get("version", 0))
    if version != CACHE_VERSION:
        raise ValueError(f"Unsupported voice cache version: {version}")
    expected_size = int(metadata["embedding_size"])
    if embedding.size != expected_size:
        raise ValueError(
            f"Voice cache embedding has {embedding.size} values; expected {expected_size}."
        )

    info = VoiceCacheInfo(
        name=str(metadata["name"]),
        source_seconds=float(metadata["source_seconds"]),
        embedding_size=expected_size,
        version=version,
    )
    condition = SpeakerCondition(
        embedding=embedding,
        source_seconds=info.source_seconds,
    )
    return info, condition
