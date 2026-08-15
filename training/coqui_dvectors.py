"""Build a YourTTS/Coqui-compatible d-vector file from CAMPPlus embeddings.

This is a training-only bridge. The final inference runtime does not depend on
Coqui TTS or PyTorch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_embedding(path: Path) -> list[float]:
    with np.load(path, allow_pickle=False) as archive:
        embedding = np.asarray(archive["embedding"], dtype=np.float32).reshape(-1)
    if embedding.size != 512:
        raise ValueError(f"Expected CAMPPlus 512-D embedding, got {embedding.size}: {path}")
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"Invalid speaker embedding: {path}")
    return (embedding / norm).tolist()


def build_mapping(index_jsonl: Path, root: Path) -> dict[str, dict[str, object]]:
    """Load rows containing `key`, `speaker`, and `embedding` fields."""
    mapping: dict[str, dict[str, object]] = {}
    with index_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["key"])
            speaker = str(row["speaker"])
            embedding_path = root / str(row["embedding"])
            if key in mapping:
                raise ValueError(f"Duplicate d-vector key at line {line_number}: {key}")
            mapping[key] = {
                "name": speaker,
                "embedding": load_embedding(embedding_path),
            }
    if not mapping:
        raise ValueError("No d-vector records found")
    return mapping


def save_coqui_mapping(mapping: dict[str, dict[str, object]], output: Path) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required only for writing Coqui's .pth format") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mapping, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path, help="JSONL with key/speaker/embedding fields")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mapping = build_mapping(args.index, args.root)
    save_coqui_mapping(mapping, args.output)
    print(f"entries={len(mapping)} output={args.output}")


if __name__ == "__main__":
    main()
