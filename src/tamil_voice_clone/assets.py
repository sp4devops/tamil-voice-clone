from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelAsset:
    name: str
    url: str
    filename: str
    expected_size_bytes: int | None = None


CAMPPLUS_VOXCELEB = ModelAsset(
    name="CAMPPlus VoxCeleb speaker encoder",
    url=(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
    ),
    filename="3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    expected_size_bytes=29_596_978,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(asset: ModelAsset, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / asset.filename
    if destination.is_file():
        if asset.expected_size_bytes is None or destination.stat().st_size == asset.expected_size_bytes:
            return destination
        destination.unlink()

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(asset.url, headers={"User-Agent": "tamil-voice-clone/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)

    if asset.expected_size_bytes is not None and temporary.stat().st_size != asset.expected_size_bytes:
        size = temporary.stat().st_size
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded {asset.name} has unexpected size {size}; "
            f"expected {asset.expected_size_bytes}."
        )

    temporary.replace(destination)
    return destination
