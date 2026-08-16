#!/usr/bin/env python3
"""Batch-convert Parler source WAVs with pinned Seed-VC v2."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-vc-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=40)
    parser.add_argument("--intelligibility", type=float, default=0.85)
    parser.add_argument("--similarity", type=float, default=0.90)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--convert-style", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.diffusion_steps <= 100:
        parser.error("--diffusion-steps must be between 1 and 100")
    for name in ("intelligibility", "similarity", "top_p"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be positive")
    return args


def validate_inputs(args: argparse.Namespace) -> list[Path]:
    seed_vc_dir = args.seed_vc_dir.resolve()
    source_dir = args.source_dir.resolve()
    reference = args.reference.resolve()
    if not (seed_vc_dir / "configs/v2/vc_wrapper.yaml").is_file():
        raise FileNotFoundError("Pinned Seed-VC v2 config is missing")
    if not reference.is_file():
        raise FileNotFoundError(f"Reference audio not found: {reference}")
    reference_info = sf.info(reference)
    if reference_info.duration < 20.0:
        raise ValueError(
            f"Reference is only {reference_info.duration:.2f}s; high-quality conversion "
            "requires the agreed 20–30 second reference"
        )
    sources = sorted(source_dir.glob("*.wav"))
    if not sources:
        raise FileNotFoundError(f"No source WAV files found in {source_dir}")
    return sources


def load_wrapper(seed_vc_dir: Path):
    import torch
    import yaml
    from hydra.utils import instantiate
    from omegaconf import DictConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    sys.path.insert(0, str(seed_vc_dir))
    with working_directory(seed_vc_dir):
        config = DictConfig(
            yaml.safe_load((seed_vc_dir / "configs/v2/vc_wrapper.yaml").read_text())
        )
        wrapper = instantiate(config)
        wrapper.load_checkpoints(ar_checkpoint_path=None, cfm_checkpoint_path=None)
        wrapper.to(device=device, dtype=dtype)
        wrapper.eval()
        wrapper.setup_ar_caches(
            max_batch_size=1,
            max_seq_len=4096,
            dtype=dtype,
            device=device,
        )
    return wrapper, device, dtype


def convert_one(
    wrapper,
    source: Path,
    reference: Path,
    args: argparse.Namespace,
    device,
    dtype,
) -> tuple[int, np.ndarray]:
    with working_directory(args.seed_vc_dir.resolve()):
        generator = wrapper.convert_voice_with_streaming(
            source_audio_path=str(source),
            target_audio_path=str(reference),
            diffusion_steps=args.diffusion_steps,
            length_adjust=1.0,
            intelligebility_cfg_rate=args.intelligibility,
            similarity_cfg_rate=args.similarity,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            convert_style=args.convert_style,
            anonymization_only=False,
            device=device,
            dtype=dtype,
            stream_output=True,
        )
        final_audio = None
        for output in generator:
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("Unexpected Seed-VC streaming output")
            _, final_audio = output

    if final_audio is None or not isinstance(final_audio, tuple) or len(final_audio) != 2:
        raise RuntimeError(f"Seed-VC returned no final audio for {source.name}")
    sample_rate, audio = final_audio
    audio_array = np.asarray(audio, dtype=np.float32).squeeze()
    if audio_array.ndim != 1 or audio_array.size < int(sample_rate) // 4:
        raise RuntimeError(f"Invalid Seed-VC output for {source.name}")
    if not np.isfinite(audio_array).all():
        raise RuntimeError(f"Non-finite Seed-VC output for {source.name}")
    return int(sample_rate), audio_array


def main() -> None:
    args = parse_args()
    sources = validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = args.reference.resolve()
    reference_info = sf.info(reference)

    wrapper, device, dtype = load_wrapper(args.seed_vc_dir.resolve())
    report: dict[str, object] = {
        "reference": {
            "duration_seconds": round(reference_info.duration, 3),
            "sample_rate": reference_info.samplerate,
            "channels": reference_info.channels,
        },
        "device": str(device),
        "dtype": str(dtype),
        "settings": {
            "diffusion_steps": args.diffusion_steps,
            "intelligibility_cfg_rate": args.intelligibility,
            "similarity_cfg_rate": args.similarity,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "convert_style": args.convert_style,
        },
        "cases": [],
    }

    for source in sources:
        started = time.monotonic()
        print(f"Converting {source.name} on {device} with {dtype}", flush=True)
        sample_rate, audio = convert_one(
            wrapper,
            source.resolve(),
            reference,
            args,
            device,
            dtype,
        )
        peak = float(np.max(np.abs(audio)))
        if peak > 0.98:
            audio = audio * (0.98 / peak)
            peak = 0.98
        destination = args.output_dir / source.name
        sf.write(destination, audio, sample_rate, subtype="PCM_24")
        report["cases"].append(
            {
                "source": source.name,
                "output": destination.name,
                "duration_seconds": round(audio.size / sample_rate, 3),
                "peak": round(peak, 6),
                "conversion_seconds": round(time.monotonic() - started, 3),
            }
        )
        (args.output_dir / "seed_vc_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        del audio
        gc.collect()

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
