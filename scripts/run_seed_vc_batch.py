#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-vc-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=40)
    parser.add_argument("--intelligibility", type=float, default=0.85)
    parser.add_argument("--similarity", type=float, default=0.90)
    args = parser.parse_args()

    sys.path.insert(0, str(args.seed_vc_dir.resolve()))
    from inference_v2 import convert_voice_v2  # type: ignore

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_info = sf.info(args.reference)
    inference_args = SimpleNamespace(
        ar_checkpoint_path=None,
        cfm_checkpoint_path=None,
        diffusion_steps=args.diffusion_steps,
        length_adjust=1.0,
        intelligibility_cfg_rate=args.intelligibility,
        similarity_cfg_rate=args.similarity,
        top_p=0.85,
        temperature=0.85,
        repetition_penalty=1.1,
        convert_style=True,
        anonymization_only=False,
        compile=False,
    )

    report: dict[str, object] = {
        "reference": {
            "duration_seconds": round(reference_info.duration, 3),
            "sample_rate": reference_info.samplerate,
            "channels": reference_info.channels,
        },
        "settings": vars(inference_args),
        "cases": [],
    }

    for source in sorted(args.source_dir.glob("*.wav")):
        started = time.time()
        converted = convert_voice_v2(str(source), str(args.reference), inference_args)
        if converted is None:
            raise RuntimeError(f"Seed-VC returned no audio for {source.name}")
        sample_rate, audio = converted
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim != 1 or audio.size < int(sample_rate) // 4:
            raise RuntimeError(f"Invalid Seed-VC output for {source.name}")
        if not np.isfinite(audio).all():
            raise RuntimeError(f"Non-finite Seed-VC output for {source.name}")
        peak = float(np.max(np.abs(audio)))
        if peak > 0.98:
            audio = audio * (0.98 / peak)
            peak = 0.98
        destination = args.output_dir / source.name
        sf.write(destination, audio, int(sample_rate), subtype="PCM_16")
        report["cases"].append({
            "source": source.name,
            "output": destination.name,
            "duration_seconds": round(len(audio) / int(sample_rate), 3),
            "peak": round(peak, 6),
            "conversion_seconds": round(time.time() - started, 3),
        })

    (args.output_dir / "seed_vc_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
