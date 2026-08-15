from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from transformers import AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dhee-Indic-F5 voice_001 baseline")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--cases", default="eval/voice_001_baseline_cases.json")
    parser.add_argument("--output-dir", default="outputs/voice_001_dhee")
    parser.add_argument("--model-id", default="dheeyantra/dhee-indic-f5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    token = os.environ.get("HF_TOKEN") or None

    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        token=token,
    )
    load_seconds = time.perf_counter() - started

    results: list[dict[str, object]] = []
    for case in payload["cases"]:
        case_started = time.perf_counter()
        audio = model(
            case["text"],
            ref_audio_path=args.reference,
            ref_text=args.reference_text,
        )
        waveform = np.asarray(audio)
        if waveform.dtype == np.int16:
            waveform = waveform.astype(np.float32) / 32768.0
        waveform = waveform.astype(np.float32).reshape(-1)
        output_path = output_dir / f"{case['id']}.wav"
        sf.write(output_path, waveform, samplerate=24000)

        duration = len(waveform) / 24000
        elapsed = time.perf_counter() - case_started
        results.append(
            {
                "id": case["id"],
                "language": case["language"],
                "text": case["text"],
                "path": str(output_path),
                "duration_seconds": duration,
                "generation_seconds": elapsed,
                "real_time_factor": elapsed / duration if duration else None,
                "peak": float(np.max(np.abs(waveform))) if waveform.size else 0.0,
            }
        )

    report = {
        "voice_id": payload["voice_id"],
        "model_id": args.model_id,
        "reference": Path(args.reference).name,
        "model_load_seconds": load_seconds,
        "results": results,
        "acceptance_status": "PENDING_SPEAKER_AND_LISTENING_REVIEW",
    }
    (output_dir / "baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
