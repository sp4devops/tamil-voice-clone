from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from run_dhee_baseline import prepare_compatible_snapshot, validate_waveform
from run_indicf5_diagnostic import (
    COMPAT_MODEL_ID,
    MODEL_ID,
    load_compat_model_direct,
    repair_and_prove_weights,
)
from run_indicf5_stage1_multilingual import (
    SAMPLE_RATE,
    TARGETS,
    synthesize_mixed_spans,
    synthesize_with_unicode_duration,
)

SELECTED_SEEDS = {"english": 11, "mixed": 31}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTS-only Stage 1 IndicF5 runtime benchmark")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--effective-reference-text-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = Path(args.reference)
    effective_ref_text = Path(args.effective_reference_text_file).read_text(encoding="utf-8").strip()
    if not effective_ref_text:
        raise SystemExit("Effective reference transcript is empty")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = prepare_compatible_snapshot(
        MODEL_ID,
        COMPAT_MODEL_ID,
        os.environ.get("HF_TOKEN") or None,
    )
    started = time.perf_counter()
    model = load_compat_model_direct(snapshot)
    model_load_seconds = time.perf_counter() - started

    weights = repair_and_prove_weights(model, snapshot)
    if not bool(weights["effective_weights_verified"]):
        raise SystemExit("Effective model weight validation failed")

    targets: dict[str, object] = {}
    transient_wavs: list[Path] = []
    try:
        for target_name in ("english", "mixed"):
            seed = SELECTED_SEEDS[target_name]
            torch.manual_seed(seed)
            np.random.seed(seed)
            target_text = str(TARGETS[target_name]["text"])

            started = time.perf_counter()
            if target_name == "mixed":
                waveform, duration_plan = synthesize_mixed_spans(
                    model=model,
                    reference=reference,
                    effective_ref_text=effective_ref_text,
                )
            else:
                waveform, duration_plan = synthesize_with_unicode_duration(
                    model=model,
                    reference=reference,
                    effective_ref_text=effective_ref_text,
                    target_text=target_text,
                )
            generation_seconds = time.perf_counter() - started

            signal = validate_waveform(waveform, f"runtime_{target_name}")
            wav_path = output_dir / f".{target_name}_runtime.wav"
            sf.write(wav_path, waveform, SAMPLE_RATE, subtype="PCM_16")
            transient_wavs.append(wav_path)
            duration_seconds = float(sf.info(wav_path).duration)

            targets[target_name] = {
                "seed": seed,
                "generation_seconds": round(generation_seconds, 3),
                "duration_seconds": round(duration_seconds, 3),
                "real_time_factor": round(generation_seconds / duration_seconds, 3) if duration_seconds else None,
                "signal_metrics": signal,
                "duration_plan": duration_plan,
            }
            print(
                "runtime_target_metrics="
                + json.dumps(
                    {
                        "target": target_name,
                        "seed": seed,
                        "generation_seconds": round(generation_seconds, 3),
                        "duration_seconds": round(duration_seconds, 3),
                    },
                    sort_keys=True,
                )
            )

        report = {
            "status": "PASS",
            "strategy": "STAGE1_TTS_ONLY_SELECTED_SEEDS",
            "architecture": platform.machine(),
            "model_id": MODEL_ID,
            "model_load_seconds": round(model_load_seconds, 3),
            "effective_weights_verified": True,
            "targets": targets,
            "note": "No Whisper verifier model is instantiated in this runtime benchmark.",
        }
        (output_dir / "runtime_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("runtime_benchmark=PASS")
    finally:
        for wav_path in transient_wavs:
            wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
