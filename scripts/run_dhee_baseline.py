from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download, snapshot_download

SAMPLE_RATE = 24000
COMPATIBLE_CONFIG = {
    "architectures": ["INF5Model"],
    "auto_map": {
        "AutoConfig": "model.INF5Config",
        "AutoModel": "model.INF5Model",
    },
    "ckpt_path": "checkpoints/model_best.pt",
    "model_type": "inf5",
    "remove_sil": True,
    "speed": 1.0,
    "torch_dtype": "float32",
    "transformers_version": "4.49.0",
    "vocab_path": "checkpoints/vocab.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dhee-Indic-F5 voice_001 baseline")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--cases", default="eval/voice_001_baseline_cases.json")
    parser.add_argument("--output-dir", default="outputs/voice_001_dhee")
    parser.add_argument("--model-id", default="dheeyantra/dhee-indic-f5")
    parser.add_argument("--compat-model-id", default="AbhishekDelMundu/IndicF5")
    return parser.parse_args()


def prepare_compatible_snapshot(model_id: str, compat_model_id: str, token: str | None) -> Path:
    """Repair Dhee metadata while retaining its weights and IndicF5 vocabulary.

    The compatibility model supplies a loader that correctly strips the
    ``ema_model`` / ``_orig_mod`` safetensors prefixes before loading weights.
    The previous loader accepted those keys with ``strict=False`` but silently
    left the acoustic model effectively uninitialised.
    """

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            token=token,
            allow_patterns=[
                "model.safetensors",
                "checkpoints/*",
                "f5_tts/*",
                "f5_tts/**/*",
            ],
        )
    )
    model_py = Path(
        hf_hub_download(
            repo_id=compat_model_id,
            filename="model.py",
            token=token,
        )
    )
    vocab_file = Path(
        hf_hub_download(
            repo_id=compat_model_id,
            filename="checkpoints/vocab.txt",
            token=token,
        )
    )

    local_model_py = snapshot / "model.py"
    local_vocab_path = snapshot / "checkpoints" / "vocab.txt"
    local_vocab_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_py, local_model_py)
    shutil.copyfile(vocab_file, local_vocab_path)

    source = local_model_py.read_text(encoding="utf-8")
    vocab_lookup = (
        'vocab_path = hf_hub_download(config.name_or_path, '
        'filename="checkpoints/vocab.txt")'
    )
    if vocab_lookup not in source:
        raise RuntimeError("IndicF5 compatibility loader vocabulary lookup changed")
    source = source.replace(
        vocab_lookup,
        'vocab_path = os.environ["INDICF5_VOCAB_PATH"]',
    )
    source = source.replace(
        "hf_hub_download(config.name_or_path,",
        "hf_hub_download(os.environ.get('INDICF5_WEIGHTS_REPO', config.name_or_path),",
    )
    local_model_py.write_text(source, encoding="utf-8")

    (snapshot / "config.json").write_text(
        json.dumps(COMPATIBLE_CONFIG, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.environ["INDICF5_WEIGHTS_REPO"] = model_id
    os.environ["INDICF5_VOCAB_PATH"] = str(local_vocab_path.resolve())
    return snapshot


def validate_waveform(waveform: np.ndarray, case_id: str) -> dict[str, float | int]:
    """Reject silence, DC-only output, NaNs and implausibly short waveforms."""

    if waveform.size < SAMPLE_RATE // 2:
        raise RuntimeError(f"{case_id}: waveform is shorter than 0.5 seconds")
    if not np.isfinite(waveform).all():
        raise RuntimeError(f"{case_id}: waveform contains NaN or infinity")

    peak = float(np.max(np.abs(waveform)))
    mean = float(np.mean(waveform))
    ac_rms = float(np.std(waveform))
    peak_to_peak = float(np.ptp(waveform))
    unique_samples = int(np.unique(np.round(waveform, decimals=6)).size)

    if ac_rms < 1e-4 or peak_to_peak < 1e-3 or unique_samples < 64:
        raise RuntimeError(
            f"{case_id}: invalid non-speech waveform "
            f"(ac_rms={ac_rms:.8f}, peak_to_peak={peak_to_peak:.8f}, "
            f"unique_samples={unique_samples}, mean={mean:.8f})"
        )

    return {
        "peak": peak,
        "mean": mean,
        "ac_rms": ac_rms,
        "peak_to_peak": peak_to_peak,
        "unique_samples": unique_samples,
    }


def main() -> None:
    from transformers import AutoModel

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    token = os.environ.get("HF_TOKEN") or None

    started = time.perf_counter()
    snapshot = prepare_compatible_snapshot(args.model_id, args.compat_model_id, token)
    model = AutoModel.from_pretrained(
        snapshot,
        trust_remote_code=True,
        local_files_only=True,
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
        signal_metrics = validate_waveform(waveform, case["id"])

        output_path = output_dir / f"{case['id']}.wav"
        sf.write(output_path, waveform, samplerate=SAMPLE_RATE)

        duration = len(waveform) / SAMPLE_RATE
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
                **signal_metrics,
            }
        )

    report = {
        "voice_id": payload["voice_id"],
        "model_id": args.model_id,
        "compat_model_id": args.compat_model_id,
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
