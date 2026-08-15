from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from jiwer import wer
from speechbrain.inference.speaker import EncoderClassifier
from transformers import pipeline

TARGET_GLOBAL_COSINE = 0.72
TARGET_MIN_WINDOW_COSINE = 0.64
TARGET_WINDOW_SPREAD = 0.16
TARGET_ADJACENT_DROP = 0.12
TARGET_WER = 0.08
SPEAKER_SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate voice_001 synthesis")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--cases", default="eval/voice_001_baseline_cases.json")
    parser.add_argument("--output", default="outputs/voice_001_eval/report.json")
    parser.add_argument(
        "--speaker-model",
        default="speechbrain/spkrec-ecapa-voxceleb",
    )
    parser.add_argument("--asr-model", default="openai/whisper-base")
    return parser.parse_args()


def load_mono(path: Path, target_sample_rate: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
    peak = waveform.abs().max().item()
    if peak > 0:
        waveform = waveform / peak * 0.95
    return waveform


def embedding(classifier: EncoderClassifier, waveform: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        value = classifier.encode_batch(waveform).reshape(-1).float()
    return torch.nn.functional.normalize(value, dim=0)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(left, right).item())


def window_embeddings(
    classifier: EncoderClassifier,
    waveform: torch.Tensor,
    window_seconds: float = 2.5,
    hop_seconds: float = 1.25,
) -> list[tuple[float, float, torch.Tensor]]:
    signal = waveform.reshape(-1)
    window = int(window_seconds * SPEAKER_SAMPLE_RATE)
    hop = int(hop_seconds * SPEAKER_SAMPLE_RATE)
    if signal.numel() <= window:
        if float(torch.sqrt(torch.mean(signal**2)).item()) < 0.005:
            return []
        return [(0.0, signal.numel() / SPEAKER_SAMPLE_RATE, embedding(classifier, signal.unsqueeze(0)))]

    starts = list(range(0, signal.numel() - window + 1, hop))
    final_start = signal.numel() - window
    if starts[-1] != final_start:
        starts.append(final_start)

    result: list[tuple[float, float, torch.Tensor]] = []
    for start in starts:
        chunk = signal[start : start + window]
        rms = float(torch.sqrt(torch.mean(chunk**2)).item())
        if rms < 0.005:
            continue
        result.append(
            (
                start / SPEAKER_SAMPLE_RATE,
                (start + window) / SPEAKER_SAMPLE_RATE,
                embedding(classifier, chunk.unsqueeze(0)),
            )
        )
    return result


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^0-9a-z\u0b80-\u0bff]+", " ", text)
    return " ".join(text.split())


def waveform_diagnostics(path: Path) -> dict[str, float | int]:
    waveform, sample_rate = sf.read(path, dtype="float32")
    waveform = np.asarray(waveform).reshape(-1)
    return {
        "sample_rate": int(sample_rate),
        "duration_seconds": float(waveform.size / sample_rate),
        "peak": float(np.max(np.abs(waveform))) if waveform.size else 0.0,
        "rms": float(np.sqrt(np.mean(waveform**2))) if waveform.size else 0.0,
        "dc_mean": float(np.mean(waveform)) if waveform.size else 0.0,
        "clipped_sample_ratio": float(np.mean(np.abs(waveform) >= 0.999)) if waveform.size else 0.0,
        "unique_samples_6dp": int(np.unique(np.round(waveform, decimals=6)).size),
    }


def main() -> None:
    args = parse_args()
    cases_payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    samples_dir = Path(args.samples_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    speaker = EncoderClassifier.from_hparams(
        source=args.speaker_model,
        savedir="models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    reference_waveform = load_mono(Path(args.reference), SPEAKER_SAMPLE_RATE)
    reference_embedding = embedding(speaker, reference_waveform)

    asr = pipeline(
        task="automatic-speech-recognition",
        model=args.asr_model,
        device=-1,
        torch_dtype=torch.float32,
    )

    results: list[dict[str, object]] = []
    for case in cases_payload["cases"]:
        sample_path = samples_dir / f"{case['id']}.wav"
        generated = load_mono(sample_path, SPEAKER_SAMPLE_RATE)
        full_embedding = embedding(speaker, generated)
        full_similarity = cosine(reference_embedding, full_embedding)

        window_rows = []
        window_scores = []
        for start, end, current_embedding in window_embeddings(speaker, generated):
            score = cosine(reference_embedding, current_embedding)
            window_scores.append(score)
            window_rows.append(
                {
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                    "reference_cosine": score,
                }
            )

        min_window = min(window_scores) if window_scores else full_similarity
        max_window = max(window_scores) if window_scores else full_similarity
        spread = max_window - min_window
        adjacent_drop = (
            max(abs(right - left) for left, right in zip(window_scores, window_scores[1:]))
            if len(window_scores) > 1
            else 0.0
        )

        language = "english" if case["language"] == "en-IN" else "tamil"
        asr_result = asr(
            str(sample_path),
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        transcription = str(asr_result.get("text", "")).strip()
        normalized_reference = normalize_text(case["text"])
        normalized_transcription = normalize_text(transcription)
        error_rate = (
            float(wer(normalized_reference, normalized_transcription))
            if normalized_reference
            else math.nan
        )

        identity_pass = (
            full_similarity >= TARGET_GLOBAL_COSINE
            and min_window >= TARGET_MIN_WINDOW_COSINE
            and spread <= TARGET_WINDOW_SPREAD
            and adjacent_drop <= TARGET_ADJACENT_DROP
        )
        asr_target_pass = error_rate <= TARGET_WER

        results.append(
            {
                "id": case["id"],
                "language": case["language"],
                "expected_text": case["text"],
                "asr_transcription": transcription,
                "normalized_expected": normalized_reference,
                "normalized_transcription": normalized_transcription,
                "word_error_rate": error_rate,
                "speaker_similarity": {
                    "global_cosine": full_similarity,
                    "min_window_cosine": min_window,
                    "max_window_cosine": max_window,
                    "window_spread": spread,
                    "max_adjacent_window_change": adjacent_drop,
                    "windows": window_rows,
                },
                "waveform": waveform_diagnostics(sample_path),
                "identity_thresholds_pass": identity_pass,
                "asr_target_pass": asr_target_pass,
                "human_listening_required": True,
            }
        )

    report = {
        "voice_id": cases_payload["voice_id"],
        "speaker_model": args.speaker_model,
        "asr_model": args.asr_model,
        "thresholds": {
            "global_cosine_min": TARGET_GLOBAL_COSINE,
            "short_window_cosine_min": TARGET_MIN_WINDOW_COSINE,
            "window_spread_max": TARGET_WINDOW_SPREAD,
            "adjacent_window_change_max": TARGET_ADJACENT_DROP,
            "word_error_rate_max": TARGET_WER,
        },
        "results": results,
        "all_identity_thresholds_pass": all(
            bool(item["identity_thresholds_pass"]) for item in results
        ),
        "all_asr_targets_pass": all(bool(item["asr_target_pass"]) for item in results),
        "final_status": "PENDING_HUMAN_LISTENING_REVIEW",
        "notes": [
            "ECAPA cosine is an independent automated speaker-similarity diagnostic, not proof of identity.",
            "Whisper-base WER can over-penalize Tamil and code-switched speech; listen before rejecting pronunciation solely from ASR.",
        ],
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
