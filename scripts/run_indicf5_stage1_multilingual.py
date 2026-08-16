from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
from contextlib import redirect_stderr, redirect_stdout
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel

from run_dhee_baseline import prepare_compatible_snapshot, validate_waveform
from run_indicf5_diagnostic import (
    COMPAT_MODEL_ID,
    MODEL_ID,
    load_compat_model_direct,
    repair_and_prove_weights,
    resolve_reference_text,
    safe_normalize,
    tamil_clean,
)

SAMPLE_RATE = 24000
TARGETS = {
    "english": {
        "text": "This is my natural English voice, speaking clearly and smoothly.",
        "seeds": (11, 29),
    },
    "mixed": {
        "text": "வணக்கம், this is my voice. இன்று Kubernetes சரியாக வேலை செய்கிறது.",
        "seeds": (17, 31),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 English and mixed-language IndicF5 gate")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-text-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def transcribe_output(
    verifier: WhisperModel,
    path: Path,
    *,
    language: str | None,
    beam_size: int,
) -> str:
    segments, _ = verifier.transcribe(
        str(path),
        language=language,
        beam_size=beam_size,
        vad_filter=False,
        condition_on_previous_text=False,
        temperature=0,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def latin_words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.casefold())


def verify_english(verifier: WhisperModel, path: Path, target: str) -> dict[str, object]:
    primary = transcribe_output(verifier, path, language="en", beam_size=5)
    secondary = transcribe_output(verifier, path, language="en", beam_size=1)
    expected = safe_normalize(target)
    observed = safe_normalize(primary)
    secondary_norm = safe_normalize(secondary)
    overall = SequenceMatcher(None, expected, observed).ratio() if expected else 0.0
    consensus = SequenceMatcher(None, observed, secondary_norm).ratio() if observed else 0.0
    length_ratio = len(observed) / max(1, len(expected))
    duration = float(sf.info(path).duration)
    accepted = bool(
        2.0 <= duration <= 8.0
        and overall >= 0.88
        and consensus >= 0.88
        and 0.80 <= length_ratio <= 1.25
    )
    return {
        "duration_seconds": round(duration, 3),
        "target_char_count": len(expected),
        "asr_char_count": len(observed),
        "overall_asr_similarity": round(overall, 4),
        "decode_consensus_similarity": round(consensus, 4),
        "asr_length_ratio": round(length_ratio, 4),
        "accepted": accepted,
    }


def verify_mixed(verifier: WhisperModel, path: Path, target: str) -> dict[str, object]:
    primary = transcribe_output(verifier, path, language=None, beam_size=5)
    secondary = transcribe_output(verifier, path, language=None, beam_size=1)

    expected = safe_normalize(target)
    observed = safe_normalize(primary)
    secondary_norm = safe_normalize(secondary)
    overall = SequenceMatcher(None, expected, observed).ratio() if expected else 0.0
    consensus = SequenceMatcher(None, observed, secondary_norm).ratio() if observed else 0.0
    length_ratio = len(observed) / max(1, len(expected))

    expected_tamil = tamil_clean(target)
    observed_tamil = tamil_clean(primary)
    tamil_similarity = (
        SequenceMatcher(None, expected_tamil, observed_tamil).ratio()
        if expected_tamil
        else 0.0
    )

    observed_latin = set(latin_words(primary))
    required_keywords = {"voice", "kubernetes"}
    keyword_coverage = len(required_keywords & observed_latin) / len(required_keywords)
    duration = float(sf.info(path).duration)

    accepted = bool(
        2.0 <= duration <= 9.0
        and overall >= 0.72
        and consensus >= 0.80
        and 0.70 <= length_ratio <= 1.35
        and tamil_similarity >= 0.78
        and keyword_coverage == 1.0
    )
    return {
        "duration_seconds": round(duration, 3),
        "target_char_count": len(expected),
        "asr_char_count": len(observed),
        "overall_asr_similarity": round(overall, 4),
        "decode_consensus_similarity": round(consensus, 4),
        "asr_length_ratio": round(length_ratio, 4),
        "tamil_asr_similarity": round(tamil_similarity, 4),
        "required_english_keyword_coverage": round(keyword_coverage, 4),
        "accepted": accepted,
    }


def generate_candidate(
    *,
    model,
    verifier: WhisperModel,
    reference: Path,
    effective_ref_text: str,
    output_dir: Path,
    target_name: str,
    target_text: str,
    seed: int,
) -> tuple[Path, dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.perf_counter()
    captured = io.StringIO()
    try:
        with redirect_stdout(captured), redirect_stderr(captured):
            generated = model(
                target_text,
                ref_audio_path=str(reference),
                ref_text=effective_ref_text,
            )
    except Exception as exc:
        raise RuntimeError(
            f"IndicF5 {target_name} generation failed: {type(exc).__name__}"
        ) from None
    generation_seconds = time.perf_counter() - started

    waveform = np.asarray(generated)
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    waveform = waveform.astype(np.float32).reshape(-1)
    signal = validate_waveform(waveform, f"stage1_{target_name}_seed_{seed}")

    candidate = output_dir / f".{target_name}_seed_{seed}.wav"
    sf.write(candidate, waveform, SAMPLE_RATE, subtype="PCM_16")
    if target_name == "english":
        verification = verify_english(verifier, candidate, target_text)
    else:
        verification = verify_mixed(verifier, candidate, target_text)

    metrics: dict[str, object] = {
        "seed": seed,
        "generation_seconds": round(generation_seconds, 3),
        "signal_metrics": signal,
        "verification": verification,
    }
    print(
        "stage1_candidate_metrics="
        + json.dumps(
            {"target": target_name, **metrics},
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return candidate, metrics


def candidate_score(metrics: dict[str, object]) -> float:
    verification = metrics["verification"]
    assert isinstance(verification, dict)
    return float(verification["overall_asr_similarity"])


def main() -> None:
    args = parse_args()
    reference = Path(args.reference)
    protected_text = Path(args.reference_text_file).read_text(encoding="utf-8").strip()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    verifier = WhisperModel(
        "large-v3-turbo",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
    )

    alignment, effective_ref_text = resolve_reference_text(verifier, reference, protected_text)
    if not bool(alignment["accepted"]):
        raise SystemExit("Reference transcript alignment/consensus gate failed")

    snapshot = prepare_compatible_snapshot(
        MODEL_ID,
        COMPAT_MODEL_ID,
        os.environ.get("HF_TOKEN") or None,
    )
    started = time.perf_counter()
    model = load_compat_model_direct(snapshot)
    model_load_seconds = time.perf_counter() - started

    weights = repair_and_prove_weights(model, snapshot)
    if not weights["effective_weights_verified"]:
        raise SystemExit("Effective model weight validation failed")

    report: dict[str, object] = {
        "status": "PENDING_HUMAN_LISTENING_REVIEW",
        "strategy": "STAGE1_TWO_SEED_ENGLISH_AND_MIXED",
        "model_id": MODEL_ID,
        "reference_text_source": alignment["mode"],
        "reference_alignment": alignment,
        "weight_validation": weights,
        "model_load_seconds": round(model_load_seconds, 3),
        "targets": {},
    }

    all_temporary: list[Path] = []
    try:
        for target_name, config in TARGETS.items():
            target_text = str(config["text"])
            seeds = tuple(config["seeds"])
            candidates: list[tuple[Path, dict[str, object]]] = []
            for seed in seeds:
                path, metrics = generate_candidate(
                    model=model,
                    verifier=verifier,
                    reference=reference,
                    effective_ref_text=effective_ref_text,
                    output_dir=output_dir,
                    target_name=target_name,
                    target_text=target_text,
                    seed=int(seed),
                )
                all_temporary.append(path)
                candidates.append((path, metrics))
                verification = metrics["verification"]
                assert isinstance(verification, dict)
                if not bool(verification["accepted"]):
                    raise SystemExit(f"{target_name} rejected by strict ASR gate")

            chosen_path, chosen_metrics = max(candidates, key=lambda item: candidate_score(item[1]))
            final_path = output_dir / f"{target_name}_listening_sample.wav"
            chosen_path.replace(final_path)
            all_temporary.remove(chosen_path)
            for path, _ in candidates:
                if path != chosen_path:
                    path.unlink(missing_ok=True)
                    if path in all_temporary:
                        all_temporary.remove(path)

            targets_report = report["targets"]
            assert isinstance(targets_report, dict)
            targets_report[target_name] = {
                "two_seed_gate_passed": True,
                "candidate_metrics": [metrics for _, metrics in candidates],
                "selected_seed": int(chosen_metrics["seed"]),
                "selected_output": final_path.name,
            }

        (output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("stage1_multilingual_safe_report=" + json.dumps(report, ensure_ascii=True, sort_keys=True))
    except BaseException:
        for path in all_temporary:
            path.unlink(missing_ok=True)
        for name in ("english_listening_sample.wav", "mixed_listening_sample.wav"):
            (output_dir / name).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
